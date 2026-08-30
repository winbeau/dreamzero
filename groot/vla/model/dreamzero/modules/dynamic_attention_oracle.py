"""Chunked dense-attention analysis for dynamic sparse M1 supervision.

The released DreamZero geometry makes a materialized per-layer attention
tensor impractical: ``40 heads x 1,785 queries x 7,920 video keys`` exceeds
half a billion elements.  This module computes exact dense softmax statistics
in query chunks and retains only per-head aggregates plus ranked key profiles.

It is deliberately independent from the executor.  Oracle collection must not
change the model output, and its overhead is never included in sparse latency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_KEEP_RATIOS = (1.0, 0.75, 0.50, 0.35, 0.25, 0.20, 0.10)
DEFAULT_TOP_P_THRESHOLDS = (0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class OracleThresholds:
    """Quality constraints used to derive the minimum supervised budget."""

    mass: float = 0.90
    output_cosine: float = 0.999
    output_relative_l2: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.mass <= 1.0:
            raise ValueError("mass threshold must lie in (0, 1]")
        if not -1.0 <= self.output_cosine <= 1.0:
            raise ValueError("output_cosine threshold must lie in [-1, 1]")
        if self.output_relative_l2 < 0.0:
            raise ValueError("output_relative_l2 must be non-negative")


def _validated_ratios(keep_ratios: Sequence[float]) -> tuple[float, ...]:
    ratios = tuple(float(ratio) for ratio in keep_ratios)
    if not ratios:
        raise ValueError("keep_ratios must not be empty")
    if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("keep ratios must lie in (0, 1]")
    if len(set(ratios)) != len(ratios):
        raise ValueError("keep ratios must be unique")
    return tuple(sorted(ratios, reverse=True))


def deterministic_query_sample_indices(
    query_length: int,
    max_queries: int | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return stable, spatially distributed query positions.

    ``max_queries=None`` records every query.  Otherwise positions are selected
    by integer linspace, including both sequence endpoints.  No RNG state or
    task-specific rule is involved.
    """

    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if max_queries is None or max_queries >= query_length:
        return torch.arange(query_length, device=device, dtype=torch.long)
    if max_queries <= 0:
        raise ValueError("max_queries must be positive or None")
    if max_queries == 1:
        return torch.tensor([query_length // 2], device=device, dtype=torch.long)
    return torch.linspace(
        0,
        query_length - 1,
        steps=max_queries,
        device=device,
        dtype=torch.float64,
    ).round().to(torch.long)


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    # Quantiles are analysis-only and computed in FP32 for stable reports.
    return torch.quantile(values.float(), q, dim=-1)


def analyze_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    keep_ratios: Sequence[float] = DEFAULT_KEEP_RATIOS,
    top_p_thresholds: Sequence[float] = DEFAULT_TOP_P_THRESHOLDS,
    query_indices: torch.Tensor | None = None,
    query_chunk_size: int = 16,
    support_ratio: float = 0.20,
) -> dict[str, torch.Tensor | tuple[float, ...]]:
    """Compute exact per-head dense and top-k attention statistics.

    Args:
        query: RoPE-applied queries ``[B, Lq, H, D]``.
        key: RoPE-applied video keys ``[B, Lk, H, D]``.
        value: Video values ``[B, Lk, H, Dv]``.
        query_indices: Optional original query positions to analyze.  The
            selected positions remain explicit in the returned record.

    Returns:
        Tensor-valued metrics are organized as ``[R, H]`` for keep-ratio
        metrics and ``[P, H]`` for top-p counts.  ``ranked_key_indices`` is
        ``[H, ceil(Lk * support_ratio)]`` and provides the nested profile used
        by later fixed-shape budget buckets.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, L, H, D]")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("query/key/value batch, key length, and head shapes differ")
    if query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3]:
        raise ValueError("query and key head dimensions differ")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if not 0.0 < support_ratio <= 1.0:
        raise ValueError("support_ratio must lie in (0, 1]")

    ratios = _validated_ratios(keep_ratios)
    top_p = tuple(float(threshold) for threshold in top_p_thresholds)
    if any(not 0.0 < threshold <= 1.0 for threshold in top_p):
        raise ValueError("top-p thresholds must lie in (0, 1]")

    batch, query_length, num_heads, head_dim = query.shape
    key_length = key.shape[1]
    if query_indices is None:
        query_indices = torch.arange(
            query_length,
            device=query.device,
            dtype=torch.long,
        )
    if query_indices.ndim != 1 or query_indices.numel() == 0:
        raise ValueError("query_indices must be a non-empty 1D tensor")
    if query_indices.device != query.device:
        query_indices = query_indices.to(query.device)
    if int(query_indices.min()) < 0 or int(query_indices.max()) >= query_length:
        raise IndexError("query_indices are outside the query sequence")

    sampled_query = query.index_select(1, query_indices)
    qh = sampled_query.permute(0, 2, 1, 3).float()
    kh = key.permute(0, 2, 1, 3).float()
    vh = value.permute(0, 2, 1, 3).float()
    scale = head_dim**-0.5

    per_ratio_mass: list[list[torch.Tensor]] = [[] for _ in ratios]
    per_ratio_cosine: list[list[torch.Tensor]] = [[] for _ in ratios]
    per_ratio_relative_l2: list[list[torch.Tensor]] = [[] for _ in ratios]
    per_top_p_count: list[list[torch.Tensor]] = [[] for _ in top_p]
    entropy_chunks: list[torch.Tensor] = []
    max_mass_chunks: list[torch.Tensor] = []
    key_importance = torch.zeros(
        (batch, num_heads, key_length),
        device=query.device,
        dtype=torch.float32,
    )

    for start in range(0, qh.shape[2], query_chunk_size):
        stop = min(start + query_chunk_size, qh.shape[2])
        q_chunk = qh[:, :, start:stop]
        logits = torch.matmul(q_chunk, kh.transpose(-1, -2)) * scale
        probability = logits.softmax(dim=-1)
        dense_output = torch.matmul(probability, vh)
        key_importance.add_(probability.sum(dim=2))

        sorted_probability, sorted_indices = probability.sort(
            dim=-1,
            descending=True,
        )
        cumulative_mass = sorted_probability.cumsum(dim=-1)
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum(dim=-1)
        entropy_chunks.append(entropy / math.log(max(2, key_length)))
        max_mass_chunks.append(sorted_probability[..., 0])

        for top_p_index, threshold in enumerate(top_p):
            count = ((cumulative_mass < threshold).sum(dim=-1) + 1).clamp_max(
                key_length
            )
            per_top_p_count[top_p_index].append(count)

        for ratio_index, ratio in enumerate(ratios):
            keep = max(1, min(key_length, round(key_length * ratio)))
            retained_mass = cumulative_mass[..., keep - 1]
            per_ratio_mass[ratio_index].append(retained_mass)
            if keep == key_length:
                sparse_output = dense_output
            else:
                retained_probability = torch.zeros_like(probability)
                retained_probability.scatter_(
                    -1,
                    sorted_indices[..., :keep],
                    sorted_probability[..., :keep],
                )
                retained_probability.div_(retained_mass.unsqueeze(-1))
                sparse_output = torch.matmul(retained_probability, vh)
            cosine = F.cosine_similarity(dense_output, sparse_output, dim=-1)
            relative_l2 = (
                torch.linalg.vector_norm(sparse_output - dense_output, dim=-1)
                / torch.linalg.vector_norm(dense_output, dim=-1).clamp_min(1e-12)
            )
            per_ratio_cosine[ratio_index].append(cosine)
            per_ratio_relative_l2[ratio_index].append(relative_l2)

    # Flatten batch/query observations but preserve heads for M1 labels.
    def stack_observations(chunks: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(chunks, dim=-1).permute(1, 0, 2).flatten(1)

    ratio_mass = torch.stack(
        [stack_observations(chunks) for chunks in per_ratio_mass]
    )
    ratio_cosine = torch.stack(
        [stack_observations(chunks) for chunks in per_ratio_cosine]
    )
    ratio_relative_l2 = torch.stack(
        [stack_observations(chunks) for chunks in per_ratio_relative_l2]
    )
    top_p_count = torch.stack(
        [stack_observations(chunks).float() for chunks in per_top_p_count]
    )
    entropy_observations = stack_observations(entropy_chunks)
    max_mass_observations = stack_observations(max_mass_chunks)

    normalized_importance = key_importance / float(batch * query_indices.numel())
    mean_key_importance = normalized_importance.mean(dim=0)
    support_length = max(1, min(key_length, round(key_length * support_ratio)))
    ranked_key_indices = mean_key_importance.topk(
        k=support_length,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices

    return {
        "keep_ratios": ratios,
        "top_p_thresholds": top_p,
        "query_indices": query_indices,
        "mass_mean": ratio_mass.mean(dim=-1),
        "mass_p05": _quantile(ratio_mass, 0.05),
        "mass_min": ratio_mass.min(dim=-1).values,
        "output_cosine_mean": ratio_cosine.mean(dim=-1),
        "output_cosine_p05": _quantile(ratio_cosine, 0.05),
        "output_cosine_min": ratio_cosine.min(dim=-1).values,
        "output_relative_l2_mean": ratio_relative_l2.mean(dim=-1),
        "output_relative_l2_p95": _quantile(ratio_relative_l2, 0.95),
        "output_relative_l2_max": ratio_relative_l2.max(dim=-1).values,
        "top_p_token_count_mean": top_p_count.mean(dim=-1),
        "top_p_token_count_p95": _quantile(top_p_count, 0.95),
        "normalized_entropy_mean": entropy_observations.mean(dim=-1),
        "max_attention_mass_mean": max_mass_observations.mean(dim=-1),
        "key_importance": mean_key_importance,
        "ranked_key_indices": ranked_key_indices,
    }


def minimum_oracle_budget(
    statistics: dict[str, torch.Tensor | tuple[float, ...]],
    thresholds: OracleThresholds = OracleThresholds(),
) -> torch.Tensor:
    """Select the smallest measured keep ratio satisfying every threshold.

    The decision uses p05 retained mass, p05 output cosine, and p95 relative
    L2.  Heads with no passing sparse bucket fall back to the 100% bucket.
    """

    ratios = statistics["keep_ratios"]
    if not isinstance(ratios, tuple):
        raise TypeError("statistics keep_ratios must be a tuple")
    mass = statistics["mass_p05"]
    cosine = statistics["output_cosine_p05"]
    relative_l2 = statistics["output_relative_l2_p95"]
    if not all(isinstance(value, torch.Tensor) for value in (mass, cosine, relative_l2)):
        raise TypeError("statistics quality metrics must be tensors")

    passing = (
        (mass >= thresholds.mass)
        & (cosine >= thresholds.output_cosine)
        & (relative_l2 <= thresholds.output_relative_l2)
    )
    ratio_tensor = torch.tensor(ratios, device=mass.device, dtype=torch.float32)
    expanded = ratio_tensor[:, None].expand_as(mass)
    candidates = torch.where(passing, expanded, torch.full_like(expanded, 2.0))
    selected = candidates.min(dim=0).values
    return torch.where(selected > 1.0, torch.ones_like(selected), selected)


def head_importance_correlation(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Pearson correlation of two per-head key-importance profiles."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("importance tensors must share shape [H, K]")
    first_centered = first.float() - first.float().mean(dim=-1, keepdim=True)
    second_centered = second.float() - second.float().mean(dim=-1, keepdim=True)
    numerator = (first_centered * second_centered).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(first_centered, dim=-1)
        * torch.linalg.vector_norm(second_centered, dim=-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def support_turnover(
    previous_indices: torch.Tensor,
    current_indices: torch.Tensor,
    *,
    num_keys: int,
) -> torch.Tensor:
    """Return per-head fraction of the current support not seen previously."""

    if previous_indices.shape != current_indices.shape or previous_indices.ndim != 2:
        raise ValueError("support indices must share shape [H, K]")
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")
    if previous_indices.numel() == 0:
        return torch.zeros(
            previous_indices.shape[0],
            device=previous_indices.device,
            dtype=torch.float32,
        )
    previous_mask = torch.zeros(
        previous_indices.shape[0],
        num_keys,
        device=previous_indices.device,
        dtype=torch.bool,
    )
    previous_mask.scatter_(1, previous_indices, True)
    overlap = previous_mask.gather(1, current_indices).float().mean(dim=-1)
    return 1.0 - overlap


@dataclass(frozen=True)
class DenseAttentionOracleConfig:
    """Runtime controls for offline dense-attention evidence collection."""

    output_dir: Path
    rank: int = 0
    keep_ratios: tuple[float, ...] = DEFAULT_KEEP_RATIOS
    top_p_thresholds: tuple[float, ...] = DEFAULT_TOP_P_THRESHOLDS
    max_video_queries: int | None = 32
    max_action_queries: int | None = None
    query_chunk_size: int = 4
    support_ratio: float = 0.75
    layer_indices: tuple[int, ...] = ()
    task_id: str | None = None
    trajectory_stage: str | None = None

    def __post_init__(self) -> None:
        _validated_ratios(self.keep_ratios)
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.max_video_queries is not None and self.max_video_queries <= 0:
            raise ValueError("max_video_queries must be positive or None")
        if self.max_action_queries is not None and self.max_action_queries <= 0:
            raise ValueError("max_action_queries must be positive or None")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if not 0.0 < self.support_ratio <= 1.0:
            raise ValueError("support_ratio must lie in (0, 1]")
        if any(index < 0 for index in self.layer_indices):
            raise ValueError("layer_indices must be non-negative")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be unique")


def _tensor_json(value: torch.Tensor) -> list:
    return value.detach().float().cpu().tolist()


def _attention_statistics_record(
    statistics: dict[str, torch.Tensor | tuple[float, ...]],
) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in statistics.items():
        if key in {"key_importance", "ranked_key_indices", "query_indices"}:
            continue
        if isinstance(value, torch.Tensor):
            record[key] = _tensor_json(value)
        else:
            record[key] = list(value)
    record["sampled_query_indices"] = (
        statistics["query_indices"].detach().cpu().tolist()  # type: ignore[union-attr]
    )
    return record


class DenseAttentionOracleCollector:
    """Collect per-step/layer/head Oracle features without changing outputs."""

    schema_version = 2

    def __init__(self, config: DenseAttentionOracleConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        request_pattern = re.compile(
            rf"^rank{re.escape(str(self.config.rank))}_request(\d+)\.jsonl$"
        )
        existing_request_indices = [
            int(match.group(1))
            for path in self.config.output_dir.glob(
                f"rank{self.config.rank}_request*.jsonl"
            )
            if (match := request_pattern.match(path.name)) is not None
        ]
        self.request_index = max(existing_request_indices, default=-1)
        self.request_metadata: dict[str, object] = {}
        self.next_request_metadata: dict[str, object] = {}
        self.step_context: dict[str, int] | None = None
        self.records: list[dict[str, object]] = []
        self.support_profiles: dict[str, torch.Tensor] = {}
        self.previous_support: dict[tuple[int, str, str], torch.Tensor] = {}
        self.cfg_branch: str | None = None
        self.last_flush_paths: tuple[Path, Path] | None = None

    def set_next_request_metadata(
        self,
        *,
        task_id: str | None = None,
        trajectory_stage: str | None = None,
        sample_metadata: dict[str, object] | None = None,
    ) -> None:
        if self.records or self.step_context is not None:
            raise RuntimeError("cannot change Oracle metadata during an active request")
        self.next_request_metadata = {
            "task_id": task_id,
            "trajectory_stage": trajectory_stage,
            "sample_metadata": dict(sample_metadata or {}),
        }

    @property
    def active(self) -> bool:
        return self.request_index >= 0 and self.step_context is not None

    def begin_request(
        self,
        *,
        current_start_frame: int,
        instruction: object | None = None,
        task_id: str | None = None,
        trajectory_stage: str | None = None,
    ) -> None:
        if self.records:
            raise RuntimeError("flush the previous Oracle request before starting another")
        self.request_index += 1
        next_metadata = self.next_request_metadata
        self.next_request_metadata = {}
        instruction_digest = None
        if instruction is not None:
            if isinstance(instruction, torch.Tensor):
                instruction_bytes = instruction.detach().cpu().numpy().tobytes()
            else:
                instruction_bytes = str(instruction).encode("utf-8")
            instruction_digest = hashlib.sha256(instruction_bytes).hexdigest()
        self.request_metadata = {
            "request_index": self.request_index,
            "current_start_frame": int(current_start_frame),
            "instruction_sha256": instruction_digest,
            "task_id": (
                task_id
                if task_id is not None
                else next_metadata.get("task_id", self.config.task_id)
            ),
            "trajectory_stage": (
                trajectory_stage
                if trajectory_stage is not None
                else next_metadata.get(
                    "trajectory_stage", self.config.trajectory_stage
                )
            ),
            "sample_metadata": next_metadata.get("sample_metadata", {}),
        }
        self.step_context = None
        self.previous_support.clear()
        self.cfg_branch = None

    def set_cfg_branch(self, branch: str | None) -> None:
        if branch is not None and branch not in {"conditional", "unconditional"}:
            raise ValueError(f"Unsupported CFG branch: {branch}")
        self.cfg_branch = branch

    def set_step(
        self,
        *,
        scheduler_index: int,
        dit_index: int,
        scheduler_steps: int,
        timestep: int | torch.Tensor,
    ) -> None:
        if self.request_index < 0:
            raise RuntimeError("begin_request must be called before set_step")
        if isinstance(timestep, torch.Tensor):
            timestep = int(timestep.detach().flatten()[0].cpu())
        self.step_context = {
            "scheduler_index": int(scheduler_index),
            "dit_index": int(dit_index),
            "scheduler_steps": int(scheduler_steps),
            "timestep": int(timestep),
        }

    def _turnover(
        self,
        layer_index: int,
        query_kind: str,
        support: torch.Tensor,
        num_keys: int,
    ) -> torch.Tensor:
        cache_key = (layer_index, query_kind, self.cfg_branch or "single")
        previous = self.previous_support.get(cache_key)
        if previous is None:
            turnover = torch.zeros(
                support.shape[0],
                device=support.device,
                dtype=torch.float32,
            )
        else:
            turnover = support_turnover(previous, support, num_keys=num_keys)
        self.previous_support[cache_key] = support.detach()
        return turnover

    @torch.inference_mode()
    def observe(
        self,
        *,
        layer_index: int,
        video_query: torch.Tensor,
        action_query: torch.Tensor,
        video_key: torch.Tensor,
        video_value: torch.Tensor,
    ) -> None:
        if not self.active:
            return
        if self.config.layer_indices and layer_index not in self.config.layer_indices:
            return
        video_query_indices = deterministic_query_sample_indices(
            video_query.shape[1],
            self.config.max_video_queries,
            device=video_query.device,
        )
        video_statistics = analyze_dense_attention(
            video_query,
            video_key,
            video_value,
            keep_ratios=self.config.keep_ratios,
            top_p_thresholds=self.config.top_p_thresholds,
            query_indices=video_query_indices,
            query_chunk_size=self.config.query_chunk_size,
            support_ratio=self.config.support_ratio,
        )
        action_query_indices = deterministic_query_sample_indices(
            action_query.shape[1],
            self.config.max_action_queries,
            device=action_query.device,
        )
        action_statistics = analyze_dense_attention(
            action_query,
            video_key,
            video_value,
            keep_ratios=self.config.keep_ratios,
            top_p_thresholds=self.config.top_p_thresholds,
            query_indices=action_query_indices,
            query_chunk_size=self.config.query_chunk_size,
            support_ratio=self.config.support_ratio,
        )
        video_budget = minimum_oracle_budget(video_statistics)
        action_budget = minimum_oracle_budget(action_statistics)
        video_support = video_statistics["ranked_key_indices"]
        action_support = action_statistics["ranked_key_indices"]
        video_importance = video_statistics["key_importance"]
        action_importance = action_statistics["key_importance"]
        assert isinstance(video_support, torch.Tensor)
        assert isinstance(action_support, torch.Tensor)
        assert isinstance(video_importance, torch.Tensor)
        assert isinstance(action_importance, torch.Tensor)

        video_turnover = self._turnover(
            layer_index,
            "video",
            video_support,
            video_key.shape[1],
        )
        action_turnover = self._turnover(
            layer_index,
            "action",
            action_support,
            video_key.shape[1],
        )
        cfg_branch = self.cfg_branch or "single"
        branch_suffix = "" if cfg_branch == "single" else f"_b{cfg_branch}"
        profile_prefix = (
            f"r{self.config.rank}_req{self.request_index:06d}_"
            f"d{self.step_context['dit_index']:02d}_l{layer_index:02d}"
            f"{branch_suffix}"
        )
        # 7,920 keys fit in uint16.  Keep the ranking externally, never in Git.
        support_dtype = torch.uint16 if video_key.shape[1] <= 65535 else torch.int32
        self.support_profiles[f"{profile_prefix}_video"] = (
            video_support.detach().to(device="cpu", dtype=support_dtype)
        )
        self.support_profiles[f"{profile_prefix}_action"] = (
            action_support.detach().to(device="cpu", dtype=support_dtype)
        )

        self.records.append(
            {
                "schema_version": self.schema_version,
                "rank": self.config.rank,
                **self.request_metadata,
                **self.step_context,
                "cfg_branch": cfg_branch,
                "layer_index": int(layer_index),
                "num_heads": int(video_query.shape[2]),
                "num_video_queries": int(video_query.shape[1]),
                "num_sampled_video_queries": int(video_query_indices.numel()),
                "num_action_queries": int(action_query.shape[1]),
                "num_sampled_action_queries": int(action_query_indices.numel()),
                "num_video_keys": int(video_key.shape[1]),
                "video": _attention_statistics_record(video_statistics),
                "action": _attention_statistics_record(action_statistics),
                "video_oracle_min_keep_ratio": _tensor_json(video_budget),
                "action_oracle_min_keep_ratio": _tensor_json(action_budget),
                "video_support_turnover": _tensor_json(video_turnover),
                "action_support_turnover": _tensor_json(action_turnover),
                "qa_qv_key_importance_correlation": _tensor_json(
                    head_importance_correlation(action_importance, video_importance)
                ),
            }
        )

    def flush_request(self) -> tuple[Path, Path] | None:
        if self.request_index < 0 or not self.records:
            self.step_context = None
            return None
        stem = f"rank{self.config.rank}_request{self.request_index:06d}"
        jsonl_path = self.config.output_dir / f"{stem}.jsonl"
        profiles_path = self.config.output_dir / f"{stem}_profiles.pt"
        if jsonl_path.exists() or profiles_path.exists():
            raise FileExistsError(f"refusing to overwrite Oracle request {stem}")
        jsonl_path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in self.records)
        )
        torch.save(self.support_profiles, profiles_path)
        self.records.clear()
        self.support_profiles.clear()
        self.previous_support.clear()
        self.step_context = None
        self.request_metadata = {}
        self.cfg_branch = None
        self.last_flush_paths = (jsonl_path, profiles_path)
        return self.last_flush_paths
