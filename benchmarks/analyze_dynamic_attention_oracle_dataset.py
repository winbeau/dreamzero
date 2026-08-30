"""Aggregate real DROID Dense Oracle captures into M1 data and paper figures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


QUERY_KINDS = ("video", "action")
CFG_BRANCHES = ("conditional", "unconditional")
KEEP_RATIOS = (1.0, 0.75, 0.50, 0.35, 0.25, 0.20, 0.10)
TOP_P_THRESHOLDS = (0.50, 0.75, 0.90, 0.95)
SCHEDULER_INDICES = (0, 1, 2, 6, 10, 13, 14, 15)


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def discover_passed_requests(root: Path) -> list[dict]:
    requests = []
    for results_path in sorted(root.rglob("request_results.jsonl")):
        requests.extend(record for record in _read_jsonl(results_path) if record.get("passed"))
    unique = {}
    for request in requests:
        key = request["request_key"]
        if key in unique:
            raise ValueError(f"Duplicate completed request: {key}")
        unique[key] = request
    return [unique[key] for key in sorted(unique)]


def bootstrap_law_summary(
    request_matrices: np.ndarray,
    episode_ids: np.ndarray,
    *,
    repeats: int = 200,
    seed: int = 20260830,
) -> dict[str, object]:
    """Bootstrap timestep/layer differences using source episode as the unit."""

    if request_matrices.ndim != 4:
        raise ValueError("request_matrices must have shape [R, Q, T, L]")
    unique_episodes = np.unique(episode_ids)
    episode_to_indices = {
        episode: np.flatnonzero(episode_ids == episode) for episode in unique_episodes
    }
    rng = np.random.default_rng(seed)
    time_differences = np.empty((repeats, request_matrices.shape[1]), dtype=np.float64)
    layer_differences = np.empty_like(time_differences)
    for repeat in range(repeats):
        sampled_episodes = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        sampled_indices = np.concatenate(
            [episode_to_indices[episode] for episode in sampled_episodes]
        )
        sample = request_matrices[sampled_indices]
        time_differences[repeat] = np.nanmean(sample[:, :, 0, :], axis=(0, 2)) - np.nanmean(
            sample[:, :, -1, :], axis=(0, 2)
        )
        layer_count = sample.shape[-1]
        early_stop = max(1, layer_count // 3)
        late_start = max(early_stop + 1, 2 * layer_count // 3)
        layer_differences[repeat] = np.nanmean(
            sample[:, :, :, :early_stop], axis=(0, 2, 3)
        ) - np.nanmean(sample[:, :, :, late_start:], axis=(0, 2, 3))

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "positive_fraction": float(np.mean(values > 0.0)),
        }

    return {
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "source_episode_index",
        "interpretation": (
            "A positive difference supports a lower Oracle keep ratio at later "
            "timesteps/layers; a CI crossing zero does not support the law."
        ),
        "early_minus_late_timestep_budget": {
            query_kind: interval(time_differences[:, query_index])
            for query_index, query_kind in enumerate(QUERY_KINDS)
        },
        "early_minus_late_layer_budget": {
            query_kind: interval(layer_differences[:, query_index])
            for query_index, query_kind in enumerate(QUERY_KINDS)
        },
    }


class HeadTableWriter:
    def __init__(self, path: Path, batch_rows: int = 100_000) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required to write the M1 head table") from error
        self.pa = pa
        self.pq = pq
        self.path = path
        self.batch_rows = batch_rows
        self.columns: dict[str, list] = defaultdict(list)
        self.row_count = 0
        self.writer = None

    def append(self, row: dict) -> None:
        for key, value in row.items():
            self.columns[key].append(value)
        self.row_count += 1
        if len(next(iter(self.columns.values()))) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.columns or not next(iter(self.columns.values())):
            return
        table = self.pa.Table.from_pydict(dict(self.columns))
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = self.pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
            )
        self.writer.write_table(table)
        self.columns = defaultdict(list)

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


def _head_row(record: dict, query_kind: str, head: int) -> dict:
    query = record[query_kind]
    metadata = record["sample_metadata"]
    row = {
        "request_key": metadata["request_key"],
        "task_id": record["task_id"],
        "split": metadata["split"],
        "source_episode_index": metadata["source_episode_index"],
        "subset_episode_index": metadata["subset_episode_index"],
        "trajectory_stage": record["trajectory_stage"],
        "trajectory_fraction": metadata["trajectory_fraction"],
        "trajectory_step": metadata["trajectory_step"],
        "trajectory_length": metadata["trajectory_length"],
        "length_bucket": metadata["length_bucket"],
        "instruction_index": metadata["instruction_index"],
        "state_l2": metadata["state_l2"],
        "state_abs_mean": metadata["state_abs_mean"],
        "action_l2": metadata["action_l2"],
        "action_std": metadata["action_std"],
        "action_temporal_delta_l2": metadata["action_temporal_delta_l2"],
        "scheduler_index": record["scheduler_index"],
        "dit_index": record["dit_index"],
        "timestep": record["timestep"],
        "layer_index": record["layer_index"],
        "head_index": head,
        "cfg_branch": record["cfg_branch"],
        "query_kind": query_kind,
        "num_video_keys": record["num_video_keys"],
        "num_sampled_queries": record[f"num_sampled_{query_kind}_queries"],
        "oracle_min_keep_ratio": record[f"{query_kind}_oracle_min_keep_ratio"][head],
        "support_turnover": record[f"{query_kind}_support_turnover"][head],
        "vv_output_change_cosine": record[
            f"{query_kind}_vv_output_change_cosine"
        ][head],
        "vv_output_change_relative_l2": record[
            f"{query_kind}_vv_output_change_relative_l2"
        ][head],
        "qa_qv_key_importance_correlation": record[
            "qa_qv_key_importance_correlation"
        ][head],
        "normalized_entropy_mean": query["normalized_entropy_mean"][head],
        "max_attention_mass_mean": query["max_attention_mass_mean"][head],
    }
    for ratio_index, ratio in enumerate(KEEP_RATIOS):
        suffix = f"r{int(round(ratio * 100)):03d}"
        for metric in (
            "mass_mean",
            "mass_p05",
            "mass_min",
            "output_cosine_mean",
            "output_cosine_p05",
            "output_cosine_min",
            "output_relative_l2_mean",
            "output_relative_l2_p95",
            "output_relative_l2_max",
        ):
            row[f"{metric}_{suffix}"] = query[metric][ratio_index][head]
    for threshold_index, threshold in enumerate(TOP_P_THRESHOLDS):
        suffix = f"p{int(round(threshold * 100)):02d}"
        row[f"top_p_token_count_mean_{suffix}"] = query[
            "top_p_token_count_mean"
        ][threshold_index][head]
        row[f"top_p_token_count_p95_{suffix}"] = query[
            "top_p_token_count_p95"
        ][threshold_index][head]
    return row


def _save_heatmap(matrix: np.ndarray, path: Path, title: str, colorbar: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Real DiT index")
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_head_panels(cube: np.ndarray, path: Path, title: str) -> None:
    figure, axes = plt.subplots(5, 8, figsize=(20, 12), sharex=True, sharey=True)
    vmin = float(np.nanmin(cube))
    vmax = float(np.nanmax(cube))
    image = None
    for head, axis in enumerate(axes.flat):
        image = axis.imshow(
            cube[:, :, head],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"H{head}", fontsize=8)
        axis.tick_params(labelsize=6)
    figure.suptitle(title)
    figure.supxlabel("Transformer layer")
    figure.supylabel("Real DiT index")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.65, label="Keep ratio")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _compact_m1_rows(record_arrays: dict[str, np.ndarray], metadata: dict) -> Iterator[dict]:
    budget = record_arrays["budget"]
    turnover = record_arrays["turnover"]
    vv_cosine = record_arrays["vv_cosine"]
    vv_relative_l2 = record_arrays["vv_relative_l2"]
    entropy = record_arrays["entropy"]
    max_mass = record_arrays["max_mass"]
    correlation = record_arrays["correlation"]
    mass_p05 = record_arrays["mass_p05"]
    cosine_p05 = record_arrays["cosine_p05"]
    relative_l2_p95 = record_arrays["relative_l2_p95"]
    for dit_index in range(8):
        previous_index = max(0, dit_index - 1)
        previous_two_index = max(0, dit_index - 2)
        for layer_index in range(40):
            for head_index in range(40):
                current = (slice(None), slice(None), dit_index, layer_index, head_index)
                previous = (
                    slice(None),
                    slice(None),
                    previous_index,
                    layer_index,
                    head_index,
                )
                previous_two = (
                    slice(None),
                    slice(None),
                    previous_two_index,
                    layer_index,
                    head_index,
                )
                row = {
                    **metadata,
                    "dit_index": dit_index,
                    "scheduler_index": SCHEDULER_INDICES[dit_index],
                    "diffusion_timestep": int(
                        record_arrays["diffusion_timestep"][dit_index]
                    ),
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "timestep_position": dit_index / 7.0,
                    "layer_depth": layer_index / 39.0,
                    "oracle_min_keep_ratio": float(np.nanmax(budget[current])),
                    "video_oracle_min_keep_ratio": float(
                        np.nanmax(budget[0, :, dit_index, layer_index, head_index])
                    ),
                    "action_oracle_min_keep_ratio": float(
                        np.nanmax(budget[1, :, dit_index, layer_index, head_index])
                    ),
                    "previous_oracle_min_keep_ratio": float(
                        np.nanmax(budget[previous])
                    ),
                    "previous_two_oracle_min_keep_ratio": float(
                        np.nanmax(budget[previous_two])
                    ),
                    "support_turnover_mean": float(np.nanmean(turnover[current])),
                    "support_turnover_max": float(np.nanmax(turnover[current])),
                    "previous_support_turnover_max": float(
                        np.nanmax(turnover[previous])
                    ),
                    "video_support_turnover_max": float(
                        np.nanmax(turnover[0, :, dit_index, layer_index, head_index])
                    ),
                    "action_support_turnover_max": float(
                        np.nanmax(turnover[1, :, dit_index, layer_index, head_index])
                    ),
                    "vv_output_change_cosine_min": float(
                        np.nanmin(vv_cosine[current])
                    ),
                    "vv_output_change_relative_l2_max": float(
                        np.nanmax(vv_relative_l2[current])
                    ),
                    "previous_vv_output_change_relative_l2_max": float(
                        np.nanmax(vv_relative_l2[previous])
                    ),
                    "previous_two_vv_output_change_relative_l2_max": float(
                        np.nanmax(vv_relative_l2[previous_two])
                    ),
                    "normalized_entropy_mean": float(np.nanmean(entropy[current])),
                    "normalized_entropy_max": float(np.nanmax(entropy[current])),
                    "previous_normalized_entropy_mean": float(
                        np.nanmean(entropy[previous])
                    ),
                    "max_attention_mass_mean": float(np.nanmean(max_mass[current])),
                    "max_attention_mass_max": float(np.nanmax(max_mass[current])),
                    "previous_max_attention_mass_mean": float(
                        np.nanmean(max_mass[previous])
                    ),
                    "qa_qv_key_importance_correlation_mean": float(
                        np.nanmean(correlation[:, dit_index, layer_index, head_index])
                    ),
                    "qa_qv_key_importance_correlation_min": float(
                        np.nanmin(correlation[:, dit_index, layer_index, head_index])
                    ),
                    "previous_qa_qv_key_importance_correlation_mean": float(
                        np.nanmean(
                            correlation[:, previous_index, layer_index, head_index]
                        )
                    ),
                }
                for ratio_index, ratio in enumerate(KEEP_RATIOS):
                    suffix = f"r{int(round(ratio * 100)):03d}"
                    quality_index = (
                        slice(None),
                        slice(None),
                        dit_index,
                        layer_index,
                        head_index,
                        ratio_index,
                    )
                    row[f"worst_mass_p05_{suffix}"] = float(
                        np.nanmin(mass_p05[quality_index])
                    )
                    row[f"worst_output_cosine_p05_{suffix}"] = float(
                        np.nanmin(cosine_p05[quality_index])
                    )
                    row[f"worst_output_relative_l2_p95_{suffix}"] = float(
                        np.nanmax(relative_l2_p95[quality_index])
                    )
                    row[f"video_mass_p05_{suffix}"] = float(
                        np.nanmin(
                            mass_p05[
                                0, :, dit_index, layer_index, head_index, ratio_index
                            ]
                        )
                    )
                    row[f"action_mass_p05_{suffix}"] = float(
                        np.nanmin(
                            mass_p05[
                                1, :, dit_index, layer_index, head_index, ratio_index
                            ]
                        )
                    )
                yield row


def analyze(root: Path, output_dir: Path, expected_requests: int, bootstrap_repeats: int) -> dict:
    requests = discover_passed_requests(root)
    if len(requests) != expected_requests:
        raise ValueError(f"Expected {expected_requests} requests, found {len(requests)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = HeadTableWriter(output_dir / "m1_oracle_heads.parquet")
    compact_writer = HeadTableWriter(output_dir / "m1_dynamic_samples.parquet")

    shape = (len(QUERY_KINDS), len(CFG_BRANCHES), 8, 40, 40)
    budget_sum = np.zeros(shape, dtype=np.float64)
    budget_square_sum = np.zeros(shape, dtype=np.float64)
    dense_sum = np.zeros(shape, dtype=np.float64)
    turnover_sum = np.zeros(shape, dtype=np.float64)
    mass20_sum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)
    request_matrices = []
    episode_ids = []
    request_summaries = []
    split_counts = Counter()
    stage_counts = Counter()

    try:
        for request in requests:
            capture_path = Path(request["capture_jsonl"])
            trajectory_stage = request.get(
                "trajectory_stage", request["request_key"].rsplit("_", 1)[-1]
            )
            matrix_sum = np.zeros((len(QUERY_KINDS), 8, 40), dtype=np.float64)
            matrix_count = np.zeros_like(matrix_sum, dtype=np.int64)
            request_shape = (len(QUERY_KINDS), len(CFG_BRANCHES), 8, 40, 40)
            request_arrays = {
                name: np.full(request_shape, np.nan, dtype=np.float32)
                for name in (
                    "budget",
                    "turnover",
                    "vv_cosine",
                    "vv_relative_l2",
                    "entropy",
                    "max_mass",
                )
            }
            request_arrays["correlation"] = np.full(
                (len(CFG_BRANCHES), 8, 40, 40), np.nan, dtype=np.float32
            )
            request_arrays["diffusion_timestep"] = np.full(8, -1, dtype=np.int32)
            quality_shape = (*request_shape, len(KEEP_RATIOS))
            for name in ("mass_p05", "cosine_p05", "relative_l2_p95"):
                request_arrays[name] = np.full(quality_shape, np.nan, dtype=np.float32)
            record_count = 0
            for record in _read_jsonl(capture_path):
                record_count += 1
                branch_index = CFG_BRANCHES.index(record["cfg_branch"])
                dit_index = int(record["dit_index"])
                layer_index = int(record["layer_index"])
                request_arrays["diffusion_timestep"][dit_index] = int(
                    record["timestep"]
                )
                for query_index, query_kind in enumerate(QUERY_KINDS):
                    budgets = np.asarray(
                        record[f"{query_kind}_oracle_min_keep_ratio"], dtype=np.float64
                    )
                    turnover = np.asarray(
                        record[f"{query_kind}_support_turnover"], dtype=np.float64
                    )
                    mass20 = np.asarray(record[query_kind]["mass_mean"][5], dtype=np.float64)
                    target = (query_index, branch_index, dit_index, layer_index)
                    budget_sum[target] += budgets
                    budget_square_sum[target] += budgets**2
                    dense_sum[target] += budgets == 1.0
                    turnover_sum[target] += turnover
                    mass20_sum[target] += mass20
                    count[target] += 1
                    matrix_sum[query_index, dit_index, layer_index] += budgets.sum()
                    matrix_count[query_index, dit_index, layer_index] += budgets.size
                    request_arrays["budget"][target] = budgets
                    request_arrays["turnover"][target] = turnover
                    request_arrays["vv_cosine"][target] = np.asarray(
                        record[f"{query_kind}_vv_output_change_cosine"], dtype=np.float32
                    )
                    request_arrays["vv_relative_l2"][target] = np.asarray(
                        record[f"{query_kind}_vv_output_change_relative_l2"],
                        dtype=np.float32,
                    )
                    request_arrays["entropy"][target] = np.asarray(
                        record[query_kind]["normalized_entropy_mean"], dtype=np.float32
                    )
                    request_arrays["max_mass"][target] = np.asarray(
                        record[query_kind]["max_attention_mass_mean"], dtype=np.float32
                    )
                    request_arrays["mass_p05"][target] = np.asarray(
                        record[query_kind]["mass_p05"], dtype=np.float32
                    ).T
                    request_arrays["cosine_p05"][target] = np.asarray(
                        record[query_kind]["output_cosine_p05"], dtype=np.float32
                    ).T
                    request_arrays["relative_l2_p95"][target] = np.asarray(
                        record[query_kind]["output_relative_l2_p95"], dtype=np.float32
                    ).T
                    for head in range(40):
                        writer.append(_head_row(record, query_kind, head))
                request_arrays["correlation"][
                    branch_index, dit_index, layer_index
                ] = np.asarray(
                    record["qa_qv_key_importance_correlation"], dtype=np.float32
                )
            if record_count != 640:
                raise ValueError(f"{request['request_key']} has {record_count} records")
            compact_metadata = {
                "request_key": request["request_key"],
                "task_id": next(_read_jsonl(capture_path))["task_id"],
                "split": request["split"],
                "source_episode_index": request["source_episode_index"],
                "subset_episode_index": request["subset_episode_index"],
                "trajectory_stage": trajectory_stage,
                "trajectory_fraction": request["trajectory_fraction"],
                "trajectory_step": request["trajectory_step"],
                "trajectory_length": request["trajectory_length"],
                "length_bucket": request["length_bucket"],
                "instruction_index": request["instruction_index"],
                "state_l2": request["state_l2"],
                "state_abs_mean": request["state_abs_mean"],
                "action_l2": request["action_l2"],
                "action_std": request["action_std"],
                "action_temporal_delta_l2": request["action_temporal_delta_l2"],
            }
            for compact_row in _compact_m1_rows(request_arrays, compact_metadata):
                compact_writer.append(compact_row)
            request_matrix = np.divide(
                matrix_sum,
                matrix_count,
                out=np.full_like(matrix_sum, np.nan),
                where=matrix_count > 0,
            )
            request_matrices.append(request_matrix)
            episode_ids.append(int(request["source_episode_index"]))
            split_counts[request["split"]] += 1
            stage_counts[trajectory_stage] += 1
            request_summaries.append(
                {
                    "request_key": request["request_key"],
                    "source_episode_index": request["source_episode_index"],
                    "split": request["split"],
                    "trajectory_stage": trajectory_stage,
                    "mean_video_oracle_budget": float(np.mean(request_matrix[0])),
                    "mean_action_oracle_budget": float(np.mean(request_matrix[1])),
                }
            )
    finally:
        writer.close()
        compact_writer.close()

    mean_budget = np.divide(
        budget_sum,
        count,
        out=np.full_like(budget_sum, np.nan),
        where=count > 0,
    )
    variance_budget = np.divide(
        budget_square_sum,
        count,
        out=np.full_like(budget_sum, np.nan),
        where=count > 0,
    ) - mean_budget**2
    task_std = np.sqrt(np.maximum(variance_budget, 0.0))
    dense_rate = np.divide(dense_sum, count, where=count > 0)
    mean_turnover = np.divide(turnover_sum, count, where=count > 0)
    mean_mass20 = np.divide(mass20_sum, count, where=count > 0)

    np.savez_compressed(
        output_dir / "oracle_budget_cube.npz",
        mean_budget=mean_budget,
        task_std=task_std,
        dense_fallback_rate=dense_rate,
        mean_support_turnover=mean_turnover,
        mean_mass_retention_at_20pct=mean_mass20,
        query_kinds=np.asarray(QUERY_KINDS),
        cfg_branches=np.asarray(CFG_BRANCHES),
    )
    with (output_dir / "request_budget_summary.jsonl").open("w", encoding="utf-8") as handle:
        for request_summary in request_summaries:
            handle.write(json.dumps(request_summary) + "\n")

    for query_index, query_kind in enumerate(QUERY_KINDS):
        mean_heatmap = np.nanmean(mean_budget[query_index], axis=(0, 3))
        dense_heatmap = np.nanmean(dense_rate[query_index], axis=(0, 3))
        std_heatmap = np.nanmean(task_std[query_index], axis=(0, 3))
        mass_heatmap = np.nanmean(mean_mass20[query_index], axis=(0, 3))
        _save_heatmap(
            mean_heatmap,
            output_dir / f"{query_kind}_oracle_budget_timestep_layer.png",
            f"{query_kind.title()} Oracle minimum keep ratio",
            "Mean keep ratio",
        )
        _save_heatmap(
            dense_heatmap,
            output_dir / f"{query_kind}_dense_fallback_timestep_layer.png",
            f"{query_kind.title()} Dense fallback rate",
            "Fraction of heads at 100%",
        )
        _save_heatmap(
            std_heatmap,
            output_dir / f"{query_kind}_task_std_timestep_layer.png",
            f"{query_kind.title()} cross-request Oracle budget standard deviation",
            "Budget standard deviation",
        )
        _save_heatmap(
            mass_heatmap,
            output_dir / f"{query_kind}_mass20_timestep_layer.png",
            f"{query_kind.title()} Dense attention mass retained at 20%",
            "Retained mass",
        )
        head_cube = np.nanmean(mean_budget[query_index], axis=0)
        _save_head_panels(
            head_cube,
            output_dir / f"{query_kind}_oracle_budget_per_head.png",
            f"{query_kind.title()} Oracle budget: complete timestep/layer map per head",
        )

    request_matrix_array = np.stack(request_matrices)
    law_summary = bootstrap_law_summary(
        request_matrix_array,
        np.asarray(episode_ids),
        repeats=bootstrap_repeats,
    )
    for query_index, query_kind in enumerate(QUERY_KINDS):
        global_matrix = np.nanmean(request_matrix_array[:, query_index], axis=0)
        law_summary[f"{query_kind}_mean_budget_by_dit"] = np.nanmean(
            global_matrix, axis=1
        ).tolist()
        law_summary[f"{query_kind}_mean_budget_by_layer"] = np.nanmean(
            global_matrix, axis=0
        ).tolist()
        law_summary[f"{query_kind}_timestep_adjacent_violations"] = int(
            np.sum(np.diff(np.nanmean(global_matrix, axis=1)) > 0.0)
        )
        law_summary[f"{query_kind}_layer_adjacent_violations"] = int(
            np.sum(np.diff(np.nanmean(global_matrix, axis=0)) > 0.0)
        )
    (output_dir / "oracle_law_summary.json").write_text(
        json.dumps(law_summary, indent=2) + "\n"
    )

    summary = {
        "request_count": len(requests),
        "episode_count": len(set(episode_ids)),
        "split_request_counts": dict(split_counts),
        "stage_request_counts": dict(stage_counts),
        "capture_record_count": len(requests) * 640,
        "head_row_count": writer.row_count,
        "expected_head_row_count": len(requests) * 640 * 2 * 40,
        "m1_head_table": str(output_dir / "m1_oracle_heads.parquet"),
        "m1_dynamic_samples": str(output_dir / "m1_dynamic_samples.parquet"),
        "budget_cube": str(output_dir / "oracle_budget_cube.npz"),
        "law_summary": str(output_dir / "oracle_law_summary.json"),
        "compact_row_count": compact_writer.row_count,
        "expected_compact_row_count": len(requests) * 8 * 40 * 40,
        "passed": (
            writer.row_count == len(requests) * 640 * 2 * 40
            and compact_writer.row_count == len(requests) * 8 * 40 * 40
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, default=108)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    args = parser.parse_args()
    summary = analyze(
        args.oracle_root,
        args.output_dir,
        args.expected_requests,
        args.bootstrap_repeats,
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
