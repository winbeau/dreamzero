"""Calibrate and evaluate late-step VV linear extrapolation from Dense Oracle traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


DIT_TIMESTEPS = np.asarray([999, 986, 972, 892, 749, 535, 416, 249])
BRANCHES = ("conditional", "unconditional")
MODALITIES = ("video", "action")
LAYER_BUCKETS = {
    "early": range(0, 12),
    "middle": range(12, 28),
    "late": range(28, 40),
}


def fit_linear_alpha(
    current: torch.Tensor,
    previous: torch.Tensor,
    previous_two: torch.Tensor,
) -> torch.Tensor:
    """Least-squares scalar alpha independently for each leading row."""

    if current.shape != previous.shape or current.shape != previous_two.shape:
        raise ValueError("VV tensors must share a shape")
    target = current.float() - previous.float()
    direction = previous.float() - previous_two.float()
    numerator = (target * direction).sum(dim=-1)
    denominator = direction.square().sum(dim=-1)
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    ).clamp(0.0, 2.0)


def extrapolation_metrics(
    current: torch.Tensor,
    previous: torch.Tensor,
    previous_two: torch.Tensor,
    alpha: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row cosine and relative L2 for one linear VV prediction."""

    if current.shape != previous.shape or current.shape != previous_two.shape:
        raise ValueError("VV tensors must share a shape")
    alpha_tensor = torch.as_tensor(
        alpha,
        device=previous.device,
        dtype=torch.float32,
    )
    while alpha_tensor.ndim < previous.ndim:
        alpha_tensor = alpha_tensor.unsqueeze(-1)
    predicted = previous.float() + alpha_tensor * (
        previous.float() - previous_two.float()
    )
    current_float = current.float()
    cosine = F.cosine_similarity(current_float, predicted, dim=-1)
    relative_l2 = (
        torch.linalg.vector_norm(current_float - predicted, dim=-1)
        / torch.linalg.vector_norm(predicted, dim=-1).clamp_min(1e-12)
    )
    return cosine, relative_l2


def select_sentinel_threshold(
    previous_error: np.ndarray,
    current_safe: np.ndarray,
    eligible: np.ndarray,
    *,
    maximum_false_extrapolation_rate: float = 0.01,
) -> dict[str, float | int]:
    """Choose the largest previous-error threshold satisfying a safety rate."""

    if not (
        previous_error.shape == current_safe.shape == eligible.shape
    ):
        raise ValueError("sentinel arrays must share a shape")
    candidate_scores = np.unique(previous_error[eligible & np.isfinite(previous_error)])
    best = {
        "threshold": -1.0,
        "selected": 0,
        "unsafe": 0,
        "false_extrapolation_rate": 0.0,
    }
    for threshold in candidate_scores:
        selected = eligible & (previous_error <= threshold)
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        unsafe_count = int((selected & ~current_safe).sum())
        false_rate = unsafe_count / selected_count
        if (
            false_rate <= maximum_false_extrapolation_rate
            and selected_count >= int(best["selected"])
        ):
            best = {
                "threshold": float(threshold),
                "selected": selected_count,
                "unsafe": unsafe_count,
                "false_extrapolation_rate": float(false_rate),
            }
    return best


def _profile_records(root: Path) -> list[dict[str, object]]:
    records = []
    for profile_path in sorted(root.glob("gpu*_shard*/capture/rank*_request*_profiles.pt")):
        jsonl_path = profile_path.with_name(
            profile_path.name.replace("_profiles.pt", ".jsonl")
        )
        with jsonl_path.open() as handle:
            metadata = json.loads(handle.readline())
        records.append(
            {
                "profile_path": profile_path,
                "rank": int(metadata["rank"]),
                "request_index": int(metadata["request_index"]),
                "request_key": metadata["sample_metadata"]["request_key"],
                "split": metadata["sample_metadata"]["split"],
            }
        )
    if not records:
        raise RuntimeError(f"No Oracle VV profiles found under {root}")
    return records


def _vv_key(
    record: dict[str, object],
    dit_index: int,
    layer_index: int,
    branch: str,
    modality: str,
) -> str:
    return (
        f"r{record['rank']}_req{int(record['request_index']):06d}_"
        f"d{dit_index:02d}_l{layer_index:02d}_b{branch}_{modality}_vv"
    )


def _load_profiles(record: dict[str, object]) -> dict[str, torch.Tensor]:
    return torch.load(
        record["profile_path"],
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )


def _schedule_alpha(dit_index: int) -> float:
    numerator = DIT_TIMESTEPS[dit_index] - DIT_TIMESTEPS[dit_index - 1]
    denominator = DIT_TIMESTEPS[dit_index - 1] - DIT_TIMESTEPS[dit_index - 2]
    if denominator == 0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 2.0))


def _fit_alpha_table(records: Iterable[dict[str, object]]) -> np.ndarray:
    shape = (8, 40, len(BRANCHES), len(MODALITIES), 40)
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    for record in records:
        profiles = _load_profiles(record)
        for dit_index in range(2, 8):
            for layer_index in range(40):
                for branch_index, branch in enumerate(BRANCHES):
                    for modality_index, modality in enumerate(MODALITIES):
                        current = profiles[
                            _vv_key(record, dit_index, layer_index, branch, modality)
                        ].float()
                        previous = profiles[
                            _vv_key(record, dit_index - 1, layer_index, branch, modality)
                        ].float()
                        previous_two = profiles[
                            _vv_key(record, dit_index - 2, layer_index, branch, modality)
                        ].float()
                        target = current - previous
                        direction = previous - previous_two
                        numerator[
                            dit_index,
                            layer_index,
                            branch_index,
                            modality_index,
                        ] += (target * direction).sum(dim=-1).numpy()
                        denominator[
                            dit_index,
                            layer_index,
                            branch_index,
                            modality_index,
                        ] += direction.square().sum(dim=-1).numpy()
        del profiles
    alpha = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.clip(alpha, 0.0, 2.0)


def _empty_metric_arrays(num_requests: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (num_requests, 6, 40, len(BRANCHES), len(MODALITIES), 40)
    return (
        np.full(shape, np.nan, dtype=np.float32),
        np.full(shape, np.nan, dtype=np.float32),
    )


def _evaluate_split(
    records: list[dict[str, object]],
    alpha_table: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[str, int, str, str], dict[str, float]],
]:
    fitted_cosine, fitted_l2 = _empty_metric_arrays(len(records))
    accumulators: dict[tuple[str, int, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "cosine_sum": 0.0,
            "cosine_min": 1.0,
            "l2_sum": 0.0,
            "l2_max": 0.0,
            "safe": 0.0,
        }
    )
    methods = ("reuse", "alpha1", "scheduler", "fitted")
    for request_index, record in enumerate(records):
        profiles = _load_profiles(record)
        for dit_index in range(2, 8):
            schedule_alpha = _schedule_alpha(dit_index)
            for layer_index in range(40):
                layer_bucket = next(
                    name
                    for name, layers in LAYER_BUCKETS.items()
                    if layer_index in layers
                )
                for branch_index, branch in enumerate(BRANCHES):
                    for modality_index, modality in enumerate(MODALITIES):
                        current = profiles[
                            _vv_key(record, dit_index, layer_index, branch, modality)
                        ]
                        previous = profiles[
                            _vv_key(record, dit_index - 1, layer_index, branch, modality)
                        ]
                        previous_two = profiles[
                            _vv_key(record, dit_index - 2, layer_index, branch, modality)
                        ]
                        alpha_values = {
                            "reuse": 0.0,
                            "alpha1": 1.0,
                            "scheduler": schedule_alpha,
                            "fitted": torch.from_numpy(
                                alpha_table[
                                    dit_index,
                                    layer_index,
                                    branch_index,
                                    modality_index,
                                ]
                            ),
                        }
                        for method in methods:
                            cosine, relative_l2 = extrapolation_metrics(
                                current,
                                previous,
                                previous_two,
                                alpha_values[method],
                            )
                            cosine_np = cosine.numpy()
                            l2_np = relative_l2.numpy()
                            if method == "fitted":
                                fitted_cosine[
                                    request_index,
                                    dit_index - 2,
                                    layer_index,
                                    branch_index,
                                    modality_index,
                                ] = cosine_np
                                fitted_l2[
                                    request_index,
                                    dit_index - 2,
                                    layer_index,
                                    branch_index,
                                    modality_index,
                                ] = l2_np
                            for bucket in (layer_bucket, "all"):
                                accumulator = accumulators[
                                    (method, dit_index, modality, bucket)
                                ]
                                accumulator["count"] += cosine_np.size
                                accumulator["cosine_sum"] += float(cosine_np.sum())
                                accumulator["cosine_min"] = min(
                                    accumulator["cosine_min"],
                                    float(cosine_np.min()),
                                )
                                accumulator["l2_sum"] += float(l2_np.sum())
                                accumulator["l2_max"] = max(
                                    accumulator["l2_max"],
                                    float(l2_np.max()),
                                )
                                accumulator["safe"] += float(
                                    ((cosine_np >= 0.999) & (l2_np <= 0.05)).sum()
                                )
        del profiles
    return fitted_cosine, fitted_l2, accumulators


def _render_accumulators(
    split: str,
    accumulators: dict[tuple[str, int, str, str], dict[str, float]],
) -> list[dict[str, object]]:
    rows = []
    for (method, dit_index, modality, layer_bucket), values in sorted(
        accumulators.items()
    ):
        count = values["count"]
        rows.append(
            {
                "split": split,
                "method": method,
                "dit_index": dit_index,
                "timestep": int(DIT_TIMESTEPS[dit_index]),
                "modality": modality,
                "layer_bucket": layer_bucket,
                "count": int(count),
                "cosine_mean": values["cosine_sum"] / count,
                "cosine_min": values["cosine_min"],
                "relative_l2_mean": values["l2_sum"] / count,
                "relative_l2_max": values["l2_max"],
                "safe_fraction": values["safe"] / count,
            }
        )
    return rows


def _cell_gate(
    cosine: np.ndarray,
    relative_l2: np.ndarray,
    *,
    minimum_cosine: float,
    maximum_relative_l2: float,
) -> np.ndarray:
    return (
        np.quantile(cosine, 0.05, axis=0) >= minimum_cosine
    ) & (
        np.quantile(relative_l2, 0.95, axis=0) <= maximum_relative_l2
    )


def _selection_summary(
    name: str,
    cosine: np.ndarray,
    relative_l2: np.ndarray,
    selected: np.ndarray,
    *,
    minimum_cosine: float,
    maximum_relative_l2: float,
) -> dict[str, object]:
    expanded = np.broadcast_to(selected, cosine.shape)
    selected_cosine = cosine[expanded]
    selected_l2 = relative_l2[expanded]
    safe = (
        (selected_cosine >= minimum_cosine)
        & (selected_l2 <= maximum_relative_l2)
    )
    return {
        "name": name,
        "selected_signatures": int(selected_cosine.size),
        "safe_signatures": int(safe.sum()),
        "false_extrapolation_rate": (
            float((~safe).mean()) if safe.size else None
        ),
        "cosine_mean": (
            float(selected_cosine.mean()) if selected_cosine.size else None
        ),
        "cosine_min": (
            float(selected_cosine.min()) if selected_cosine.size else None
        ),
        "relative_l2_mean": (
            float(selected_l2.mean()) if selected_l2.size else None
        ),
        "relative_l2_max": (
            float(selected_l2.max()) if selected_l2.size else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.05)
    parser.add_argument("--late-dit-start", type=int, default=5)
    parser.add_argument("--late-layer-start", type=int, default=28)
    parser.add_argument(
        "--maximum-false-extrapolation-rate",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()

    records = _profile_records(args.oracle_root)
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    if {split: len(rows) for split, rows in by_split.items()} != {
        "train": 72,
        "validation": 18,
        "test": 18,
    }:
        raise RuntimeError("Oracle VV split counts are not 72/18/18")

    alpha_table = _fit_alpha_table(by_split["train"])
    split_metrics = {}
    summary_rows = []
    for split in ("train", "validation", "test"):
        cosine, relative_l2, accumulators = _evaluate_split(
            by_split[split],
            alpha_table,
        )
        split_metrics[split] = (cosine, relative_l2)
        summary_rows.extend(_render_accumulators(split, accumulators))

    train_cosine, train_l2 = split_metrics["train"]
    validation_cosine, validation_l2 = split_metrics["validation"]
    test_cosine, test_l2 = split_metrics["test"]
    train_good = _cell_gate(
        train_cosine,
        train_l2,
        minimum_cosine=args.minimum_cosine,
        maximum_relative_l2=args.maximum_relative_l2,
    )
    validation_good = _cell_gate(
        validation_cosine,
        validation_l2,
        minimum_cosine=args.minimum_cosine,
        maximum_relative_l2=args.maximum_relative_l2,
    )
    eligible = train_good & validation_good
    dit_mask = np.arange(2, 8)[:, None, None, None, None] >= args.late_dit_start
    layer_mask = np.arange(40)[None, :, None, None, None] >= args.late_layer_start
    eligible &= dit_mask & layer_mask

    validation_safe = (
        validation_cosine >= args.minimum_cosine
    ) & (
        validation_l2 <= args.maximum_relative_l2
    )
    validation_previous_l2 = np.full_like(validation_l2, np.inf)
    validation_previous_l2[:, 1:] = validation_l2[:, :-1]
    sentinel = select_sentinel_threshold(
        validation_previous_l2,
        validation_safe,
        np.broadcast_to(eligible, validation_l2.shape),
        maximum_false_extrapolation_rate=args.maximum_false_extrapolation_rate,
    )
    sentinel_threshold = float(sentinel["threshold"])
    test_previous_l2 = np.full_like(test_l2, np.inf)
    test_previous_l2[:, 1:] = test_l2[:, :-1]
    test_sentinel_selected = (
        np.broadcast_to(eligible, test_l2.shape)
        & (test_previous_l2 <= sentinel_threshold)
    )

    candidate_cells = int((dit_mask & layer_mask).sum()) * len(BRANCHES) * len(
        MODALITIES
    ) * 40
    payload = {
        "schema_version": 1,
        "oracle_root": str(args.oracle_root),
        "split_counts": {split: len(rows) for split, rows in by_split.items()},
        "quality_gate": {
            "minimum_cosine": args.minimum_cosine,
            "maximum_relative_l2": args.maximum_relative_l2,
        },
        "late_region": {
            "dit_start": args.late_dit_start,
            "layer_start": args.late_layer_start,
            "candidate_cells": candidate_cells,
            "eligible_cells": int(eligible.sum()),
            "eligible_cell_fraction": float(eligible.sum() / candidate_cells),
        },
        "alpha": {
            "mean_by_dit": {
                str(dit_index): float(alpha_table[dit_index].mean())
                for dit_index in range(2, 8)
            },
            "min": float(alpha_table[2:].min()),
            "max": float(alpha_table[2:].max()),
            "scheduler": {
                str(dit_index): _schedule_alpha(dit_index)
                for dit_index in range(2, 8)
            },
        },
        "validation_selection": _selection_summary(
            "frozen_eligible_cells",
            validation_cosine,
            validation_l2,
            eligible,
            minimum_cosine=args.minimum_cosine,
            maximum_relative_l2=args.maximum_relative_l2,
        ),
        "test_selection": _selection_summary(
            "frozen_eligible_cells",
            test_cosine,
            test_l2,
            eligible,
            minimum_cosine=args.minimum_cosine,
            maximum_relative_l2=args.maximum_relative_l2,
        ),
        "sentinel_calibration": sentinel,
        "test_sentinel_selection": _selection_summary(
            "frozen_eligible_cells_with_previous_error_sentinel",
            test_cosine,
            test_l2,
            test_sentinel_selected,
            minimum_cosine=args.minimum_cosine,
            maximum_relative_l2=args.maximum_relative_l2,
        ),
        "summary_rows": summary_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "alpha_table.npy", alpha_table)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "summary_rows"}, indent=2))


if __name__ == "__main__":
    main()
