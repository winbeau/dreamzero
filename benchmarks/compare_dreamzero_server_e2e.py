from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _measured_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in report["records"] if record["phase"] == "measured"]
    if not records:
        raise ValueError("report contains no measured records")
    return records


def _paired_latencies(
    dense: dict[str, Any],
    sparse: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    dense_records = _measured_records(dense)
    sparse_records = _measured_records(sparse)
    if dense.get("seed") != sparse.get("seed"):
        raise ValueError("dense and sparse reports use different request seeds")
    if len(dense_records) != len(sparse_records):
        raise ValueError("dense and sparse reports have different measured counts")

    for dense_record, sparse_record in zip(dense_records, sparse_records, strict=True):
        if dense_record["request_index"] != sparse_record["request_index"]:
            raise ValueError("dense and sparse measured request indices do not align")
        if dense_record.get("request_key") != sparse_record.get("request_key"):
            raise ValueError("dense and sparse measured request keys do not align")
        if dense_record.get("action_shape") != sparse_record.get("action_shape"):
            raise ValueError("dense and sparse action shapes do not align")

    dense_latency = np.asarray(
        [record["latency_seconds"] for record in dense_records], dtype=np.float64
    )
    sparse_latency = np.asarray(
        [record["latency_seconds"] for record in sparse_records], dtype=np.float64
    )
    if np.any(dense_latency <= 0.0) or np.any(sparse_latency <= 0.0):
        raise ValueError("latencies must be positive")
    return dense_latency, sparse_latency


def _paired_actions(
    dense: dict[str, Any],
    sparse: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    dense_records = _measured_records(dense)
    sparse_records = _measured_records(sparse)
    dense_present = ["action" in record for record in dense_records]
    sparse_present = ["action" in record for record in sparse_records]
    if not any(dense_present) and not any(sparse_present):
        return None
    if not all(dense_present) or not all(sparse_present):
        raise ValueError("dense and sparse reports must both contain every action")
    dense_actions = np.asarray(
        [record["action"] for record in dense_records], dtype=np.float64
    )
    sparse_actions = np.asarray(
        [record["action"] for record in sparse_records], dtype=np.float64
    )
    if dense_actions.shape != sparse_actions.shape:
        raise ValueError("dense and sparse action arrays do not align")
    return dense_actions.reshape(len(dense_records), -1), sparse_actions.reshape(
        len(sparse_records), -1
    )


def compare_reports(
    dense: dict[str, Any],
    sparse: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    dense_latency, sparse_latency = _paired_latencies(dense, sparse)
    paired_speedups = dense_latency / sparse_latency
    log_speedups = np.log(paired_speedups)
    paired_geomean = float(np.exp(log_speedups.mean()))

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        paired_speedups.size,
        size=(bootstrap_samples, paired_speedups.size),
    )
    bootstrap_geomeans = np.exp(log_speedups[bootstrap_indices].mean(axis=1))
    ci_low, ci_high = np.quantile(bootstrap_geomeans, [0.025, 0.975])

    comparison = {
        "dense_label": dense.get("label"),
        "sparse_label": sparse.get("label"),
        "seed": dense.get("seed"),
        "paired_requests": int(paired_speedups.size),
        "dense_mean_seconds": float(dense_latency.mean()),
        "sparse_mean_seconds": float(sparse_latency.mean()),
        "mean_latency_speedup": float(dense_latency.mean() / sparse_latency.mean()),
        "dense_p50_seconds": float(np.median(dense_latency)),
        "sparse_p50_seconds": float(np.median(sparse_latency)),
        "p50_latency_speedup": float(
            np.median(dense_latency) / np.median(sparse_latency)
        ),
        "dense_p90_seconds": float(np.quantile(dense_latency, 0.90)),
        "sparse_p90_seconds": float(np.quantile(sparse_latency, 0.90)),
        "p90_latency_speedup": float(
            np.quantile(dense_latency, 0.90) / np.quantile(sparse_latency, 0.90)
        ),
        "paired_arithmetic_mean_speedup": float(paired_speedups.mean()),
        "paired_geometric_mean_speedup": paired_geomean,
        "paired_geometric_mean_speedup_ci95": [float(ci_low), float(ci_high)],
        "sparse_faster_fraction": float(np.mean(paired_speedups > 1.0)),
        "paired_speedup_min": float(paired_speedups.min()),
        "paired_speedup_max": float(paired_speedups.max()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    paired_actions = _paired_actions(dense, sparse)
    if paired_actions is not None:
        dense_actions, sparse_actions = paired_actions
        dense_norm = np.linalg.norm(dense_actions, axis=1)
        sparse_norm = np.linalg.norm(sparse_actions, axis=1)
        denominator = dense_norm * sparse_norm
        cosine = np.divide(
            np.sum(dense_actions * sparse_actions, axis=1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        both_zero = (dense_norm <= 1e-12) & (sparse_norm <= 1e-12)
        cosine[both_zero] = 1.0
        relative_l2 = np.linalg.norm(
            sparse_actions - dense_actions, axis=1
        ) / np.maximum(dense_norm, 1e-12)
        worst_index = int(np.argmax(relative_l2))
        measured = _measured_records(dense)
        worst_record = measured[worst_index]
        comparison.update(
            {
                "action_cosine_mean": float(cosine.mean()),
                "action_cosine_min": float(cosine.min()),
                "action_relative_l2_mean": float(relative_l2.mean()),
                "action_relative_l2_max": float(relative_l2.max()),
                "worst_action_request_index": int(
                    worst_record["request_index"]
                ),
                "worst_action_request_key": worst_record.get("request_key"),
                "action_cosine_per_request": cosine.tolist(),
                "action_relative_l2_per_request": relative_l2.tolist(),
            }
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired dense and sparse DreamZero server benchmarks."
    )
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    dense = json.loads(args.dense.read_text())
    sparse = json.loads(args.sparse.read_text())
    comparison = compare_reports(
        dense,
        sparse,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
