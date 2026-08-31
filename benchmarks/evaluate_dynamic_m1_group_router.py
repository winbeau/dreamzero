"""Re-evaluate calibrated M1 after four-shape grouping and downstream fallback."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from benchmarks.train_dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    FEATURE_COLUMNS,
    PACKED_PROXY_INPUT_COLUMNS,
    PRIOR_KEYS,
    add_deployment_features,
    bootstrap_test_metrics,
    budget_indices,
    required_columns,
    route_metrics,
    sequential_predict,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
    quantize_grouped_budgets,
)


def prepare_m1_evaluation_frame(
    oracle_table: Path,
    bundle: dict[str, Any],
    *,
    splits: Iterable[str],
) -> pd.DataFrame:
    splits = tuple(str(split) for split in splits)
    feature_columns = tuple(bundle.get("feature_columns", FEATURE_COLUMNS))
    input_columns = (
        PACKED_PROXY_INPUT_COLUMNS
        if "previous_packed_route_support_turnover_max" in feature_columns
        else ()
    )
    frame = pd.read_parquet(
        oracle_table,
        columns=required_columns(input_columns),
        filters=[("split", "in", list(splits))],
    )
    if frame.empty:
        raise ValueError(f"Oracle table contains no rows for splits {splits}")
    prior_table = bundle.get("prior_table")
    if not isinstance(prior_table, pd.DataFrame):
        raise TypeError("M1 bundle prior_table must be a pandas DataFrame")
    frame = frame.merge(
        prior_table,
        on=list(PRIOR_KEYS),
        how="left",
        validate="many_to_one",
    )
    prior_columns = (
        "prior_budget_mean_tlh",
        "prior_budget_std_tlh",
        "prior_critical_rate_tlh",
    )
    if frame[list(prior_columns)].isna().any().any():
        raise ValueError("M1 prior table does not cover all evaluation rows")
    return add_deployment_features(frame)


def apply_grouped_route_fallback(
    frame: pd.DataFrame,
    result: dict[str, np.ndarray],
    risk_table: DownstreamHeadRiskTable,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    """Apply executor quantization and conservative downstream masks per row."""

    required = {"request_key", "dit_index", "layer_index", "head_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"M1 evaluation frame is missing keys: {sorted(missing)}")
    row_count = len(frame)
    for name in ("prediction", "route_confidence", "fallback"):
        if name not in result or len(result[name]) != row_count:
            raise ValueError(f"M1 result field {name} does not align with frame")
    dit = frame["dit_index"].to_numpy(dtype=np.int64)
    layer = frame["layer_index"].to_numpy(dtype=np.int64)
    head = frame["head_index"].to_numpy(dtype=np.int64)
    if (
        np.any(dit < 0)
        or np.any(dit >= risk_table.num_dit_steps)
        or np.any(layer < 0)
        or np.any(layer >= risk_table.num_layers)
        or np.any(head < 0)
        or np.any(head >= risk_table.num_heads)
    ):
        raise ValueError("M1 evaluation row lies outside downstream risk grid")

    prediction = np.asarray(result["prediction"], dtype=np.int64)
    if np.any(prediction < 0) or np.any(prediction >= len(BUDGET_BUCKETS)):
        raise ValueError("M1 prediction index is outside fixed budget buckets")
    raw_keep = BUDGET_BUCKETS[prediction]
    classifier_fallback = np.asarray(result["fallback"], dtype=bool)
    downstream_scanned = risk_table.scanned[dit, layer, head]
    downstream_safe = risk_table.safe[dit, layer, head]
    downstream_unknown = ~downstream_scanned
    downstream_unsafe = downstream_scanned & ~downstream_safe
    combined_fallback = (
        classifier_fallback | downstream_unknown | downstream_unsafe
    )
    effective_keep = raw_keep.copy()
    effective_keep[combined_fallback] = 1.0
    grouped_keep = quantize_grouped_budgets(effective_keep)
    grouped_prediction = budget_indices(grouped_keep)

    grouped_result = {
        **result,
        "prediction": grouped_prediction,
        "fallback": combined_fallback,
    }
    routes = frame[
        [
            "request_key",
            "split",
            "source_episode_index",
            "dit_index",
            "layer_index",
            "head_index",
            "oracle_min_keep_ratio",
        ]
    ].copy()
    routes["m1_keep_ratio"] = raw_keep
    routes["grouped_keep_ratio"] = grouped_keep
    routes["route_confidence"] = np.asarray(
        result["route_confidence"],
        dtype=np.float64,
    )
    routes["classifier_fallback"] = classifier_fallback
    routes["downstream_unknown_fallback"] = downstream_unknown
    routes["downstream_unsafe_fallback"] = downstream_unsafe
    routes["combined_fallback"] = combined_fallback

    group_counts = (
        routes.groupby(
            ["request_key", "dit_index", "layer_index"],
            sort=False,
        )["grouped_keep_ratio"]
        .nunique()
        .to_numpy(dtype=np.int64)
    )
    diagnostics = {
        "classifier_fallback_rate": float(np.mean(classifier_fallback)),
        "downstream_unknown_fallback_rate": float(np.mean(downstream_unknown)),
        "downstream_unsafe_fallback_rate": float(np.mean(downstream_unsafe)),
        "combined_fallback_rate": float(np.mean(combined_fallback)),
        "mean_grouped_keep_ratio": float(np.mean(grouped_keep)),
        "dense_grouped_route_rate": float(np.mean(grouped_keep == 1.0)),
        "maximum_groups_per_request_timestep_layer": int(group_counts.max()),
        "mean_groups_per_request_timestep_layer": float(group_counts.mean()),
        "unknown_or_unsafe_sparse_count": int(
            np.sum((downstream_unknown | downstream_unsafe) & (grouped_keep < 1.0))
        ),
        "q_k_budget_coupling": "same grouped_keep_ratio",
    }
    if diagnostics["maximum_groups_per_request_timestep_layer"] > 4:
        raise RuntimeError("Grouped M1 evaluation produced more than four shapes")
    if diagnostics["unknown_or_unsafe_sparse_count"] != 0:
        raise RuntimeError("Unknown or unsafe downstream heads remained sparse")
    return grouped_result, routes, diagnostics


def evaluate_grouped_split(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    risk_table: DownstreamHeadRiskTable,
    *,
    bootstrap_repeats: int = 200,
) -> tuple[dict[str, object], pd.DataFrame]:
    if bootstrap_repeats <= 0:
        raise ValueError("bootstrap_repeats must be positive")
    result = sequential_predict(
        bundle["estimator"],
        frame,
        bundle.get("confidence_calibrator"),
        bundle["policy"],
        feature_columns=tuple(bundle.get("feature_columns", FEATURE_COLUMNS)),
    )
    grouped_result, routes, diagnostics = apply_grouped_route_fallback(
        frame,
        result,
        risk_table,
    )
    executor_truth_keep = quantize_grouped_budgets(
        frame["oracle_min_keep_ratio"].to_numpy(dtype=np.float64)
    )
    truth = budget_indices(executor_truth_keep)
    metrics = route_metrics(frame, truth, grouped_result)
    return (
        {
            **metrics,
            **diagnostics,
            "source_episodes": int(frame["source_episode_index"].nunique()),
            "requests": int(frame["request_key"].nunique()),
            "truth_quantization": "Oracle keep ratio rounded upward to executor bucket",
            "bootstrap": {
                "repeats": bootstrap_repeats,
                "unit": "source_episode_index",
                "metrics": bootstrap_test_metrics(
                    frame,
                    truth,
                    grouped_result,
                    bootstrap_repeats,
                ),
            },
        },
        routes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-table", type=Path, required=True)
    parser.add_argument("--m1-bundle", type=Path, required=True)
    parser.add_argument("--downstream-risk-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--false-sparse-limit", type=float, default=0.01)
    parser.add_argument("--mass-gate-rate", type=float, default=0.95)
    parser.add_argument("--minimum-macro-f1", type=float, default=0.50)
    parser.add_argument("--write-routes", action="store_true")
    args = parser.parse_args()

    bundle = joblib.load(args.m1_bundle)
    risk_table = DownstreamHeadRiskTable.from_json(args.downstream_risk_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_results = {}
    for split in args.splits:
        frame = prepare_m1_evaluation_frame(
            args.oracle_table,
            bundle,
            splits=(split,),
        )
        metrics, routes = evaluate_grouped_split(
            frame,
            bundle,
            risk_table,
            bootstrap_repeats=args.bootstrap_repeats,
        )
        split_results[split] = metrics
        if args.write_routes:
            routes.to_parquet(
                args.output_dir / f"{split}_grouped_routes.parquet",
                index=False,
            )

    statistical_gates = all(
        metrics["false_sparse_rate"] < args.false_sparse_limit
        and metrics["mass_p05_at_least_0_9_rate"] >= args.mass_gate_rate
        and metrics["macro_f1"] >= args.minimum_macro_f1
        and metrics["unknown_or_unsafe_sparse_count"] == 0
        for metrics in split_results.values()
    )
    summary = {
        "oracle_table": str(args.oracle_table),
        "m1_bundle": str(args.m1_bundle),
        "downstream_risk_table": str(args.downstream_risk_table),
        "splits": split_results,
        "gates": {
            "false_sparse_limit": args.false_sparse_limit,
            "mass_gate_rate": args.mass_gate_rate,
            "minimum_macro_f1": args.minimum_macro_f1,
            "maximum_execution_groups": 4,
            "unknown_or_unsafe_sparse_count": 0,
        },
        "statistical_gates_passed": bool(statistical_gates),
        "final_action_video_replay": "pending real DreamZero policy replay",
        "passed": False,
        "reason": (
            "Post-grouping statistics do not replace final action/video and "
            "closed-loop gates."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
