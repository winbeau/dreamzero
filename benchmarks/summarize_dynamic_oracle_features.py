"""Summarize the full dynamic Oracle table into reproducible paper statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BUDGET_BUCKETS = (0.10, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    return json.loads(frame[columns].to_json(orient="records"))


def summarize(compact_table: Path, budget_cube: Path, output_dir: Path) -> dict:
    columns = [
        "request_key",
        "split",
        "trajectory_stage",
        "source_episode_index",
        "dit_index",
        "layer_index",
        "head_index",
        "oracle_min_keep_ratio",
        "video_oracle_min_keep_ratio",
        "action_oracle_min_keep_ratio",
        "support_turnover_max",
        "vv_output_change_relative_l2_max",
        "qa_qv_key_importance_correlation_mean",
        "worst_mass_p05_r020",
    ]
    frame = pd.read_parquet(compact_table, columns=columns)
    frame["split"] = frame["split"].replace({"validation": "val"})
    budget_values = np.asarray(BUDGET_BUCKETS)
    raw_budget = frame["oracle_min_keep_ratio"].to_numpy()
    nearest_budget = np.argmin(
        np.abs(raw_budget[:, None] - budget_values[None, :]), axis=1
    )
    frame["oracle_budget_bucket"] = budget_values[nearest_budget]
    frame["timestep_bucket"] = pd.cut(
        frame["dit_index"],
        bins=(-1, 2, 4, 7),
        labels=("early_0_2", "middle_3_4", "late_5_7"),
    )
    # The boundaries describe the observed U-shape rather than assuming a
    # monotonic depth law: early stem, sparse middle trough, late recovery.
    frame["layer_bucket"] = pd.cut(
        frame["layer_index"],
        bins=(-1, 11, 27, 39),
        labels=("early_0_11", "middle_12_27", "late_28_39"),
    )

    timestep = (
        frame.groupby("dit_index", observed=True)
        .agg(
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            video_mean=("video_oracle_min_keep_ratio", "mean"),
            action_mean=("action_oracle_min_keep_ratio", "mean"),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
            turnover_mean=("support_turnover_max", "mean"),
            vv_relative_l2_mean=("vv_output_change_relative_l2_max", "mean"),
        )
        .reset_index()
    )
    layer = (
        frame.groupby("layer_index", observed=True)
        .agg(
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            video_mean=("video_oracle_min_keep_ratio", "mean"),
            action_mean=("action_oracle_min_keep_ratio", "mean"),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
            task_std=("oracle_min_keep_ratio", "std"),
        )
        .reset_index()
    )
    head = (
        frame.groupby("head_index", observed=True)
        .agg(
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            oracle_std=("oracle_min_keep_ratio", "std"),
            oracle_p90=("oracle_min_keep_ratio", lambda value: value.quantile(0.90)),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
            turnover_mean=("support_turnover_max", "mean"),
            vv_relative_l2_mean=("vv_output_change_relative_l2_max", "mean"),
            qa_qv_correlation_mean=(
                "qa_qv_key_importance_correlation_mean",
                "mean",
            ),
        )
        .reset_index()
    )
    timestep_bucket = (
        frame.groupby("timestep_bucket", observed=True)
        .agg(
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
        )
        .reset_index()
    )
    layer_bucket = (
        frame.groupby("layer_bucket", observed=True)
        .agg(
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
        )
        .reset_index()
    )
    stage = (
        frame.groupby("trajectory_stage", observed=True)
        .agg(
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            oracle_std=("oracle_min_keep_ratio", "std"),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
        )
        .reset_index()
    )
    split = (
        frame.groupby("split", observed=True)
        .agg(
            episodes=("source_episode_index", "nunique"),
            requests=("request_key", "nunique"),
            oracle_mean=("oracle_min_keep_ratio", "mean"),
            dense_rate=("oracle_min_keep_ratio", lambda value: (value == 1.0).mean()),
        )
        .reset_index()
    )
    label_counts = frame["oracle_budget_bucket"].value_counts().sort_index()

    cube = np.load(budget_cube)
    mean_budget = cube["mean_budget"]
    task_std = cube["task_std"]
    dense_fallback = cube["dense_fallback_rate"]
    mass20 = cube["mean_mass_retention_at_20pct"]
    conditional_unconditional_gap = np.abs(
        mean_budget[:, 0] - mean_budget[:, 1]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestep.to_parquet(output_dir / "oracle_timestep_summary.parquet", index=False)
    layer.to_parquet(output_dir / "oracle_layer_summary.parquet", index=False)
    head.to_parquet(output_dir / "oracle_head_summary.parquet", index=False)

    lowest_heads = head.nsmallest(8, "oracle_mean")
    highest_heads = head.nlargest(8, "oracle_mean")
    summary = {
        "row_count": len(frame),
        "request_count": int(frame["request_key"].nunique()),
        "episode_count": int(frame["source_episode_index"].nunique()),
        "budget_distribution": {
            f"{ratio:.2f}": {
                "count": int(label_counts.get(ratio, 0)),
                "fraction": float(label_counts.get(ratio, 0) / len(frame)),
            }
            for ratio in BUDGET_BUCKETS
        },
        "overall": {
            "oracle_mean": float(frame["oracle_min_keep_ratio"].mean()),
            "oracle_std": float(frame["oracle_min_keep_ratio"].std()),
            "dense_rate": float((frame["oracle_min_keep_ratio"] == 1.0).mean()),
            "fixed_20pct_worst_mass_p05_pass_rate": float(
                (frame["worst_mass_p05_r020"] >= 0.9).mean()
            ),
        },
        "timestep": _records(timestep, list(timestep.columns)),
        "timestep_bucket": _records(timestep_bucket, list(timestep_bucket.columns)),
        "layer": _records(layer, list(layer.columns)),
        "layer_bucket": _records(layer_bucket, list(layer_bucket.columns)),
        "trajectory_stage": _records(stage, list(stage.columns)),
        "split": _records(split, list(split.columns)),
        "lowest_budget_heads": _records(lowest_heads, list(lowest_heads.columns)),
        "highest_budget_heads": _records(highest_heads, list(highest_heads.columns)),
        "cube_diagnostics": {
            "conditional_unconditional_budget_gap_mean": float(
                np.nanmean(conditional_unconditional_gap)
            ),
            "conditional_unconditional_budget_gap_p95": float(
                np.nanquantile(conditional_unconditional_gap, 0.95)
            ),
            "task_std_mean": float(np.nanmean(task_std)),
            "task_std_p95": float(np.nanquantile(task_std, 0.95)),
            "dense_fallback_rate_mean": float(np.nanmean(dense_fallback)),
            "mass20_mean": float(np.nanmean(mass20)),
        },
        "passed": bool(
            len(frame) == 108 * 8 * 40 * 40
            and len(timestep) == 8
            and len(layer) == 40
            and len(head) == 40
        ),
    }
    (output_dir / "oracle_feature_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-table", type=Path, required=True)
    parser.add_argument("--budget-cube", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.compact_table, args.budget_cube, args.output_dir)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
