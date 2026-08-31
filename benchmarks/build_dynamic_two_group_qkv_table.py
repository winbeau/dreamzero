"""Build a two-shape critical/normal QKV table from deployment-safe M1 priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedHeadGroupBudgetTable,
    canonical_budget,
)


KEYS = ("dit_index", "layer_index", "head_index")


def build_two_group_qkv_table(
    prior: pd.DataFrame,
    *,
    critical_threshold: float = 0.80,
    critical_history_ratio: float = 1.00,
    critical_current_ratio: float = 0.75,
    normal_history_ratio: float = 0.35,
    normal_current_ratio: float = 0.35,
    num_dit_steps: int = 8,
    num_layers: int = 40,
    num_heads: int = 40,
    name: str = "m1_prior_critical080_qkv_two_group",
) -> DynamicPackedHeadGroupBudgetTable:
    required = {*KEYS, "prior_budget_mean_tlh"}
    missing = required - set(prior.columns)
    if missing:
        raise ValueError(f"M1 prior table is missing columns: {sorted(missing)}")
    if not 0.0 <= critical_threshold <= 1.0:
        raise ValueError("critical_threshold must lie in [0, 1]")
    if prior.duplicated(list(KEYS)).any():
        raise ValueError("M1 prior table contains duplicate (t,l,h) rows")

    ordered = prior.sort_values(list(KEYS))
    expected = pd.MultiIndex.from_product(
        (range(num_dit_steps), range(num_layers), range(num_heads)),
        names=KEYS,
    )
    actual = pd.MultiIndex.from_frame(ordered[list(KEYS)])
    missing_grid = expected.difference(actual)
    extra_grid = actual.difference(expected)
    if len(missing_grid) or len(extra_grid):
        raise ValueError(
            "M1 prior table does not cover the requested dense grid: "
            f"missing={len(missing_grid)} extra={len(extra_grid)}"
        )

    critical_history_ratio = canonical_budget(critical_history_ratio)
    critical_current_ratio = canonical_budget(critical_current_ratio)
    normal_history_ratio = canonical_budget(normal_history_ratio)
    normal_current_ratio = canonical_budget(normal_current_ratio)
    critical = (
        ordered["prior_budget_mean_tlh"].to_numpy(dtype=np.float64)
        >= critical_threshold
    ).reshape(num_dit_steps, num_layers, num_heads)
    history = np.where(
        critical,
        critical_history_ratio,
        normal_history_ratio,
    )
    current = np.where(
        critical,
        critical_current_ratio,
        normal_current_ratio,
    )

    def cube(values: np.ndarray) -> tuple[tuple[tuple[float, ...], ...], ...]:
        return tuple(
            tuple(tuple(float(value) for value in heads) for heads in row)
            for row in values
        )

    return DynamicPackedHeadGroupBudgetTable(
        head_keep_ratios=cube(history),
        head_current_keep_ratios=cube(current),
        name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--critical-threshold", type=float, default=0.80)
    parser.add_argument("--critical-history-ratio", type=float, default=1.00)
    parser.add_argument("--critical-current-ratio", type=float, default=0.75)
    parser.add_argument("--normal-history-ratio", type=float, default=0.35)
    parser.add_argument("--normal-current-ratio", type=float, default=0.35)
    parser.add_argument("--name", default="m1_prior_critical080_qkv_two_group")
    args = parser.parse_args()

    table = build_two_group_qkv_table(
        pd.read_parquet(args.prior_table),
        critical_threshold=args.critical_threshold,
        critical_history_ratio=args.critical_history_ratio,
        critical_current_ratio=args.critical_current_ratio,
        normal_history_ratio=args.normal_history_ratio,
        normal_current_ratio=args.normal_current_ratio,
        name=args.name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
    history = np.asarray(table.head_keep_ratios)
    current = np.asarray(table.head_current_keep_ratios)
    critical = (history == args.critical_history_ratio) & (
        current == args.critical_current_ratio
    )
    per_cell = critical.sum(axis=2)
    summary = {
        "output": str(args.output),
        "name": table.name,
        "shape": list(history.shape),
        "maximum_groups_per_timestep_layer": table.num_groups,
        "critical_threshold": args.critical_threshold,
        "critical_head_fraction": float(critical.mean()),
        "critical_heads_per_cell_mean": float(per_cell.mean()),
        "critical_heads_per_cell_min": int(per_cell.min()),
        "critical_heads_per_cell_max": int(per_cell.max()),
        "mean_history_keep_ratio": float(history.mean()),
        "mean_current_qkv_keep_ratio": float(current.mean()),
        "critical_ratios": {
            "history": args.critical_history_ratio,
            "current_qkv": args.critical_current_ratio,
        },
        "normal_ratios": {
            "history": args.normal_history_ratio,
            "current_qkv": args.normal_current_ratio,
        },
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
