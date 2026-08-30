"""Build a four-ratio Packed-M2 head table from leakage-safe M1 priors.

This builder is an executor/ablation bridge. The final online policy will use
the calibrated M1 prediction and confidence fallback per request; the train-only
``(t,l,h)`` prior provides a deterministic table for validating grouped kernels
before that runtime feature path is connected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedHeadGroupBudgetTable,
)


EXECUTOR_BUCKETS = np.asarray((0.25, 0.50, 0.75, 1.00), dtype=np.float64)
KEYS = ("dit_index", "layer_index", "head_index")


def quantize_executor_budgets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    indices = np.searchsorted(EXECUTOR_BUCKETS, values - 1e-12, side="left")
    return EXECUTOR_BUCKETS[np.clip(indices, 0, len(EXECUTOR_BUCKETS) - 1)]


def build_head_group_table(
    prior: pd.DataFrame,
    *,
    num_dit_steps: int = 8,
    num_layers: int = 40,
    num_heads: int = 40,
    name: str = "m1_prior_mean_group4",
) -> DynamicPackedHeadGroupBudgetTable:
    required = {*KEYS, "prior_budget_mean_tlh"}
    missing_columns = required - set(prior.columns)
    if missing_columns:
        raise ValueError(f"M1 prior table is missing columns: {sorted(missing_columns)}")
    if prior.duplicated(list(KEYS)).any():
        raise ValueError("M1 prior table contains duplicate (t,l,h) rows")
    ordered = prior.sort_values(list(KEYS))
    expected = pd.MultiIndex.from_product(
        (range(num_dit_steps), range(num_layers), range(num_heads)),
        names=KEYS,
    )
    actual = pd.MultiIndex.from_frame(ordered[list(KEYS)])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            "M1 prior table does not cover the requested dense grid: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    quantized = quantize_executor_budgets(
        ordered["prior_budget_mean_tlh"].to_numpy(dtype=np.float64)
    ).reshape(num_dit_steps, num_layers, num_heads)
    return DynamicPackedHeadGroupBudgetTable(
        head_keep_ratios=tuple(
            tuple(tuple(float(value) for value in heads) for heads in row)
            for row in quantized
        ),
        name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = build_head_group_table(pd.read_parquet(args.prior_table))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
    values = np.asarray(table.head_keep_ratios)
    summary = {
        "output": str(args.output),
        "name": table.name,
        "shape": list(values.shape),
        "mean_history_keep_ratio": float(values.mean()),
        "dense_head_fraction": float(np.mean(values == 1.0)),
        "bucket_counts": {
            str(value): int(count)
            for value, count in zip(*np.unique(values, return_counts=True))
        },
        "maximum_groups_per_timestep_layer": table.num_groups,
    }
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
