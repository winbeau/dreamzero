"""Build reproducible timestep/layer Packed-M2 ablation budget tables.

The builder uses only the aggregated Dense Oracle summaries.  It preserves the
empirical ordering of sensitive timesteps/layers but assigns a small, fixed set
of budget buckets with a matched distribution, so the timestep-only,
layer-only, and timestep+layer ablations have comparable average compute.
These tables are ablation policies, not substitutes for the calibrated M1
classifier or its Dense fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedBudgetTable,
)


PROFILES = {
    # Speed-oriented fixed-shape ablation. Mean assigned budget is about 37%.
    "aggressive": (
        (0.20, 0.35, 0.50, 0.75),
        (0.35, 0.30, 0.25, 0.10),
    ),
    # Conservative structure check before the per-head M1 policy is active.
    "quality": (
        (0.50, 0.75, 1.00),
        (0.30, 0.45, 0.25),
    ),
}


def rank_assign(
    scores: np.ndarray,
    *,
    budgets: tuple[float, ...],
    fractions: tuple[float, ...],
) -> np.ndarray:
    """Assign higher budgets to larger empirical sensitivity scores."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("scores must be a non-empty vector")
    if len(budgets) != len(fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("budget fractions must align and sum to one")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(fractions)
    assignments = np.empty(len(scores), dtype=np.float64)
    for rank, index in enumerate(order):
        quantile = (rank + 0.5) / len(scores)
        bucket_index = int(np.searchsorted(cumulative, quantile, side="right"))
        assignments[index] = budgets[min(bucket_index, len(budgets) - 1)]
    return assignments


def build_tables(
    timestep_summary: pd.DataFrame,
    layer_summary: pd.DataFrame,
    *,
    profile: str,
) -> dict[str, DynamicPackedBudgetTable]:
    budgets, fractions = PROFILES[profile]
    timestep = timestep_summary.sort_values("dit_index")
    layer = layer_summary.sort_values("layer_index")
    if timestep["dit_index"].tolist() != list(range(8)):
        raise ValueError("Timestep summary must contain all eight real DiT indices")
    if layer["layer_index"].tolist() != list(range(40)):
        raise ValueError("Layer summary must contain all forty Transformer layers")

    timestep_scores = timestep["oracle_mean"].to_numpy(dtype=np.float64)
    layer_scores = layer["oracle_mean"].to_numpy(dtype=np.float64)
    timestep_budget = rank_assign(
        timestep_scores,
        budgets=budgets,
        fractions=fractions,
    )
    layer_budget = rank_assign(
        layer_scores,
        budgets=budgets,
        fractions=fractions,
    )

    # Standardize before combining because the empirical timestep effect is
    # much smaller than the U-shaped layer effect. Equal z-score weight makes
    # the joint ablation test both axes instead of reproducing layer-only.
    timestep_z = (timestep_scores - timestep_scores.mean()) / timestep_scores.std()
    layer_z = (layer_scores - layer_scores.mean()) / layer_scores.std()
    joint_scores = (timestep_z[:, None] + layer_z[None, :]).reshape(-1)
    joint_budget = rank_assign(
        joint_scores,
        budgets=budgets,
        fractions=fractions,
    ).reshape(8, 40)

    fixed_budget = float(np.mean(joint_budget))
    fixed_bucket = min(budgets, key=lambda value: abs(value - fixed_budget))
    matrices = {
        "fixed_matched": np.full((8, 40), fixed_bucket, dtype=np.float64),
        "timestep_only": np.repeat(timestep_budget[:, None], 40, axis=1),
        "layer_only": np.repeat(layer_budget[None, :], 8, axis=0),
        "timestep_layer": joint_budget,
    }
    return {
        name: DynamicPackedBudgetTable(
            history_keep_ratios=tuple(map(tuple, matrix.tolist())),
            current_keep_ratios=tuple(map(tuple, matrix.tolist())),
            name=f"{profile}_{name}",
        )
        for name, matrix in matrices.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep-summary", type=Path, required=True)
    parser.add_argument("--layer-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quality")
    args = parser.parse_args()

    tables = build_tables(
        pd.read_parquet(args.timestep_summary),
        pd.read_parquet(args.layer_summary),
        profile=args.profile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"profile": args.profile, "tables": {}}
    for name, table in tables.items():
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
        values = np.asarray(table.history_keep_ratios)
        summary["tables"][name] = {
            "path": str(path),
            "mean_budget": float(values.mean()),
            "bucket_counts": {
                str(value): int(count)
                for value, count in zip(*np.unique(values, return_counts=True))
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
