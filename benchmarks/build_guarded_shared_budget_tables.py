"""Build conservative shared-shape Packed-M2 schedules from Dense Oracle ranks.

The builder intentionally avoids per-Head execution.  It selects a Cartesian
product of the lowest-sensitivity real DiT steps and Transformer layers, while
protecting an explicit Dense denoise prefix.  Two tables separate historical
K/V sparsity from joint current-token sparsity using the same selected cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedBudgetTable,
    canonical_budget,
)


def lowest_sensitivity_indices(
    summary: pd.DataFrame,
    *,
    index_column: str,
    expected_count: int,
    selected_count: int,
    minimum_index: int = 0,
) -> tuple[int, ...]:
    """Select a stable Oracle-ranked subset after validating complete coverage."""

    ordered = summary.sort_values(index_column)
    expected = list(range(expected_count))
    if ordered[index_column].tolist() != expected:
        raise ValueError(f"{index_column} summary must cover {expected_count} indices")
    if not 0 < selected_count <= expected_count - minimum_index:
        raise ValueError("selected_count exceeds the eligible Oracle grid")
    eligible = ordered.loc[ordered[index_column] >= minimum_index]
    selected = eligible.sort_values(
        ["oracle_mean", index_column],
        kind="stable",
    ).head(selected_count)
    return tuple(sorted(selected[index_column].astype(int).tolist()))


def build_guarded_tables(
    timestep_summary: pd.DataFrame,
    layer_summary: pd.DataFrame,
    *,
    sparse_timestep_count: int = 3,
    sparse_layer_count: int = 20,
    dense_dit_prefix: int = 2,
    sparse_history_keep_ratio: float = 0.75,
    sparse_current_keep_ratio: float = 0.75,
) -> tuple[dict[str, DynamicPackedBudgetTable], dict[str, object]]:
    """Return history-only and joint tables with one shared sparse cell shape."""

    history_ratio = canonical_budget(sparse_history_keep_ratio)
    current_ratio = canonical_budget(sparse_current_keep_ratio)
    if not 0 <= dense_dit_prefix < 8:
        raise ValueError("dense_dit_prefix must lie in [0, 7]")
    timesteps = lowest_sensitivity_indices(
        timestep_summary,
        index_column="dit_index",
        expected_count=8,
        selected_count=sparse_timestep_count,
        minimum_index=dense_dit_prefix,
    )
    layers = lowest_sensitivity_indices(
        layer_summary,
        index_column="layer_index",
        expected_count=40,
        selected_count=sparse_layer_count,
    )

    sparse_mask = np.zeros((8, 40), dtype=bool)
    sparse_mask[np.ix_(timesteps, layers)] = True
    dense = np.ones((8, 40), dtype=np.float64)
    history = np.where(sparse_mask, history_ratio, dense)
    joint_current = np.where(sparse_mask, current_ratio, dense)
    prefix = f"oracle_guarded_t{len(timesteps)}_l{len(layers)}"
    tables = {
        "history_only": DynamicPackedBudgetTable(
            history_keep_ratios=tuple(map(tuple, history.tolist())),
            current_keep_ratios=tuple(map(tuple, dense.tolist())),
            name=f"{prefix}_history_only",
        ),
        "joint": DynamicPackedBudgetTable(
            history_keep_ratios=tuple(map(tuple, history.tolist())),
            current_keep_ratios=tuple(map(tuple, joint_current.tolist())),
            name=f"{prefix}_joint",
        ),
    }
    summary = {
        "dense_dit_prefix": dense_dit_prefix,
        "selected_timesteps": list(timesteps),
        "selected_layers": list(layers),
        "sparse_cell_count": int(sparse_mask.sum()),
        "sparse_cell_fraction": float(sparse_mask.mean()),
        "sparse_history_keep_ratio": history_ratio,
        "sparse_current_keep_ratio": current_ratio,
        "history_mean_keep_ratio": float(history.mean()),
        "joint_current_mean_keep_ratio": float(joint_current.mean()),
    }
    return tables, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep-summary", type=Path, required=True)
    parser.add_argument("--layer-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sparse-timestep-count", type=int, default=3)
    parser.add_argument("--sparse-layer-count", type=int, default=20)
    parser.add_argument("--dense-dit-prefix", type=int, default=2)
    parser.add_argument("--sparse-history-keep-ratio", type=float, default=0.75)
    parser.add_argument("--sparse-current-keep-ratio", type=float, default=0.75)
    args = parser.parse_args()

    tables, summary = build_guarded_tables(
        pd.read_parquet(args.timestep_summary),
        pd.read_parquet(args.layer_summary),
        sparse_timestep_count=args.sparse_timestep_count,
        sparse_layer_count=args.sparse_layer_count,
        dense_dit_prefix=args.dense_dit_prefix,
        sparse_history_keep_ratio=args.sparse_history_keep_ratio,
        sparse_current_keep_ratio=args.sparse_current_keep_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
        paths[name] = str(path)
    payload = {**summary, "tables": paths}
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
