"""Build current-token sentinel budgets aligned with packed propagation.

The base timestep/layer table continues to control historical KV everywhere.
Only the current-token budget at a propagation boundary is promoted.  The
larger exact current-token delta is therefore available immediately before the
packed state is spatially propagated back to the full video grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedBudgetTable,
    canonical_budget,
)


def propagation_boundary_layers(
    *,
    num_layers: int,
    dense_prefix_layers: int,
    dense_suffix_layers: int,
    propagate_every: int,
) -> tuple[int, ...]:
    """Return packed layer indices whose output is spatially propagated."""

    if propagate_every <= 0:
        raise ValueError("propagate_every must be positive")
    sparse_end = num_layers - dense_suffix_layers
    if dense_prefix_layers < 0 or sparse_end <= dense_prefix_layers:
        raise ValueError("Dense boundaries leave no packed middle layers")
    middle = range(dense_prefix_layers, sparse_end)
    return tuple(
        layer_index
        for layer_index in middle
        if (
            (layer_index - dense_prefix_layers + 1) % propagate_every == 0
            or layer_index == sparse_end - 1
        )
    )


def build_sentinel_table(
    base: DynamicPackedBudgetTable,
    *,
    sentinel_current_keep_ratio: float,
    dense_prefix_layers: int = 1,
    dense_suffix_layers: int = 1,
    propagate_every: int = 5,
) -> DynamicPackedBudgetTable:
    """Promote current compute only at packed propagation boundaries."""

    sentinel = canonical_budget(sentinel_current_keep_ratio)
    boundaries = propagation_boundary_layers(
        num_layers=base.num_layers,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        propagate_every=propagate_every,
    )
    current = np.asarray(base.current_keep_ratios, dtype=np.float64).copy()
    for layer_index in boundaries:
        current[:, layer_index] = np.maximum(current[:, layer_index], sentinel)
    suffix = str(int(round(sentinel * 100)))
    return DynamicPackedBudgetTable(
        history_keep_ratios=base.history_keep_ratios,
        current_keep_ratios=tuple(map(tuple, current.tolist())),
        name=f"{base.name}_prop{propagate_every}_current_sentinel{suffix}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sentinel-current-keep-ratios",
        type=float,
        nargs="+",
        default=[0.75, 1.0],
    )
    parser.add_argument("--dense-prefix-layers", type=int, default=1)
    parser.add_argument("--dense-suffix-layers", type=int, default=1)
    parser.add_argument("--propagate-every", type=int, default=5)
    args = parser.parse_args()

    base = DynamicPackedBudgetTable.from_json(args.base_table)
    boundaries = propagation_boundary_layers(
        num_layers=base.num_layers,
        dense_prefix_layers=args.dense_prefix_layers,
        dense_suffix_layers=args.dense_suffix_layers,
        propagate_every=args.propagate_every,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_summary: dict[str, object] = {}
    summary: dict[str, object] = {
        "base_table": str(args.base_table),
        "propagate_every": args.propagate_every,
        "dense_prefix_layers": args.dense_prefix_layers,
        "dense_suffix_layers": args.dense_suffix_layers,
        "boundary_layers": list(boundaries),
        "tables": table_summary,
    }
    for ratio in args.sentinel_current_keep_ratios:
        table = build_sentinel_table(
            base,
            sentinel_current_keep_ratio=ratio,
            dense_prefix_layers=args.dense_prefix_layers,
            dense_suffix_layers=args.dense_suffix_layers,
            propagate_every=args.propagate_every,
        )
        suffix = str(int(round(canonical_budget(ratio) * 100)))
        path = args.output_dir / f"current_sentinel_{suffix}.json"
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
        current = np.asarray(table.current_keep_ratios)
        history = np.asarray(table.history_keep_ratios)
        table_summary[suffix] = {
            "path": str(path),
            "name": table.name,
            "mean_history_budget": float(history.mean()),
            "mean_current_budget": float(current.mean()),
            "early_mean_current_budget": float(current[0, 1:-1].mean()),
            "late_mean_current_budget": float(current[-1, 1:-1].mean()),
        }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
