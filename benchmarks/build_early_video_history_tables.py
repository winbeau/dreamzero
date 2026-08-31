"""Raise only early-layer video-query historical K/V budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedBudgetTable,
    canonical_budget,
)


EARLY_LAYER_INDICES = range(1, 14)


def raise_early_history_floor(
    table: DynamicPackedBudgetTable,
    *,
    floor: float,
) -> DynamicPackedBudgetTable:
    floor = canonical_budget(floor)
    history = [list(row) for row in table.history_keep_ratios]
    for row in history:
        for layer_index in EARLY_LAYER_INDICES:
            row[layer_index] = max(row[layer_index], floor)
    return DynamicPackedBudgetTable(
        history_keep_ratios=tuple(tuple(row) for row in history),
        current_keep_ratios=table.current_keep_ratios,
        name=f"{table.name}_early_history_{int(round(100 * floor))}",
    )


def candidate_tables(
    table: DynamicPackedBudgetTable,
) -> tuple[DynamicPackedBudgetTable, ...]:
    return tuple(
        raise_early_history_floor(table, floor=floor)
        for floor in (0.50, 0.75, 1.00)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = DynamicPackedBudgetTable.from_json(args.base_table)
    if base.num_dit_steps != 8 or base.num_layers != 40:
        raise ValueError("DreamZero early-history tables require an 8x40 base table")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for table in candidate_tables(base):
        path = args.output_dir / f"{table.name}.json"
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
