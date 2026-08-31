"""Build fixed 8x40 action-history schedules for Packed M2 ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicDenseActionHistoryTable,
)


NUM_DIT_STEPS = 8
NUM_LAYERS = 40
DENSE_PREFIX_LAYERS = 1
DENSE_SUFFIX_LAYERS = 1


def make_table(
    *,
    name: str,
    dit_indices: range,
    layer_indices: range,
) -> DynamicDenseActionHistoryTable:
    enabled_dits = set(dit_indices)
    enabled_layers = set(layer_indices)
    return DynamicDenseActionHistoryTable(
        enabled_cells=tuple(
            tuple(
                dit_index in enabled_dits and layer_index in enabled_layers
                for layer_index in range(NUM_LAYERS)
            )
            for dit_index in range(NUM_DIT_STEPS)
        ),
        name=name,
    )


def candidate_tables() -> tuple[DynamicDenseActionHistoryTable, ...]:
    all_dits = range(NUM_DIT_STEPS)
    all_middle = range(DENSE_PREFIX_LAYERS, NUM_LAYERS - DENSE_SUFFIX_LAYERS)
    return (
        make_table(name="none", dit_indices=range(0), layer_indices=range(0)),
        make_table(name="all_middle", dit_indices=all_dits, layer_indices=all_middle),
        make_table(name="early_layers", dit_indices=all_dits, layer_indices=range(1, 14)),
        make_table(name="middle_layers", dit_indices=all_dits, layer_indices=range(14, 28)),
        make_table(name="late_layers", dit_indices=all_dits, layer_indices=range(28, 39)),
        make_table(
            name="early_dit_all_middle",
            dit_indices=range(0, 2),
            layer_indices=all_middle,
        ),
        make_table(
            name="late_dit_all_middle",
            dit_indices=range(4, 8),
            layer_indices=all_middle,
        ),
        make_table(
            name="late_dit_late_layers",
            dit_indices=range(4, 8),
            layer_indices=range(28, 39),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for table in candidate_tables():
        path = args.output_dir / f"{table.name}.json"
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
