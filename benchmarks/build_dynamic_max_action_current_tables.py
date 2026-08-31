"""Build fixed 8x40 maximum-current K/V schedules for action queries.

Packed M2 scatters and repacks current-video state at propagation boundaries.
The maximum packed prefix is therefore freshest at a segment entry and most
stale at its exit.  These schedules isolate that distinction without changing
the number of real DiT evaluations or the packed video-query compute shape.
"""

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
PROPAGATE_EVERY = 5


def propagation_segment_layers(
    *,
    num_layers: int = NUM_LAYERS,
    dense_prefix_layers: int = DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = DENSE_SUFFIX_LAYERS,
    propagate_every: int = PROPAGATE_EVERY,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return packed-segment entry and exit Transformer layer indices."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if dense_prefix_layers < 0 or dense_suffix_layers < 0:
        raise ValueError("Dense prefix/suffix layer counts must be non-negative")
    sparse_end = num_layers - dense_suffix_layers
    if dense_prefix_layers >= sparse_end:
        raise ValueError("Packed middle must contain at least one layer")
    if propagate_every <= 0:
        raise ValueError("propagate_every must be positive")

    entries = tuple(
        range(dense_prefix_layers, sparse_end, propagate_every)
    )
    exits = tuple(
        min(entry + propagate_every - 1, sparse_end - 1)
        for entry in entries
    )
    return entries, exits


def make_table(
    *,
    name: str,
    dit_indices: range,
    layer_indices: tuple[int, ...] | range,
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
    segment_entries, segment_exits = propagation_segment_layers()
    return (
        make_table(name="none", dit_indices=range(0), layer_indices=range(0)),
        make_table(
            name="all_middle",
            dit_indices=all_dits,
            layer_indices=all_middle,
        ),
        make_table(
            name="segment_entries",
            dit_indices=all_dits,
            layer_indices=segment_entries,
        ),
        make_table(
            name="segment_exits",
            dit_indices=all_dits,
            layer_indices=segment_exits,
        ),
        make_table(
            name="early_dit_segment_entries",
            dit_indices=range(0, 2),
            layer_indices=segment_entries,
        ),
        make_table(
            name="late_dit_segment_entries",
            dit_indices=range(4, 8),
            layer_indices=segment_entries,
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
