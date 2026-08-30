"""Build current-token budgets aligned with packed propagation segments.

The base timestep/layer table continues to control historical KV everywhere.
The primary tables hold the current-token prefix constant inside each packed
segment, preventing a token with skipped hidden-state updates from being
reintroduced before propagation. Boundary-only sentinel tables are also
emitted as an explicit negative/diagnostic ablation.
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


def propagation_segments(
    *,
    num_layers: int,
    dense_prefix_layers: int,
    dense_suffix_layers: int,
    propagate_every: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition packed layers at the exact propagation boundaries."""

    boundaries = set(
        propagation_boundary_layers(
            num_layers=num_layers,
            dense_prefix_layers=dense_prefix_layers,
            dense_suffix_layers=dense_suffix_layers,
            propagate_every=propagate_every,
        )
    )
    sparse_end = num_layers - dense_suffix_layers
    segments: list[tuple[int, ...]] = []
    start = dense_prefix_layers
    for layer_index in range(dense_prefix_layers, sparse_end):
        if layer_index in boundaries:
            segments.append(tuple(range(start, layer_index + 1)))
            start = layer_index + 1
    return tuple(segments)


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


def build_segment_max_table(
    base: DynamicPackedBudgetTable,
    *,
    current_floor: float | None = None,
    dense_prefix_layers: int = 1,
    dense_suffix_layers: int = 1,
    propagate_every: int = 5,
) -> DynamicPackedBudgetTable:
    """Keep the current-token active prefix stable within every segment.

    Allowing the prefix to shrink and then expand within a segment reintroduces
    tokens whose hidden states skipped intervening Transformer layers.  Each
    segment therefore uses its maximum configured current budget.  Historical
    KV budgets remain layer-dependent because they carry no mutable packed
    state between layers.
    """

    floor = canonical_budget(current_floor) if current_floor is not None else None
    segments = propagation_segments(
        num_layers=base.num_layers,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        propagate_every=propagate_every,
    )
    current = np.asarray(base.current_keep_ratios, dtype=np.float64).copy()
    for dit_index in range(base.num_dit_steps):
        for segment in segments:
            segment_ratio = float(current[dit_index, list(segment)].max())
            if floor is not None:
                segment_ratio = max(segment_ratio, floor)
            current[dit_index, list(segment)] = segment_ratio
    suffix = "max" if floor is None else f"floor{int(round(floor * 100))}"
    return DynamicPackedBudgetTable(
        history_keep_ratios=base.history_keep_ratios,
        current_keep_ratios=tuple(map(tuple, current.tolist())),
        name=f"{base.name}_prop{propagate_every}_segment_{suffix}",
    )


def build_segment_group_floor_table(
    base: DynamicPackedBudgetTable,
    *,
    segment_indices: tuple[int, ...],
    current_floor: float,
    dense_prefix_layers: int = 1,
    dense_suffix_layers: int = 1,
    propagate_every: int = 5,
) -> DynamicPackedBudgetTable:
    """Promote selected stable segments while leaving all others at segment max."""

    floor = canonical_budget(current_floor)
    segments = propagation_segments(
        num_layers=base.num_layers,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        propagate_every=propagate_every,
    )
    if not segment_indices or len(set(segment_indices)) != len(segment_indices):
        raise ValueError("segment_indices must be non-empty and unique")
    if any(not 0 <= index < len(segments) for index in segment_indices):
        raise ValueError("segment index is outside the packed propagation segments")
    stable = build_segment_max_table(
        base,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        propagate_every=propagate_every,
    )
    current = np.asarray(stable.current_keep_ratios, dtype=np.float64).copy()
    for segment_index in segment_indices:
        layers = list(segments[segment_index])
        current[:, layers] = np.maximum(current[:, layers], floor)
    group = "_".join(str(index) for index in segment_indices)
    return DynamicPackedBudgetTable(
        history_keep_ratios=base.history_keep_ratios,
        current_keep_ratios=tuple(map(tuple, current.tolist())),
        name=(
            f"{base.name}_prop{propagate_every}_segments{group}_"
            f"floor{int(round(floor * 100))}"
        ),
    )


def build_timestep_segment_policy(
    base: DynamicPackedBudgetTable,
    *,
    promotions: tuple[tuple[tuple[int, ...], tuple[int, ...], float], ...],
    name: str,
    dense_prefix_layers: int = 1,
    dense_suffix_layers: int = 1,
    propagate_every: int = 5,
) -> DynamicPackedBudgetTable:
    """Apply fixed segment floors to selected real DiT buckets."""

    segments = propagation_segments(
        num_layers=base.num_layers,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        propagate_every=propagate_every,
    )
    stable = build_segment_max_table(
        base,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        propagate_every=propagate_every,
    )
    current = np.asarray(stable.current_keep_ratios, dtype=np.float64).copy()
    for dit_indices, segment_indices, raw_floor in promotions:
        floor = canonical_budget(raw_floor)
        if any(not 0 <= index < base.num_dit_steps for index in dit_indices):
            raise ValueError("DiT index is outside the dynamic budget table")
        if any(not 0 <= index < len(segments) for index in segment_indices):
            raise ValueError("segment index is outside the packed propagation segments")
        for dit_index in dit_indices:
            for segment_index in segment_indices:
                layers = list(segments[segment_index])
                current[dit_index, layers] = np.maximum(
                    current[dit_index, layers], floor
                )
    return DynamicPackedBudgetTable(
        history_keep_ratios=base.history_keep_ratios,
        current_keep_ratios=tuple(map(tuple, current.tolist())),
        name=name,
    )


def build_history_floor_table(
    base: DynamicPackedBudgetTable,
    *,
    history_floor: float,
) -> DynamicPackedBudgetTable:
    """Promote historical K/V only, preserving current packed compute exactly."""

    floor = canonical_budget(history_floor)
    history = np.maximum(
        np.asarray(base.history_keep_ratios, dtype=np.float64), floor
    )
    suffix = str(int(round(floor * 100)))
    return DynamicPackedBudgetTable(
        history_keep_ratios=tuple(map(tuple, history.tolist())),
        current_keep_ratios=base.current_keep_ratios,
        name=f"{base.name}_history_floor{suffix}",
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
    parser.add_argument(
        "--segment-current-floor-ratios",
        type=float,
        nargs="*",
        default=[0.75],
        help=(
            "Optional conservative floors for segment-stable current budgets; "
            "a no-floor segment-max table is always emitted."
        ),
    )
    parser.add_argument(
        "--segment-floor-groups",
        nargs="*",
        default=["0,1", "2,3"],
        help="Comma-separated propagation-segment groups for localized floors.",
    )
    parser.add_argument("--segment-group-floor-ratio", type=float, default=0.75)
    parser.add_argument(
        "--history-floor-ratios",
        type=float,
        nargs="*",
        default=[0.75, 1.0],
        help="Historical-K/V-only floors emitted for the timestep policies.",
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
    segments = propagation_segments(
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
        "segments": [list(segment) for segment in segments],
        "tables": table_summary,
    }

    def record_table(key: str, path: Path, table: DynamicPackedBudgetTable) -> None:
        path.write_text(json.dumps(table.to_dict(), indent=2) + "\n")
        current = np.asarray(table.current_keep_ratios)
        history = np.asarray(table.history_keep_ratios)
        table_summary[key] = {
            "path": str(path),
            "name": table.name,
            "mean_history_budget": float(history.mean()),
            "mean_current_budget": float(current.mean()),
            "early_mean_current_budget": float(current[0, 1:-1].mean()),
            "late_mean_current_budget": float(current[-1, 1:-1].mean()),
        }

    segment_max = build_segment_max_table(
        base,
        dense_prefix_layers=args.dense_prefix_layers,
        dense_suffix_layers=args.dense_suffix_layers,
        propagate_every=args.propagate_every,
    )
    record_table("segment_max", args.output_dir / "segment_max.json", segment_max)
    for ratio in args.segment_current_floor_ratios:
        floor = canonical_budget(ratio)
        suffix = str(int(round(floor * 100)))
        table = build_segment_max_table(
            base,
            current_floor=floor,
            dense_prefix_layers=args.dense_prefix_layers,
            dense_suffix_layers=args.dense_suffix_layers,
            propagate_every=args.propagate_every,
        )
        record_table(
            f"segment_floor_{suffix}",
            args.output_dir / f"segment_floor_{suffix}.json",
            table,
        )

    group_floor = canonical_budget(args.segment_group_floor_ratio)
    for raw_group in args.segment_floor_groups:
        indices = tuple(int(value) for value in raw_group.split(",") if value)
        table = build_segment_group_floor_table(
            base,
            segment_indices=indices,
            current_floor=group_floor,
            dense_prefix_layers=args.dense_prefix_layers,
            dense_suffix_layers=args.dense_suffix_layers,
            propagate_every=args.propagate_every,
        )
        group = "_".join(str(index) for index in indices)
        suffix = str(int(round(group_floor * 100)))
        record_table(
            f"segment_group_{group}_floor_{suffix}",
            args.output_dir / f"segment_group_{group}_floor_{suffix}.json",
            table,
        )

    all_segments = tuple(range(len(segments)))
    balanced = build_timestep_segment_policy(
        base,
        promotions=(
            ((0, 1, 2), (0, 1, 2), 0.75),
            ((3, 4), all_segments, 0.50),
            ((3, 4), (0, 1), 0.75),
            ((5, 6, 7), all_segments, 0.35),
            ((5, 6, 7), (0,), 0.50),
        ),
        name="timestep_segment_balanced",
        dense_prefix_layers=args.dense_prefix_layers,
        dense_suffix_layers=args.dense_suffix_layers,
        propagate_every=args.propagate_every,
    )
    record_table(
        "timestep_segment_balanced",
        args.output_dir / "timestep_segment_balanced.json",
        balanced,
    )
    quality = build_timestep_segment_policy(
        base,
        promotions=(
            ((0, 1, 2), (0, 1, 2), 0.75),
            ((3, 4), all_segments, 0.50),
            ((3, 4), (0, 1, 2), 0.75),
            ((5, 6, 7), all_segments, 0.50),
        ),
        name="timestep_segment_quality",
        dense_prefix_layers=args.dense_prefix_layers,
        dense_suffix_layers=args.dense_suffix_layers,
        propagate_every=args.propagate_every,
    )
    record_table(
        "timestep_segment_quality",
        args.output_dir / "timestep_segment_quality.json",
        quality,
    )
    for policy_key, policy in (
        ("timestep_segment_balanced", balanced),
        ("timestep_segment_quality", quality),
    ):
        for ratio in args.history_floor_ratios:
            floor = canonical_budget(ratio)
            suffix = str(int(round(floor * 100)))
            table = build_history_floor_table(policy, history_floor=floor)
            record_table(
                f"{policy_key}_history_floor_{suffix}",
                args.output_dir / f"{policy_key}_history_floor_{suffix}.json",
                table,
            )

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
        record_table(f"boundary_sentinel_{suffix}", path, table)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
