"""Fixed-shape dynamic budget tables for the Packed Middle Stack.

The table is deliberately small: eight real DiT evaluations by forty
Transformer layers, with values restricted to the paper's seven keep-ratio
buckets.  Runtime selection changes only nested-prefix lengths; it never
changes the Router ordering or launches a per-head dynamic-shape kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


BUDGET_BUCKETS = (0.10, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)


def canonical_budget(value: float) -> float:
    """Return the exact configured bucket represented by ``value``."""

    value = float(value)
    for bucket in BUDGET_BUCKETS:
        if abs(value - bucket) <= 1e-6:
            return bucket
    raise ValueError(
        f"Budget {value} is not one of the fixed buckets {BUDGET_BUCKETS}"
    )


def bucket_at_least(value: float) -> float:
    """Conservatively quantize a continuous budget upward."""

    value = float(value)
    for bucket in BUDGET_BUCKETS:
        if value <= bucket + 1e-12:
            return bucket
    return BUDGET_BUCKETS[-1]


def stabilize_current_budgets_for_segments(
    layer_ratios: Sequence[tuple[float, float]],
    *,
    segment_length: int,
) -> tuple[tuple[float, float], ...]:
    """Hold mutable current-token budgets fixed inside packed segments.

    Historical KV is an immutable per-layer input and may retain its original
    layer budget. Current video hidden states cannot safely shrink and later
    expand before a propagation/repack boundary because reactivated tokens
    would have skipped intervening Transformer layers.
    """

    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    canonical = tuple(
        (canonical_budget(history), canonical_budget(current))
        for history, current in layer_ratios
    )
    stabilized: list[tuple[float, float]] = []
    for start in range(0, len(canonical), segment_length):
        segment = canonical[start : start + segment_length]
        current = max(current for _, current in segment)
        stabilized.extend((history, current) for history, _ in segment)
    return tuple(stabilized)


def _canonical_matrix(
    values: Sequence[Sequence[float]],
    *,
    field_name: str,
) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(canonical_budget(value) for value in row) for row in values)
    if not matrix or not matrix[0]:
        raise ValueError(f"{field_name} must be a non-empty 2D table")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{field_name} rows must have equal layer counts")
    return matrix


@dataclass(frozen=True)
class DynamicPackedBudgetTable:
    """Nested current/history budgets indexed by ``(dit_index, layer)``."""

    history_keep_ratios: tuple[tuple[float, ...], ...]
    current_keep_ratios: tuple[tuple[float, ...], ...]
    name: str = "dynamic"

    def __post_init__(self) -> None:
        history = _canonical_matrix(
            self.history_keep_ratios,
            field_name="history_keep_ratios",
        )
        current = _canonical_matrix(
            self.current_keep_ratios,
            field_name="current_keep_ratios",
        )
        if len(history) != len(current) or len(history[0]) != len(current[0]):
            raise ValueError("Current and historical budget tables must align")
        object.__setattr__(self, "history_keep_ratios", history)
        object.__setattr__(self, "current_keep_ratios", current)

    @property
    def num_dit_steps(self) -> int:
        return len(self.history_keep_ratios)

    @property
    def num_layers(self) -> int:
        return len(self.history_keep_ratios[0])

    def ratios(self, dit_index: int, layer_index: int) -> tuple[float, float]:
        if not 0 <= dit_index < self.num_dit_steps:
            raise IndexError(f"dit_index {dit_index} is outside the budget table")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} is outside the budget table")
        return (
            self.history_keep_ratios[dit_index][layer_index],
            self.current_keep_ratios[dit_index][layer_index],
        )

    def maximum_current_ratio(
        self,
        dit_index: int,
        layer_indices: Iterable[int],
    ) -> float:
        ratios = [self.ratios(dit_index, layer)[1] for layer in layer_indices]
        if not ratios:
            raise ValueError("layer_indices must contain at least one layer")
        return max(ratios)

    def history_ratios(
        self,
        dit_index: int,
        layer_indices: Iterable[int],
    ) -> tuple[float, ...]:
        return tuple(
            sorted({self.ratios(dit_index, layer)[0] for layer in layer_indices})
        )

    @classmethod
    def constant(
        cls,
        *,
        num_dit_steps: int,
        num_layers: int,
        history_keep_ratio: float,
        current_keep_ratio: float,
        name: str = "constant",
    ) -> "DynamicPackedBudgetTable":
        history = canonical_budget(history_keep_ratio)
        current = canonical_budget(current_keep_ratio)
        return cls(
            history_keep_ratios=tuple(
                tuple(history for _ in range(num_layers))
                for _ in range(num_dit_steps)
            ),
            current_keep_ratios=tuple(
                tuple(current for _ in range(num_layers))
                for _ in range(num_dit_steps)
            ),
            name=name,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DynamicPackedBudgetTable":
        return cls(
            history_keep_ratios=payload["history_keep_ratios"],  # type: ignore[arg-type]
            current_keep_ratios=payload["current_keep_ratios"],  # type: ignore[arg-type]
            name=str(payload.get("name", "dynamic")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DynamicPackedBudgetTable":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "name": self.name,
            "budget_buckets": list(BUDGET_BUCKETS),
            "num_dit_steps": self.num_dit_steps,
            "num_layers": self.num_layers,
            "history_keep_ratios": [list(row) for row in self.history_keep_ratios],
            "current_keep_ratios": [list(row) for row in self.current_keep_ratios],
        }


def _canonical_budget_cube(
    values: Sequence[Sequence[Sequence[float]]],
    *,
    field_name: str,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    cube = tuple(
        tuple(tuple(canonical_budget(value) for value in groups) for groups in row)
        for row in values
    )
    if not cube or not cube[0] or not cube[0][0]:
        raise ValueError(f"{field_name} must be a non-empty 3D table")
    layer_count = len(cube[0])
    group_count = len(cube[0][0])
    if any(len(row) != layer_count for row in cube):
        raise ValueError(f"{field_name} rows must have equal layer counts")
    if any(len(groups) != group_count for row in cube for groups in row):
        raise ValueError(f"{field_name} cells must have equal group counts")
    return cube


@dataclass(frozen=True)
class DynamicPackedHeadGroupBudgetTable:
    """Per-head historical-KV and current Q/K/V budgets in shared groups.

    The optional current-token cube gives each head a nested current-video
    prefix while action/state registers remain Dense. Heads with the same
    ``(history, current)`` pair execute in one fixed-shape attention call. The
    outer Packed-M2 table still controls the shared hidden-state/FFN length.
    Every cell is restricted to at most four distinct execution groups.
    """

    head_keep_ratios: tuple[tuple[tuple[float, ...], ...], ...]
    head_current_keep_ratios: (
        tuple[tuple[tuple[float, ...], ...], ...] | None
    ) = None
    name: str = "dynamic_head_groups"

    def __post_init__(self) -> None:
        cube = _canonical_budget_cube(
            self.head_keep_ratios,
            field_name="head_keep_ratios",
        )
        current_cube = None
        if self.head_current_keep_ratios is not None:
            current_cube = _canonical_budget_cube(
                self.head_current_keep_ratios,
                field_name="head_current_keep_ratios",
            )
            if (
                len(current_cube) != len(cube)
                or len(current_cube[0]) != len(cube[0])
                or len(current_cube[0][0]) != len(cube[0][0])
            ):
                raise ValueError(
                    "Historical and current head-group budget cubes must align"
                )
        execution_group_counts = []
        for dit_index, row in enumerate(cube):
            for layer_index, heads in enumerate(row):
                current_heads = (
                    current_cube[dit_index][layer_index]
                    if current_cube is not None
                    else (None,) * len(heads)
                )
                execution_group_counts.append(
                    len(set(zip(heads, current_heads, strict=True)))
                )
        if any(count > 4 for count in execution_group_counts):
            raise ValueError(
                "Each timestep/layer may use at most four shared head groups"
            )
        object.__setattr__(self, "head_keep_ratios", cube)
        object.__setattr__(self, "head_current_keep_ratios", current_cube)

    @property
    def num_dit_steps(self) -> int:
        return len(self.head_keep_ratios)

    @property
    def num_layers(self) -> int:
        return len(self.head_keep_ratios[0])

    @property
    def num_groups(self) -> int:
        return max(
            len(self.execution_groups_for_layer(dit_index, layer_index))
            for dit_index in range(self.num_dit_steps)
            for layer_index in range(self.num_layers)
        )

    @property
    def num_heads(self) -> int:
        return len(self.head_keep_ratios[0][0])

    @property
    def group_ratios(self) -> tuple[float, ...]:
        return tuple(
            sorted(
                {
                    ratio
                    for row in self.head_keep_ratios
                    for heads in row
                    for ratio in heads
                }
            )
        )

    @property
    def group_current_ratios(self) -> tuple[float, ...]:
        if self.head_current_keep_ratios is None:
            return ()
        return tuple(
            sorted(
                {
                    ratio
                    for row in self.head_current_keep_ratios
                    for heads in row
                    for ratio in heads
                }
            )
        )

    def ratios(self, dit_index: int, layer_index: int) -> tuple[float, ...]:
        if not 0 <= dit_index < self.num_dit_steps:
            raise IndexError(f"dit_index {dit_index} is outside the head-group table")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} is outside the head-group table")
        return self.head_keep_ratios[dit_index][layer_index]

    def groups_for_layer(
        self,
        dit_index: int,
        layer_index: int,
    ) -> tuple[tuple[tuple[int, ...], float], ...]:
        groups: dict[float, list[int]] = {}
        for head_index, ratio in enumerate(self.ratios(dit_index, layer_index)):
            groups.setdefault(ratio, []).append(head_index)
        return tuple(
            (tuple(groups[ratio]), ratio)
            for ratio in sorted(groups, reverse=True)
        )

    def execution_groups_for_layer(
        self,
        dit_index: int,
        layer_index: int,
    ) -> tuple[tuple[tuple[int, ...], float, float | None], ...]:
        history = self.ratios(dit_index, layer_index)
        current = (
            self.head_current_keep_ratios[dit_index][layer_index]
            if self.head_current_keep_ratios is not None
            else (None,) * len(history)
        )
        groups: dict[tuple[float, float | None], list[int]] = {}
        for head_index, pair in enumerate(zip(history, current, strict=True)):
            groups.setdefault(pair, []).append(head_index)
        return tuple(
            (tuple(groups[pair]), pair[0], pair[1])
            for pair in sorted(
                groups,
                key=lambda item: (
                    item[1] is not None,
                    -1.0 if item[1] is None else item[1],
                    item[0],
                ),
                reverse=True,
            )
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "DynamicPackedHeadGroupBudgetTable":
        return cls(
            head_keep_ratios=payload["head_keep_ratios"],  # type: ignore[arg-type]
            head_current_keep_ratios=payload.get("head_current_keep_ratios"),  # type: ignore[arg-type]
            name=str(payload.get("name", "dynamic_head_groups")),
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "DynamicPackedHeadGroupBudgetTable":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "name": self.name,
            "budget_buckets": list(BUDGET_BUCKETS),
            "num_dit_steps": self.num_dit_steps,
            "num_layers": self.num_layers,
            "num_groups": self.num_groups,
            "num_heads": self.num_heads,
            "group_ratios": list(self.group_ratios),
            "group_current_ratios": list(self.group_current_ratios),
            "head_keep_ratios": [
                [list(heads) for heads in row]
                for row in self.head_keep_ratios
            ],
            **(
                {
                    "head_current_keep_ratios": [
                        [list(heads) for heads in row]
                        for row in self.head_current_keep_ratios
                    ]
                }
                if self.head_current_keep_ratios is not None
                else {}
            ),
        }
