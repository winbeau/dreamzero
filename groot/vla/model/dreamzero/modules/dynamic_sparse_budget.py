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
