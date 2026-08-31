"""Lightweight schema-tagged observations shared by M1 producers/consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKED_M1_OBSERVATION_SCHEMA = "dreamzero-packed-m1-proxy-v2"
PACKED_M1_OBSERVATION_METRICS = (
    "packed_route_support_turnover_max",
    "packed_route_normalized_entropy_mean",
    "packed_route_max_mass_mean",
    "packed_action_output_change_relative_l2_max",
    "packed_action_output_change_cosine_min",
    "packed_cfg_disagreement_relative_l2",
    "packed_action_output_signature_norm",
)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class M1CausalObservation:
    """Schema-specific causal metrics reduced to one value per layer/Head."""

    dit_index: int
    schema: str
    metrics: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.dit_index < 0:
            raise ValueError("causal observation DiT index must be non-negative")
        if not self.schema:
            raise ValueError("causal observation schema must be non-empty")
        if not self.metrics:
            raise ValueError("causal observation metrics must not be empty")
        shape = None
        reduced: dict[str, np.ndarray] = {}
        for name, raw_value in self.metrics.items():
            if not name:
                raise ValueError("causal observation metric names must be non-empty")
            value = np.asarray(raw_value, dtype=np.float64)
            if value.ndim != 2 or not all(value.shape):
                raise ValueError(
                    "causal observation metrics must have [layer, head] shape"
                )
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError("causal observation metric shapes do not align")
            if np.any(np.isinf(value)):
                raise ValueError("causal observation metrics must not contain infinity")
            reduced[str(name)] = _readonly(value)
        object.__setattr__(self, "metrics", reduced)

    @property
    def shape(self) -> tuple[int, int]:
        first = next(iter(self.metrics.values()))
        return tuple(int(value) for value in first.shape)

    def metric(self, name: str) -> np.ndarray:
        try:
            return self.metrics[name]
        except KeyError as error:
            raise KeyError(
                f"Causal observation schema {self.schema!r} has no metric {name}"
            ) from error


def save_packed_m1_observations(
    path: str | Path,
    observations: Sequence[M1CausalObservation | None],
    *,
    request_metadata: Mapping[str, object] | None = None,
) -> None:
    """Persist one request's reduced proxy cube without raw activations."""

    if not observations:
        raise ValueError("Packed M1 observation sequence must not be empty")
    present = np.asarray(
        [observation is not None for observation in observations],
        dtype=bool,
    )
    first = next(
        (observation for observation in observations if observation is not None),
        None,
    )
    if first is None:
        raise ValueError("Packed M1 observation sequence contains no valid step")
    if first.schema != PACKED_M1_OBSERVATION_SCHEMA:
        raise ValueError("Packed M1 observation schema is invalid")
    values = np.full(
        (
            len(observations),
            first.shape[0],
            first.shape[1],
            len(PACKED_M1_OBSERVATION_METRICS),
        ),
        np.nan,
        dtype=np.float32,
    )
    dit_indices = np.arange(len(observations), dtype=np.int64)
    for index, observation in enumerate(observations):
        if observation is None:
            continue
        if observation.schema != first.schema or observation.shape != first.shape:
            raise ValueError("Packed M1 observation sequence is not homogeneous")
        if observation.dit_index != index:
            raise ValueError("Packed M1 observation DiT indices are not contiguous")
        values[index] = np.stack(
            [observation.metric(name) for name in PACKED_M1_OBSERVATION_METRICS],
            axis=-1,
        ).astype(np.float32)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema=np.asarray(first.schema),
        metric_names=np.asarray(PACKED_M1_OBSERVATION_METRICS),
        dit_indices=dit_indices,
        present=present,
        values=values,
        request_metadata_json=np.asarray(
            json.dumps(dict(request_metadata or {}), ensure_ascii=False)
        ),
    )


def load_packed_m1_observations(
    path: str | Path,
) -> tuple[tuple[M1CausalObservation | None, ...], dict[str, object]]:
    """Load a reduced proxy artifact and revalidate its schema/geometry."""

    with np.load(Path(path), allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        metric_names = tuple(str(value) for value in payload["metric_names"])
        dit_indices = np.asarray(payload["dit_indices"], dtype=np.int64)
        present = np.asarray(payload["present"], dtype=bool)
        values = np.asarray(payload["values"], dtype=np.float64)
        metadata = json.loads(str(payload["request_metadata_json"].item()))
    if schema != PACKED_M1_OBSERVATION_SCHEMA:
        raise ValueError("Packed M1 artifact schema is invalid")
    if metric_names != PACKED_M1_OBSERVATION_METRICS:
        raise ValueError("Packed M1 artifact metric order is invalid")
    if values.ndim != 4 or values.shape[0] != len(dit_indices):
        raise ValueError("Packed M1 artifact values have invalid geometry")
    if present.shape != dit_indices.shape or not np.array_equal(
        dit_indices,
        np.arange(len(dit_indices)),
    ):
        raise ValueError("Packed M1 artifact DiT indices are invalid")
    observations: list[M1CausalObservation | None] = []
    for index, is_present in enumerate(present):
        if not is_present:
            observations.append(None)
            continue
        observations.append(
            M1CausalObservation(
                dit_index=index,
                schema=schema,
                metrics={
                    name: values[index, ..., metric_index]
                    for metric_index, name in enumerate(metric_names)
                },
            )
        )
    if not isinstance(metadata, dict):
        raise TypeError("Packed M1 artifact metadata must be an object")
    return tuple(observations), metadata
