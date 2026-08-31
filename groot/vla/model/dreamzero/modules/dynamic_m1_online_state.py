"""Causal online feature state for dynamic M1 routing.

The state deliberately separates feature production from classification.  An
offline M1 bundle may route online only when it declares the exact observation
schema on which it was trained.  Missing or mismatched history never falls
back to a static prior-only sparse route; it produces an explicit Dense
fallback mask instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
    DynamicM1GroupedRouter,
    GroupedM1StepDecision,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_observation import (
    M1CausalObservation,
)

ORACLE_HISTORY_FEATURE_NAMES = (
    "previous_support_turnover_max",
    "previous_vv_output_change_relative_l2_max",
    "previous_two_vv_output_change_relative_l2_max",
    "previous_normalized_entropy_mean",
    "previous_max_attention_mass_mean",
    "previous_qa_qv_key_importance_correlation_mean",
)
PACKED_PROXY_HISTORY_FEATURE_NAMES = (
    "previous_packed_route_support_turnover_max",
    "previous_packed_route_normalized_entropy_mean",
    "previous_packed_route_max_mass_mean",
    "previous_packed_action_output_change_relative_l2_max",
    "previous_two_packed_action_output_change_relative_l2_max",
    "previous_packed_action_output_change_cosine_min",
    "previous_packed_cfg_disagreement_relative_l2",
    "previous_packed_action_output_signature_norm",
)
OBSERVATION_FEATURE_TO_METRIC = {
    "previous_support_turnover_max": "support_turnover_max",
    "previous_vv_output_change_relative_l2_max": ("vv_output_change_relative_l2_max"),
    "previous_normalized_entropy_mean": "normalized_entropy_mean",
    "previous_max_attention_mass_mean": "max_attention_mass_mean",
    "previous_qa_qv_key_importance_correlation_mean": (
        "qa_qv_key_importance_correlation_mean"
    ),
    "previous_packed_route_support_turnover_max": ("packed_route_support_turnover_max"),
    "previous_packed_route_normalized_entropy_mean": (
        "packed_route_normalized_entropy_mean"
    ),
    "previous_packed_route_max_mass_mean": "packed_route_max_mass_mean",
    "previous_packed_action_output_change_relative_l2_max": (
        "packed_action_output_change_relative_l2_max"
    ),
    "previous_packed_action_output_change_cosine_min": (
        "packed_action_output_change_cosine_min"
    ),
    "previous_packed_cfg_disagreement_relative_l2": (
        "packed_cfg_disagreement_relative_l2"
    ),
    "previous_packed_action_output_signature_norm": (
        "packed_action_output_signature_norm"
    ),
}
SUPPORTED_FEATURE_NAMES = frozenset(
    {
        "timestep_position",
        "layer_depth",
        "head_position",
        "scheduler_position",
        "diffusion_timestep_scaled",
        "state_l2",
        "state_abs_mean",
        "history_one_available",
        "history_two_available",
        *ORACLE_HISTORY_FEATURE_NAMES,
        *PACKED_PROXY_HISTORY_FEATURE_NAMES,
        "vv_change_acceleration",
        "packed_action_output_change_acceleration",
        "prior_budget_mean_tlh",
        "prior_budget_std_tlh",
        "prior_critical_rate_tlh",
    }
)
PRIOR_KEYS = ("dit_index", "layer_index", "head_index")
PRIOR_VALUE_COLUMNS = (
    "prior_budget_mean_tlh",
    "prior_budget_std_tlh",
    "prior_critical_rate_tlh",
)
CAUSALLY_UNDEFINED_FEATURE_NAMES = frozenset(
    {
        "previous_two_vv_output_change_relative_l2_max",
        "previous_two_packed_action_output_change_relative_l2_max",
        "vv_change_acceleration",
        "packed_action_output_change_acceleration",
    }
)


def _readonly(array: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class M1HistoricalObservation:
    """Reduced per-Head signals observed after one completed real DiT."""

    dit_index: int
    schema: str
    support_turnover_max: np.ndarray
    vv_output_change_relative_l2_max: np.ndarray
    normalized_entropy_mean: np.ndarray
    max_attention_mass_mean: np.ndarray
    qa_qv_key_importance_correlation_mean: np.ndarray

    def __post_init__(self) -> None:
        if self.dit_index < 0:
            raise ValueError("historical observation DiT index must be non-negative")
        if not self.schema:
            raise ValueError("historical observation schema must be non-empty")
        names = (
            "support_turnover_max",
            "vv_output_change_relative_l2_max",
            "normalized_entropy_mean",
            "max_attention_mass_mean",
            "qa_qv_key_importance_correlation_mean",
        )
        shape = np.asarray(self.support_turnover_max).shape
        if len(shape) != 2 or not all(shape):
            raise ValueError("historical observations must have [layer, head] shape")
        for name in names:
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"historical observation {name} shape does not align")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"historical observation {name} must be finite")
            object.__setattr__(self, name, _readonly(value))
        if np.any(self.support_turnover_max < 0.0) or np.any(
            self.support_turnover_max > 1.0
        ):
            raise ValueError("support turnover must lie in [0, 1]")
        if np.any(self.vv_output_change_relative_l2_max < 0.0):
            raise ValueError("VV relative L2 must be non-negative")
        if np.any(self.normalized_entropy_mean < 0.0) or np.any(
            self.normalized_entropy_mean > 1.0
        ):
            raise ValueError("normalized entropy must lie in [0, 1]")
        if np.any(self.max_attention_mass_mean < 0.0) or np.any(
            self.max_attention_mass_mean > 1.0
        ):
            raise ValueError("maximum attention mass must lie in [0, 1]")
        if np.any(self.qa_qv_key_importance_correlation_mean < -1.0) or np.any(
            self.qa_qv_key_importance_correlation_mean > 1.0
        ):
            raise ValueError("Qa/Qv correlation must lie in [-1, 1]")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.support_turnover_max.shape)

    def metric(self, name: str) -> np.ndarray:
        if not hasattr(self, name):
            raise KeyError(f"Dense Oracle observation has no metric {name}")
        return np.asarray(getattr(self, name), dtype=np.float64)


M1Observation = M1HistoricalObservation | M1CausalObservation


@dataclass(frozen=True)
class OnlineM1FeatureBatch:
    """Causal feature cube plus explicit reasons that require Dense."""

    dit_index: int
    features: Mapping[str, np.ndarray]
    forced_early_dense: np.ndarray
    missing_contract_fallback: np.ndarray
    missing_history_fallback: np.ndarray
    schema_mismatch_fallback: np.ndarray

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("online M1 feature batch must not be empty")
        feature_shapes = {np.asarray(value).shape for value in self.features.values()}
        if len(feature_shapes) != 1:
            raise ValueError("online M1 feature shapes do not align")
        shape = next(iter(feature_shapes))
        if len(shape) != 2 or not all(shape):
            raise ValueError("online M1 features must have [layer, head] shape")
        object.__setattr__(
            self,
            "features",
            {
                key: _readonly(feature_value)
                for key, feature_value in self.features.items()
            },
        )
        for name in (
            "forced_early_dense",
            "missing_contract_fallback",
            "missing_history_fallback",
            "schema_mismatch_fallback",
        ):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"online M1 fallback {name} shape does not align")
            object.__setattr__(self, name, _readonly(value, dtype=bool))

    @property
    def shape(self) -> tuple[int, int]:
        first = next(iter(self.features.values()))
        return tuple(int(value) for value in first.shape)

    @property
    def dense_fallback(self) -> np.ndarray:
        return (
            self.forced_early_dense
            | self.missing_contract_fallback
            | self.missing_history_fallback
            | self.schema_mismatch_fallback
        )

    def summary(self) -> dict[str, object]:
        return {
            "dit_index": self.dit_index,
            "shape": list(self.shape),
            "forced_early_dense_rate": float(np.mean(self.forced_early_dense)),
            "missing_contract_fallback_rate": float(
                np.mean(self.missing_contract_fallback)
            ),
            "missing_history_fallback_rate": float(
                np.mean(self.missing_history_fallback)
            ),
            "schema_mismatch_fallback_rate": float(
                np.mean(self.schema_mismatch_fallback)
            ),
            "combined_feature_fallback_rate": float(np.mean(self.dense_fallback)),
        }


class OnlineM1FeatureState:
    """Request-local causal history with strict observer provenance gates."""

    def __init__(
        self,
        bundle: Mapping[str, Any],
        *,
        num_dit_steps: int = 8,
        num_layers: int = 40,
        num_heads: int = 40,
        force_dense_steps: int = 2,
    ) -> None:
        if min(num_dit_steps, num_layers, num_heads) <= 0:
            raise ValueError("online M1 geometry must be positive")
        if not 0 <= force_dense_steps <= num_dit_steps:
            raise ValueError("force_dense_steps is outside the DiT range")
        feature_columns = tuple(str(name) for name in bundle.get("feature_columns", ()))
        if not feature_columns:
            raise ValueError("online M1 bundle has no feature columns")
        unsupported = set(feature_columns) - SUPPORTED_FEATURE_NAMES
        if unsupported:
            raise ValueError(
                f"online M1 bundle uses unsupported features: {sorted(unsupported)}"
            )
        prior_table = bundle.get("prior_table")
        if not isinstance(prior_table, pd.DataFrame):
            raise TypeError("online M1 bundle prior_table must be a pandas DataFrame")
        required_prior = {*PRIOR_KEYS, *PRIOR_VALUE_COLUMNS}
        missing_prior = required_prior - set(prior_table.columns)
        if missing_prior:
            raise ValueError(
                f"online M1 prior table is missing: {sorted(missing_prior)}"
            )
        if prior_table.duplicated(list(PRIOR_KEYS)).any():
            raise ValueError("online M1 prior table contains duplicate grid rows")
        ordered = prior_table.sort_values(list(PRIOR_KEYS))
        expected = pd.MultiIndex.from_product(
            (range(num_dit_steps), range(num_layers), range(num_heads)),
            names=PRIOR_KEYS,
        )
        actual = pd.MultiIndex.from_frame(ordered[list(PRIOR_KEYS)])
        if len(expected.difference(actual)) or len(actual.difference(expected)):
            raise ValueError("online M1 prior table does not cover the full grid")

        contract = bundle.get("online_observation_schema")
        if contract is None:
            accepted_schemas: tuple[str, ...] = ()
        elif isinstance(contract, str):
            accepted_schemas = (contract,)
        elif isinstance(contract, Sequence):
            accepted_schemas = tuple(str(value) for value in contract)
        else:
            raise TypeError("online_observation_schema must be a string or sequence")
        if len(set(accepted_schemas)) != len(accepted_schemas) or any(
            not value for value in accepted_schemas
        ):
            raise ValueError("online observation schemas must be non-empty and unique")

        self.feature_columns = feature_columns
        self.accepted_observation_schemas = accepted_schemas
        self.num_dit_steps = num_dit_steps
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.force_dense_steps = force_dense_steps
        self.priors = {
            name: _readonly(
                ordered[name]
                .to_numpy(dtype=np.float64)
                .reshape(num_dit_steps, num_layers, num_heads)
            )
            for name in PRIOR_VALUE_COLUMNS
        }
        self._active = False
        self._completed_steps: dict[int, M1Observation | None] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_layers, self.num_heads

    def begin_request(self) -> None:
        self._completed_steps.clear()
        self._active = True

    def end_request(self) -> None:
        self._completed_steps.clear()
        self._active = False

    def complete_step(
        self,
        dit_index: int,
        observation: M1Observation | None = None,
    ) -> None:
        """Advance causal state after one real DiT, with or without a probe."""

        if not self._active:
            raise RuntimeError("begin_request must precede online M1 observation")
        expected_index = len(self._completed_steps)
        if dit_index != expected_index:
            raise ValueError(
                f"Expected completion for DiT {expected_index}, got {dit_index}"
            )
        if observation is not None:
            if observation.dit_index != dit_index:
                raise ValueError(
                    "online M1 observation index does not match completion"
                )
            if observation.shape != self.shape:
                raise ValueError("online M1 observation does not match model geometry")
        self._completed_steps[dit_index] = observation

    def observe(self, observation: M1Observation) -> None:
        self.complete_step(observation.dit_index, observation)

    def _accepted(self, observation: M1Observation | None) -> bool:
        return bool(
            observation is not None
            and observation.schema in self.accepted_observation_schemas
        )

    def features_for_step(
        self,
        *,
        dit_index: int,
        scheduler_index: int,
        scheduler_steps: int,
        diffusion_timestep: int,
        state_l2: float,
        state_abs_mean: float,
    ) -> OnlineM1FeatureBatch:
        if not self._active:
            raise RuntimeError("begin_request must precede online M1 routing")
        if not 0 <= dit_index < self.num_dit_steps:
            raise ValueError("online M1 DiT index is outside the configured range")
        if len(self._completed_steps) != dit_index:
            raise RuntimeError(
                "online M1 routing must follow every preceding real DiT completion"
            )
        if scheduler_steps <= 1 or not 0 <= scheduler_index < scheduler_steps:
            raise ValueError("online M1 scheduler context is invalid")
        if not np.isfinite(state_l2) or state_l2 < 0.0:
            raise ValueError("online M1 state_l2 must be finite and non-negative")
        if not np.isfinite(state_abs_mean) or state_abs_mean < 0.0:
            raise ValueError("online M1 state_abs_mean must be finite and non-negative")

        previous = self._completed_steps.get(dit_index - 1)
        previous_two = self._completed_steps.get(dit_index - 2)
        previous_accepted = self._accepted(previous)
        previous_two_accepted = self._accepted(previous_two)
        shape = self.shape
        nan = np.full(shape, np.nan, dtype=np.float64)

        def observation_value(
            observation: M1Observation | None,
            *,
            accepted: bool,
            metric_name: str,
        ) -> np.ndarray:
            if not accepted or observation is None:
                return nan
            try:
                return observation.metric(metric_name)
            except KeyError:
                return nan

        def previous_feature(name: str) -> np.ndarray:
            if not previous_accepted or previous is None:
                return nan
            metric_name = OBSERVATION_FEATURE_TO_METRIC[name]
            return observation_value(
                previous,
                accepted=True,
                metric_name=metric_name,
            )

        history_features = {
            name: previous_feature(name) for name in OBSERVATION_FEATURE_TO_METRIC
        }
        previous_two_vv = observation_value(
            previous_two,
            accepted=previous_two_accepted,
            metric_name="vv_output_change_relative_l2_max",
        )
        previous_two_packed_action = observation_value(
            previous_two,
            accepted=previous_two_accepted,
            metric_name="packed_action_output_change_relative_l2_max",
        )

        layer_position = np.arange(self.num_layers, dtype=np.float64)[:, None] / max(
            1, self.num_layers - 1
        )
        head_position = np.arange(self.num_heads, dtype=np.float64)[None, :] / max(
            1, self.num_heads - 1
        )
        full = lambda value: np.full(shape, value, dtype=np.float64)
        all_features = {
            "timestep_position": full(dit_index / max(1, self.num_dit_steps - 1)),
            "layer_depth": np.broadcast_to(layer_position, shape),
            "head_position": np.broadcast_to(head_position, shape),
            "scheduler_position": full(scheduler_index / (scheduler_steps - 1)),
            "diffusion_timestep_scaled": full(diffusion_timestep / 1000.0),
            "state_l2": full(state_l2),
            "state_abs_mean": full(state_abs_mean),
            "history_one_available": full(float(previous_accepted)),
            "history_two_available": full(float(previous_two_accepted)),
            **history_features,
            "previous_two_vv_output_change_relative_l2_max": previous_two_vv,
            "previous_two_packed_action_output_change_relative_l2_max": (
                previous_two_packed_action
            ),
            "vv_change_acceleration": np.abs(
                history_features["previous_vv_output_change_relative_l2_max"]
                - previous_two_vv
            ),
            "packed_action_output_change_acceleration": np.abs(
                history_features["previous_packed_action_output_change_relative_l2_max"]
                - previous_two_packed_action
            ),
            **{name: self.priors[name][dit_index] for name in PRIOR_VALUE_COLUMNS},
        }
        features = {name: all_features[name] for name in self.feature_columns}
        nonfinite_features = np.zeros(shape, dtype=bool)
        for name, value in features.items():
            if name not in CAUSALLY_UNDEFINED_FEATURE_NAMES:
                nonfinite_features |= ~np.isfinite(value)

        forced_early = np.full(
            shape,
            dit_index < self.force_dense_steps,
            dtype=bool,
        )
        missing_contract = np.full(
            shape,
            not self.accepted_observation_schemas,
            dtype=bool,
        )
        history_required = dit_index >= self.force_dense_steps
        missing_history = np.full(
            shape,
            history_required
            and not (previous is not None and previous_two is not None),
            dtype=bool,
        )
        if history_required:
            missing_history |= nonfinite_features
        schema_mismatch = np.full(
            shape,
            history_required
            and (
                (previous is not None and not previous_accepted)
                or (previous_two is not None and not previous_two_accepted)
            ),
            dtype=bool,
        )
        return OnlineM1FeatureBatch(
            dit_index=dit_index,
            features=features,
            forced_early_dense=forced_early,
            missing_contract_fallback=missing_contract,
            missing_history_fallback=missing_history,
            schema_mismatch_fallback=schema_mismatch,
        )

    def route_step(
        self,
        router: DynamicM1GroupedRouter,
        feature_batch: OnlineM1FeatureBatch,
        *,
        downstream_risk_table: DownstreamHeadRiskTable | None = None,
    ) -> GroupedM1StepDecision:
        if feature_batch.shape != self.shape:
            raise ValueError("online M1 feature batch does not match state geometry")
        return router.route_step(
            feature_batch.features,
            dit_index=feature_batch.dit_index,
            downstream_risk_table=downstream_risk_table,
            external_dense_fallback=feature_batch.dense_fallback,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "active": self._active,
            "completed_steps": tuple(self._completed_steps.items()),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        completed_steps = snapshot.get("completed_steps")
        if not isinstance(completed_steps, tuple):
            raise TypeError("invalid online M1 snapshot")
        restored: dict[int, M1Observation | None] = {}
        for entry in completed_steps:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], int)
                or (
                    entry[1] is not None
                    and not isinstance(
                        entry[1],
                        (M1HistoricalObservation, M1CausalObservation),
                    )
                )
            ):
                raise ValueError("invalid online M1 snapshot")
            restored[entry[0]] = entry[1]
        if set(restored) != set(range(len(restored))):
            raise ValueError("online M1 snapshot completions are not contiguous")
        if any(
            value is not None and value.shape != self.shape
            for value in restored.values()
        ):
            raise ValueError("online M1 snapshot geometry does not match state")
        if any(
            value is not None and value.dit_index != index
            for index, value in restored.items()
        ):
            raise ValueError("online M1 snapshot observation indices do not align")
        self._completed_steps = restored
        self._active = bool(snapshot.get("active", False))
