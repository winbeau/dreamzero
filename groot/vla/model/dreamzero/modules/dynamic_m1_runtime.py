"""Strictly causal runtime bridge from Packed-proxy M1 to Packed M2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
    DynamicM1GroupedRouter,
    GroupedM1StepDecision,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_online_state import (
    OnlineM1FeatureState,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_packed_observer import (
    PackedM1CausalObserver,
)


class DynamicM1RuntimeController:
    """Route each real DiT from observations finalized by earlier real DiTs."""

    def __init__(
        self,
        bundle: Mapping[str, Any],
        *,
        num_dit_steps: int = 8,
        num_layers: int = 40,
        num_heads: int = 40,
        force_dense_steps: int = 2,
        support_ratio: float = 0.20,
        downstream_risk_table: DownstreamHeadRiskTable | None = None,
        require_downstream_coverage: bool = True,
    ) -> None:
        self.feature_state = OnlineM1FeatureState(
            bundle,
            num_dit_steps=num_dit_steps,
            num_layers=num_layers,
            num_heads=num_heads,
            force_dense_steps=force_dense_steps,
        )
        self.router = DynamicM1GroupedRouter(
            bundle,
            require_downstream_coverage=require_downstream_coverage,
        )
        self.observer = PackedM1CausalObserver(
            num_layers=num_layers,
            num_heads=num_heads,
            support_ratio=support_ratio,
        )
        if downstream_risk_table is not None and (
            downstream_risk_table.num_dit_steps != num_dit_steps
            or downstream_risk_table.num_layers != num_layers
            or downstream_risk_table.num_heads != num_heads
        ):
            raise ValueError("Downstream risk table does not match M1 runtime geometry")
        self.downstream_risk_table = downstream_risk_table
        self.require_downstream_coverage = bool(require_downstream_coverage)
        self.num_dit_steps = num_dit_steps
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.support_ratio = float(support_ratio)
        self._condition_required = bool(
            {"state_l2", "state_abs_mean"} & set(self.feature_state.feature_columns)
        )
        self._active = False
        self._condition_available = False
        self._state_l2 = 0.0
        self._state_abs_mean = 0.0
        self._decisions: list[GroupedM1StepDecision] = []
        self._observations: list[Any | None] = []

    @property
    def active(self) -> bool:
        return self._active

    @property
    def current_decision(self) -> GroupedM1StepDecision | None:
        return self._decisions[-1] if self._decisions else None

    @property
    def decisions(self) -> tuple[GroupedM1StepDecision, ...]:
        return tuple(self._decisions)

    @property
    def observations(self) -> tuple[Any | None, ...]:
        return tuple(self._observations)

    def begin_request(
        self,
        *,
        state_l2: float | None,
        state_abs_mean: float | None,
    ) -> None:
        if self._active:
            self.feature_state.end_request()
            self.observer.end_request()
        condition_values = np.asarray((state_l2, state_abs_mean), dtype=np.float64)
        self._condition_available = bool(
            np.all(np.isfinite(condition_values)) and np.all(condition_values >= 0.0)
        )
        if self._condition_available:
            self._state_l2 = float(condition_values[0])
            self._state_abs_mean = float(condition_values[1])
        else:
            # The estimator still receives finite values, while the explicit
            # fallback below prevents them from authorizing sparse execution.
            self._state_l2 = 0.0
            self._state_abs_mean = 0.0
        self._decisions = []
        self._observations = []
        self.feature_state.begin_request()
        self.observer.begin_request()
        self._active = True

    def _finish_active_step(self) -> Any | None:
        if not self.observer.step_active:
            return None
        observation = self.observer.finish_step()
        dit_index = len(self._observations)
        if observation is not None and observation.dit_index != dit_index:
            raise RuntimeError(
                "Packed M1 observation order diverged from real DiT order"
            )
        self._observations.append(observation)
        self.feature_state.complete_step(dit_index, observation)
        return observation

    def begin_step(
        self,
        *,
        scheduler_index: int,
        dit_index: int,
        scheduler_steps: int,
        diffusion_timestep: int,
    ) -> tuple[Any | None, GroupedM1StepDecision]:
        if not self._active:
            raise RuntimeError("Dynamic M1 request must begin before routing")
        previous_observation = self._finish_active_step()
        if dit_index != len(self._decisions) or len(self._observations) != dit_index:
            raise RuntimeError("Dynamic M1 decisions do not align with real DiT order")
        feature_batch = self.feature_state.features_for_step(
            dit_index=dit_index,
            scheduler_index=scheduler_index,
            scheduler_steps=scheduler_steps,
            diffusion_timestep=diffusion_timestep,
            state_l2=self._state_l2,
            state_abs_mean=self._state_abs_mean,
        )
        condition_fallback = np.full(
            feature_batch.shape,
            self._condition_required and not self._condition_available,
            dtype=bool,
        )
        decision = self.router.route_step(
            feature_batch.features,
            dit_index=dit_index,
            downstream_risk_table=self.downstream_risk_table,
            external_dense_fallback=(feature_batch.dense_fallback | condition_fallback),
        )
        self._decisions.append(decision)
        self.observer.begin_step(dit_index)
        return previous_observation, decision

    def finish_request(self) -> Any | None:
        if not self._active:
            return None
        observation = self._finish_active_step()
        self.observer.end_request()
        self.feature_state.end_request()
        self._active = False
        return observation

    def trace(self) -> dict[str, object]:
        return {
            "active": self._active,
            "num_dit_steps": self.num_dit_steps,
            "condition_required": self._condition_required,
            "condition_available": self._condition_available,
            "support_ratio": self.support_ratio,
            "require_downstream_coverage": self.require_downstream_coverage,
            "downstream_risk_table_attached": self.downstream_risk_table is not None,
            "decision_count": len(self._decisions),
            "observation_count": len(self._observations),
            "valid_observation_count": sum(
                observation is not None for observation in self._observations
            ),
            "decisions": [decision.summary() for decision in self._decisions],
        }
