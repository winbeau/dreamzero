import numpy as np
import pandas as pd
import torch

from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    RoutePolicy,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_runtime import (
    DynamicM1RuntimeController,
)


class ConstantSparseEstimator:
    classes_ = np.arange(len(BUDGET_BUCKETS))

    def predict_proba(self, features):
        probabilities = np.full(
            (len(features), len(BUDGET_BUCKETS)),
            0.01 / (len(BUDGET_BUCKETS) - 1),
        )
        probabilities[:, 0] = 0.99
        return probabilities


def _bundle(num_dit_steps=3, num_layers=2, num_heads=2):
    prior_rows = []
    for dit_index in range(num_dit_steps):
        for layer_index in range(num_layers):
            for head_index in range(num_heads):
                prior_rows.append(
                    {
                        "dit_index": dit_index,
                        "layer_index": layer_index,
                        "head_index": head_index,
                        "prior_budget_mean_tlh": 0.5,
                        "prior_budget_std_tlh": 0.1,
                        "prior_critical_rate_tlh": 0.2,
                    }
                )
    return {
        "estimator": ConstantSparseEstimator(),
        "feature_columns": (
            "timestep_position",
            "layer_depth",
            "head_position",
            "state_l2",
            "state_abs_mean",
            "history_one_available",
            "history_two_available",
            "previous_packed_action_output_change_relative_l2_max",
            "previous_two_packed_action_output_change_relative_l2_max",
            "packed_action_output_change_acceleration",
            "prior_budget_mean_tlh",
        ),
        "budget_buckets": BUDGET_BUCKETS,
        "policy": RoutePolicy(confidence_threshold=0.8, promotion_buckets=0),
        "prior_table": pd.DataFrame(prior_rows),
        "online_observation_schema": "dreamzero-packed-m1-proxy-v2",
    }


def _record_step(controller, value):
    for branch_index, branch in enumerate(("conditional", "unconditional")):
        controller.observer.observe_route_scores(
            torch.tensor(
                [[[value + branch_index, value + branch_index + 1.0, 0.0, -1.0]]]
            ),
            cfg_branch=branch,
        )
        for layer_index in range(2):
            controller.observer.observe_action_output(
                layer_index=layer_index,
                cfg_branch=branch,
                action_output=torch.full(
                    (1, 3, 2, 4),
                    value + branch_index + layer_index + 1.0,
                ),
            )


def test_runtime_routes_only_from_completed_proxy_history():
    controller = DynamicM1RuntimeController(
        _bundle(),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
        require_downstream_coverage=False,
    )
    controller.begin_request(state_l2=2.0, state_abs_mean=0.25)

    previous0, decision0 = controller.begin_step(
        scheduler_index=0,
        dit_index=0,
        scheduler_steps=16,
        diffusion_timestep=999,
    )
    assert previous0 is None
    assert np.all(decision0.keep_ratios == 1.0)
    _record_step(controller, 0.0)

    previous1, decision1 = controller.begin_step(
        scheduler_index=2,
        dit_index=1,
        scheduler_steps=16,
        diffusion_timestep=800,
    )
    assert previous1 is not None and previous1.dit_index == 0
    assert np.all(decision1.keep_ratios == 1.0)
    _record_step(controller, 1.0)

    previous2, decision2 = controller.begin_step(
        scheduler_index=4,
        dit_index=2,
        scheduler_steps=16,
        diffusion_timestep=600,
    )
    assert previous2 is not None and previous2.dit_index == 1
    assert not np.any(decision2.fallback)
    assert np.all(decision2.keep_ratios == 0.25)
    assert (
        max(
            len(decision2.execution_groups_for_layer(layer_index))
            for layer_index in range(2)
        )
        <= 4
    )
    _record_step(controller, 2.0)

    final = controller.finish_request()
    assert final is not None and final.dit_index == 2
    assert len(controller.decisions) == 3
    assert len(controller.observations) == 3
    assert controller.trace()["valid_observation_count"] == 3


def test_runtime_missing_raw_state_condition_forces_dense():
    controller = DynamicM1RuntimeController(
        _bundle(),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
        require_downstream_coverage=False,
    )
    controller.begin_request(state_l2=None, state_abs_mean=None)
    for dit_index in range(3):
        _previous, decision = controller.begin_step(
            scheduler_index=dit_index * 2,
            dit_index=dit_index,
            scheduler_steps=16,
            diffusion_timestep=999 - dit_index * 200,
        )
        assert np.all(decision.feature_fallback)
        assert np.all(decision.keep_ratios == 1.0)
        _record_step(controller, float(dit_index))
    controller.finish_request()
    assert controller.trace()["condition_required"]
    assert not controller.trace()["condition_available"]


def test_runtime_requires_downstream_coverage_by_default():
    controller = DynamicM1RuntimeController(
        _bundle(),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    controller.begin_request(state_l2=2.0, state_abs_mean=0.25)
    for dit_index in range(3):
        _previous, decision = controller.begin_step(
            scheduler_index=dit_index * 2,
            dit_index=dit_index,
            scheduler_steps=16,
            diffusion_timestep=999 - dit_index * 200,
        )
        assert np.all(decision.keep_ratios == 1.0)
        assert np.all(decision.downstream_unknown_fallback)
        _record_step(controller, float(dit_index))
    controller.finish_request()
