import numpy as np
import pandas as pd
import pytest

from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    RoutePolicy,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DynamicM1GroupedRouter,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_online_state import (
    M1HistoricalObservation,
    OnlineM1FeatureState,
)

FEATURE_COLUMNS = (
    "timestep_position",
    "layer_depth",
    "head_position",
    "history_one_available",
    "history_two_available",
    "previous_support_turnover_max",
    "previous_vv_output_change_relative_l2_max",
    "previous_two_vv_output_change_relative_l2_max",
    "vv_change_acceleration",
    "prior_budget_mean_tlh",
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


def _prior_table(num_dit_steps=3, num_layers=2, num_heads=2):
    rows = []
    for dit_index in range(num_dit_steps):
        for layer_index in range(num_layers):
            for head_index in range(num_heads):
                rows.append(
                    {
                        "dit_index": dit_index,
                        "layer_index": layer_index,
                        "head_index": head_index,
                        "prior_budget_mean_tlh": 0.5,
                        "prior_budget_std_tlh": 0.1,
                        "prior_critical_rate_tlh": 0.2,
                    }
                )
    return pd.DataFrame(rows)


def _bundle(*, schema="dense-oracle-v1"):
    bundle = {
        "estimator": ConstantSparseEstimator(),
        "feature_columns": FEATURE_COLUMNS,
        "budget_buckets": BUDGET_BUCKETS,
        "policy": RoutePolicy(confidence_threshold=0.8, promotion_buckets=0),
        "prior_table": _prior_table(),
    }
    if schema is not None:
        bundle["online_observation_schema"] = schema
    return bundle


def _observation(dit_index, *, schema="dense-oracle-v1", vv_change=0.1):
    shape = (2, 2)
    return M1HistoricalObservation(
        dit_index=dit_index,
        schema=schema,
        support_turnover_max=np.full(shape, 0.1 + 0.01 * dit_index),
        vv_output_change_relative_l2_max=np.full(shape, vv_change),
        normalized_entropy_mean=np.full(shape, 0.5),
        max_attention_mass_mean=np.full(shape, 0.2),
        qa_qv_key_importance_correlation_mean=np.full(shape, 0.8),
    )


def _features(state, dit_index):
    return state.features_for_step(
        dit_index=dit_index,
        scheduler_index=dit_index,
        scheduler_steps=16,
        diffusion_timestep=1000 - 100 * dit_index,
        state_l2=2.0,
        state_abs_mean=0.25,
    )


def test_online_state_uses_only_completed_causal_history() -> None:
    state = OnlineM1FeatureState(
        _bundle(),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    state.begin_request()

    step0 = _features(state, 0)
    assert np.all(step0.forced_early_dense)
    state.observe(_observation(0, vv_change=0.10))

    step1 = _features(state, 1)
    assert np.all(step1.forced_early_dense)
    assert np.all(step1.features["history_one_available"] == 1.0)
    assert np.all(step1.features["history_two_available"] == 0.0)
    state.observe(_observation(1, vv_change=0.25))

    step2 = _features(state, 2)
    assert not np.any(step2.dense_fallback)
    assert np.all(step2.features["previous_vv_output_change_relative_l2_max"] == 0.25)
    assert np.all(
        step2.features["previous_two_vv_output_change_relative_l2_max"] == 0.10
    )
    assert np.allclose(step2.features["vv_change_acceleration"], 0.15)


def test_missing_or_mismatched_observer_contract_forces_dense() -> None:
    no_contract = OnlineM1FeatureState(
        _bundle(schema=None),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    no_contract.begin_request()
    assert np.all(_features(no_contract, 0).missing_contract_fallback)

    mismatch = OnlineM1FeatureState(
        _bundle(),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    mismatch.begin_request()
    mismatch.observe(_observation(0, schema="packed-proxy-v1"))
    _features(mismatch, 1)
    mismatch.observe(_observation(1))
    step2 = _features(mismatch, 2)
    assert np.all(step2.schema_mismatch_fallback)
    assert np.all(step2.dense_fallback)


def test_missing_probe_advances_real_dit_but_forces_later_dense() -> None:
    state = OnlineM1FeatureState(
        _bundle(),
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    state.begin_request()
    _features(state, 0)
    state.complete_step(0)
    _features(state, 1)
    state.observe(_observation(1))

    step2 = _features(state, 2)
    assert np.all(step2.missing_history_fallback)
    assert np.all(step2.dense_fallback)


def test_online_feature_fallback_is_enforced_by_grouped_router() -> None:
    bundle = _bundle()
    state = OnlineM1FeatureState(
        bundle,
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    router = DynamicM1GroupedRouter(bundle, require_downstream_coverage=False)
    state.begin_request()

    step0 = _features(state, 0)
    decision0 = state.route_step(router, step0)
    assert np.all(decision0.feature_fallback)
    assert np.all(decision0.keep_ratios == 1.0)
    state.observe(_observation(0, vv_change=0.10))

    step1 = _features(state, 1)
    assert np.all(state.route_step(router, step1).keep_ratios == 1.0)
    state.observe(_observation(1, vv_change=0.20))

    decision2 = state.route_step(router, _features(state, 2))
    assert not np.any(decision2.fallback)
    assert np.all(decision2.keep_ratios == 0.25)


def test_online_state_rejects_out_of_order_and_restores_missing_probes() -> None:
    bundle = _bundle()
    state = OnlineM1FeatureState(
        bundle,
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    state.begin_request()
    with pytest.raises(ValueError, match="Expected completion for DiT 0"):
        state.complete_step(1)
    state.complete_step(0)
    snapshot = state.snapshot()

    restored = OnlineM1FeatureState(
        bundle,
        num_dit_steps=3,
        num_layers=2,
        num_heads=2,
    )
    restored.restore(snapshot)
    step1 = _features(restored, 1)
    assert np.all(step1.forced_early_dense)
    restored.end_request()
    with pytest.raises(RuntimeError, match="begin_request"):
        _features(restored, 0)
