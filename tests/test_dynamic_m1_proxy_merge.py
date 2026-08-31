import numpy as np
import pytest

from benchmarks.merge_dynamic_m1_packed_proxy_features import (
    PROXY_FEATURE_COLUMNS,
    merge_proxy_features,
    proxy_feature_frame,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_observation import (
    PACKED_M1_OBSERVATION_METRICS,
    PACKED_M1_OBSERVATION_SCHEMA,
    M1CausalObservation,
    save_packed_m1_observations,
)


def _observation(dit_index: int) -> M1CausalObservation:
    return M1CausalObservation(
        dit_index=dit_index,
        schema=PACKED_M1_OBSERVATION_SCHEMA,
        metrics={
            name: np.full((2, 2), dit_index + metric_index / 10.0)
            for metric_index, name in enumerate(PACKED_M1_OBSERVATION_METRICS)
        },
    )


def test_proxy_feature_frame_is_strictly_shifted_causal_history(tmp_path) -> None:
    path = tmp_path / "request.npz"
    save_packed_m1_observations(
        path,
        tuple(_observation(index) for index in range(3)),
        request_metadata={"request_key": "request-0"},
    )

    frame = proxy_feature_frame(path, expected_dit_steps=3)
    step0 = frame.loc[frame["dit_index"] == 0]
    step1 = frame.loc[frame["dit_index"] == 1]
    step2 = frame.loc[frame["dit_index"] == 2]

    assert len(frame) == 3 * 2 * 2
    assert step0[list(PROXY_FEATURE_COLUMNS)].isna().all().all()
    assert np.allclose(
        step1["previous_packed_action_output_signature_norm"],
        0.6,
    )
    assert np.allclose(
        step2["previous_packed_action_output_signature_norm"],
        1.6,
    )
    assert np.allclose(
        step2["previous_two_packed_action_output_change_relative_l2_max"],
        0.3,
    )


def test_proxy_merge_requires_complete_one_to_one_grid(tmp_path) -> None:
    path = tmp_path / "request.npz"
    save_packed_m1_observations(
        path,
        tuple(_observation(index) for index in range(3)),
        request_metadata={"request_key": "request-0"},
    )
    proxy = proxy_feature_frame(path, expected_dit_steps=3)
    oracle = proxy[["request_key", "dit_index", "layer_index", "head_index"]].copy()
    oracle["oracle_min_keep_ratio"] = 1.0

    merged = merge_proxy_features(oracle, proxy)
    assert len(merged) == len(oracle)
    assert set(PROXY_FEATURE_COLUMNS).issubset(merged.columns)

    with pytest.raises(ValueError, match="coverage is missing"):
        merge_proxy_features(oracle, proxy.iloc[:-1])
