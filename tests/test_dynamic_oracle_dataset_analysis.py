import numpy as np
import pytest

from benchmarks.analyze_dynamic_attention_oracle_dataset import (
    _compact_m1_rows,
    bootstrap_law_summary,
)


def test_episode_bootstrap_detects_supported_time_and_layer_laws():
    matrices = np.zeros((6, 2, 8, 6), dtype=np.float64)
    for request in range(6):
        for query in range(2):
            for timestep in range(8):
                for layer in range(6):
                    matrices[request, query, timestep, layer] = (
                        1.0 - 0.05 * timestep - 0.02 * layer + 0.001 * request
                    )
    episodes = np.asarray([0, 0, 1, 1, 2, 2])

    summary = bootstrap_law_summary(matrices, episodes, repeats=200, seed=7)

    for query_kind in ("video", "action"):
        time_result = summary["early_minus_late_timestep_budget"][query_kind]
        layer_result = summary["early_minus_late_layer_budget"][query_kind]
        assert time_result["ci95_low"] > 0.0
        assert layer_result["ci95_low"] > 0.0
        assert time_result["positive_fraction"] == 1.0
        assert layer_result["positive_fraction"] == 1.0


def test_compact_m1_rows_use_conservative_label_and_temporal_lags():
    shape = (2, 2, 8, 40, 40)
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name in (
            "budget",
            "turnover",
            "vv_cosine",
            "vv_relative_l2",
            "entropy",
            "max_mass",
        )
    }
    arrays["budget"].fill(0.2)
    arrays["budget"][1, 1, 0, 0, 0] = 0.75
    arrays["vv_relative_l2"][:, :, 0, 0, 0] = 0.3
    arrays["vv_relative_l2"][:, :, 1, 0, 0] = 0.1
    arrays["correlation"] = np.ones((2, 8, 40, 40), dtype=np.float32)
    arrays["diffusion_timestep"] = np.arange(8, dtype=np.int32) * 100
    quality_shape = (*shape, 7)
    arrays["mass_p05"] = np.ones(quality_shape, dtype=np.float32)
    arrays["cosine_p05"] = np.ones(quality_shape, dtype=np.float32)
    arrays["relative_l2_p95"] = np.zeros(quality_shape, dtype=np.float32)
    metadata = {"request_key": "sample"}

    rows = _compact_m1_rows(arrays, metadata)
    first = next(rows)
    for _ in range(40 * 40 - 1):
        next(rows)
    second_timestep = next(rows)

    assert first["oracle_min_keep_ratio"] == 0.75
    assert first["previous_oracle_min_keep_ratio"] == 0.75
    assert second_timestep["oracle_min_keep_ratio"] == pytest.approx(0.2)
    assert second_timestep["previous_oracle_min_keep_ratio"] == 0.75
    assert second_timestep["previous_vv_output_change_relative_l2_max"] == pytest.approx(0.3)
