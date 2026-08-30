import numpy as np

from benchmarks.analyze_dynamic_attention_oracle_dataset import bootstrap_law_summary


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
