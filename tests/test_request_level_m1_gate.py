import numpy as np
import pandas as pd
import pytest

from benchmarks.train_request_level_m1_gate import (
    aggregate_request_features,
    build_profile_labels,
    choose_risk_thresholds,
    evaluate_realized_route,
)


def _report(label, actions, latencies):
    records = []
    for index, (action, latency) in enumerate(zip(actions, latencies, strict=True)):
        records.append(
            {
                "request_index": index,
                "request_key": f"request-{index}",
                "phase": "measured",
                "split": "train",
                "source_episode_index": index,
                "trajectory_stage": ("early", "middle", "late")[index],
                "latency_seconds": latency,
                "action": action,
            }
        )
    return {"label": label, "seed": 7, "records": records}


def test_profile_labels_choose_fastest_quality_safe_route():
    dense = _report("dense", [[1.0, 0.0]] * 3, [1.0, 1.0, 1.0])
    balanced = _report(
        "balanced",
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        [0.6, 0.6, 0.6],
    )
    conservative = _report(
        "conservative",
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [0.8, 0.8, 0.8],
    )

    labels = build_profile_labels(dense, balanced, conservative)

    assert labels["target_profile"].tolist() == [
        "balanced",
        "conservative",
        "dense",
    ]
    assert labels["target_profile_index"].tolist() == [0, 1, 2]


def test_risk_thresholds_preserve_zero_false_sparse_on_validation():
    probabilities = np.asarray(
        [
            [0.90, 0.08, 0.02],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
            [0.02, 0.08, 0.90],
        ]
    )
    truth = np.asarray([0, 1, 2, 2])

    _, prediction, metrics = choose_risk_thresholds(
        probabilities,
        truth,
        false_sparse_limit=0.01,
    )

    assert prediction.tolist() == truth.tolist()
    assert metrics["false_sparse_count"] == 0
    assert metrics["dense_fallback_rate"] == pytest.approx(0.5)


def test_request_features_use_only_first_two_historical_dit_changes():
    frame = pd.DataFrame(
        {
            "request_key": ["request"] * 4,
            "split": ["validation"] * 4,
            "source_episode_index": [9] * 4,
            "trajectory_stage": ["middle"] * 4,
            "trajectory_fraction": [0.5] * 4,
            "trajectory_length": [100] * 4,
            "state_l2": [2.0] * 4,
            "state_abs_mean": [0.2] * 4,
            "dit_index": [2, 2, 7, 7],
            "layer_index": [0, 27, 0, 27],
            "previous_support_turnover_max": [0.1, 0.2, 9.0, 9.0],
            "previous_vv_output_change_relative_l2_max": [0.3, 0.5, 99.0, 99.0],
            "previous_two_vv_output_change_relative_l2_max": [0.2, 0.4, 88.0, 88.0],
            "previous_normalized_entropy_mean": [0.6, 0.8, 7.0, 7.0],
            "previous_max_attention_mass_mean": [0.1, 0.2, 8.0, 8.0],
            "previous_qa_qv_key_importance_correlation_mean": [0.7, 0.9, -8.0, -8.0],
        }
    )
    probabilities = np.asarray(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],
        ]
    )
    result = {
        "probabilities": probabilities,
        "prediction": np.asarray([1, 2, 0, 0]),
        "raw_prediction": np.asarray([1, 2, 0, 0]),
        "route_confidence": np.asarray([0.8, 0.7, 0.9, 0.9]),
        "fallback": np.asarray([False, True, False, False]),
    }

    features = aggregate_request_features(
        frame,
        result,
        np.asarray([0.5, 0.75, 1.0]),
    )

    assert len(features) == 1
    assert features["history_vv_change_mean"].item() == pytest.approx(0.4)
    assert features["history_two_vv_change_mean"].item() == pytest.approx(0.3)
    assert features["decision_late_layer_vv_change_p95"].item() == pytest.approx(0.5)
    assert features["m1_fallback_rate"].item() == pytest.approx(0.5)


def test_realized_route_uses_dense_values_for_fallback():
    labels = pd.DataFrame(
        {
            "request_key": ["balanced", "conservative", "dense"],
            "dense_latency_seconds": [1.0, 1.0, 1.0],
            "balanced_latency_seconds": [0.5, 0.5, 0.5],
            "conservative_latency_seconds": [0.8, 0.8, 0.8],
            "balanced_cosine": [1.0, 0.9, 0.9],
            "balanced_relative_l2": [0.0, 0.2, 0.2],
            "conservative_cosine": [1.0, 1.0, 0.9],
            "conservative_relative_l2": [0.0, 0.0, 0.2],
        }
    )

    metrics = evaluate_realized_route(
        labels,
        np.asarray([0, 1, 2]),
        cosine_threshold=0.999,
        relative_l2_threshold=0.05,
    )

    assert metrics["quality_failure_count"] == 0
    assert metrics["action_cosine_min"] == 1.0
    assert metrics["mixed_e2e_speedup"] == pytest.approx(3.0 / 2.3)
