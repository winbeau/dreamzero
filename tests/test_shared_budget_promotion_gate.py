import numpy as np
import pandas as pd

from benchmarks.train_shared_budget_promotion_gate import (
    aggregate_shared_gate_features,
    choose_dense_threshold,
    episode_cross_validated_route,
)
from sklearn.linear_model import LogisticRegression


def _proxy_frame() -> pd.DataFrame:
    rows = []
    for request_index, request_key in enumerate(("safe", "unsafe")):
        for dit_index in range(8):
            for layer_index in range(3):
                for head_index in range(2):
                    rows.append(
                        {
                            "request_key": request_key,
                            "split": "train",
                            "source_episode_index": request_index,
                            "state_l2": 1.0 + request_index,
                            "state_abs_mean": 0.1 + request_index,
                            "dit_index": dit_index,
                            "layer_index": layer_index,
                            "head_index": head_index,
                            "previous_packed_route_support_turnover_max": 0.1,
                            "previous_packed_route_normalized_entropy_mean": 0.2,
                            "previous_packed_route_max_mass_mean": 0.8,
                            "previous_packed_action_output_change_relative_l2_max": 0.03,
                            "previous_two_packed_action_output_change_relative_l2_max": 0.02,
                            "previous_packed_action_output_change_cosine_min": 0.99,
                            "previous_packed_cfg_disagreement_relative_l2": 0.01,
                            "previous_packed_action_output_signature_norm": 2.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_aggregate_shared_gate_features_uses_dit_two_only() -> None:
    frame = _proxy_frame()
    row_count = len(frame)
    probabilities = np.full((row_count, 3), (0.2, 0.3, 0.5))
    prediction = np.full(row_count, 2, dtype=np.int64)
    result = {
        "probabilities": probabilities,
        "prediction": prediction,
        "fallback": np.zeros(row_count, dtype=bool),
        "route_confidence": np.full(row_count, 0.9),
    }

    features = aggregate_shared_gate_features(
        frame,
        result,
        np.asarray((0.25, 0.50, 1.00)),
    )

    assert features["request_key"].tolist() == ["safe", "unsafe"]
    assert features["state_l2"].tolist() == [1.0, 2.0]
    assert features["m1_route_keep_mean"].tolist() == [1.0, 1.0]
    assert features["candidate_m1_dense_route_rate"].tolist() == [1.0, 1.0]


def test_dense_threshold_prefers_sparse_routes_without_false_sparse() -> None:
    probabilities = np.asarray((0.1, 0.2, 0.8, 0.9))
    truth = np.asarray((0, 0, 1, 1))

    threshold, prediction, metrics = choose_dense_threshold(
        probabilities,
        truth,
        false_sparse_limit=0.01,
    )

    assert 0.2 <= threshold < 0.8
    assert prediction.tolist() == [0, 0, 1, 1]
    assert metrics["false_sparse_count"] == 0
    assert metrics["sparse_route_rate"] == 0.5


def test_episode_cross_validation_holds_out_complete_groups() -> None:
    features = pd.DataFrame(
        {"risk": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]}
    )
    truth = np.asarray((0, 0, 0, 1, 1, 1))
    groups = np.asarray((0, 1, 2, 3, 4, 5))
    sample_weight = np.where(truth == 1, 10.0, 1.0)

    prediction, metrics = episode_cross_validated_route(
        LogisticRegression(),
        features,
        truth,
        groups,
        sample_weight,
        weight_parameter="sample_weight",
        false_sparse_limit=0.01,
    )

    assert prediction.shape == truth.shape
    assert metrics["fold_count"] == 6
    assert len(metrics["folds"]) == 6
