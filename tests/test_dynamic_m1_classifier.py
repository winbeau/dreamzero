import numpy as np
import pandas as pd
import pytest

from benchmarks.train_dynamic_m1_classifier import (
    RoutePolicy,
    add_train_only_priors,
    budget_indices,
    route_metrics,
    sequential_predict,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    MappedGMMClassifier,
)


class AlwaysAggressiveEstimator:
    classes_ = np.arange(7)

    def predict_proba(self, features):
        probabilities = np.zeros((len(features), 7), dtype=np.float64)
        probabilities[:, 0] = 0.9
        probabilities[:, 1:] = 0.1 / 6.0
        return probabilities


class ZeroConfidenceCalibrator:
    def predict(self, confidence):
        return np.zeros_like(confidence)


def test_budget_indices_require_fixed_buckets():
    assert budget_indices([0.1, 0.35, 1.0]).tolist() == [0, 3, 6]
    with pytest.raises(ValueError, match="outside fixed budget"):
        budget_indices([0.3])


def test_train_priors_leave_source_episode_out():
    train = pd.DataFrame(
        {
            "source_episode_index": [10, 11],
            "dit_index": [0, 0],
            "layer_index": [0, 0],
            "head_index": [0, 0],
            "oracle_min_keep_ratio": [0.2, 0.8],
        }
    )
    validation = pd.DataFrame(
        {
            "source_episode_index": [12],
            "dit_index": [0],
            "layer_index": [0],
            "head_index": [0],
            "oracle_min_keep_ratio": [0.5],
        }
    )

    augmented_train, [augmented_validation] = add_train_only_priors(
        train, [validation]
    )

    assert augmented_train["prior_budget_mean_tlh"].tolist() == pytest.approx(
        [0.8, 0.2]
    )
    assert augmented_validation["prior_budget_mean_tlh"].item() == pytest.approx(0.5)


def test_mapped_gmm_exposes_seven_budget_probabilities():
    rng = np.random.default_rng(7)
    features = np.concatenate(
        (rng.normal(-2.0, 0.1, (30, 2)), rng.normal(2.0, 0.1, (30, 2)))
    )
    labels = np.concatenate(
        (np.zeros(30, dtype=np.int64), np.full(30, 6, dtype=np.int64))
    )
    classifier = MappedGMMClassifier(n_components=2, random_state=7).fit(
        features, labels
    )

    probabilities = classifier.predict_proba(features[:4])

    assert probabilities.shape == (4, len(BUDGET_BUCKETS))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_low_confidence_route_falls_back_dense():
    row_count = 8
    frame = pd.DataFrame(
        {
            "request_key": ["request"] * row_count,
            "source_episode_index": [0] * row_count,
            "dit_index": np.arange(row_count),
            "layer_index": [0] * row_count,
            "head_index": [0] * row_count,
        }
    )
    # sequential_predict only indexes these names; the dummy estimator ignores values.
    from benchmarks.train_dynamic_m1_classifier import FEATURE_COLUMNS

    for column in FEATURE_COLUMNS:
        frame[column] = 0.0
    for ratio in BUDGET_BUCKETS:
        suffix = f"r{int(round(ratio * 100)):03d}"
        frame[f"worst_mass_p05_{suffix}"] = ratio
        frame[f"worst_output_cosine_p05_{suffix}"] = 1.0
        frame[f"worst_output_relative_l2_p95_{suffix}"] = 0.0

    result = sequential_predict(
        AlwaysAggressiveEstimator(),
        frame,
        ZeroConfidenceCalibrator(),
        RoutePolicy(confidence_threshold=0.5, promotion_buckets=0),
    )
    metrics = route_metrics(frame, np.full(row_count, 6), result)

    assert result["fallback"].all()
    assert result["prediction"].tolist() == [6] * row_count
    assert metrics["false_sparse_rate"] == 0.0
    assert metrics["mass_p05_at_least_0_9_rate"] == 1.0
