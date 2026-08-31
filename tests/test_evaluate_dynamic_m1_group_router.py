import numpy as np
import pandas as pd
import pytest

from benchmarks.evaluate_dynamic_m1_group_router import (
    apply_grouped_route_fallback,
    evaluate_grouped_split,
)
from benchmarks.train_dynamic_m1_classifier import FEATURE_COLUMNS
from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    RoutePolicy,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
)


class AggressiveEstimator:
    classes_ = np.arange(len(BUDGET_BUCKETS))

    def predict_proba(self, features):
        probabilities = np.full(
            (len(features), len(BUDGET_BUCKETS)),
            0.05 / (len(BUDGET_BUCKETS) - 1),
        )
        probabilities[:, 0] = 0.95
        return probabilities


class IdentityCalibrator:
    def predict(self, confidence):
        return np.asarray(confidence)


def _quality_columns(frame: pd.DataFrame, truth: np.ndarray) -> None:
    for ratio in BUDGET_BUCKETS:
        suffix = f"r{round(ratio * 100):03d}"
        safe = ratio >= truth
        frame[f"worst_mass_p05_{suffix}"] = np.where(safe, 0.95, 0.50)
        frame[f"worst_output_cosine_p05_{suffix}"] = np.where(safe, 1.0, 0.90)
        frame[f"worst_output_relative_l2_p95_{suffix}"] = np.where(
            safe,
            0.0,
            0.20,
        )


def test_apply_grouped_route_adds_risk_fallback_before_metrics() -> None:
    frame = pd.DataFrame(
        {
            "request_key": ["request"] * 4,
            "split": ["validation"] * 4,
            "source_episode_index": [0] * 4,
            "dit_index": [0] * 4,
            "layer_index": [0] * 4,
            "head_index": [0, 1, 2, 3],
            "oracle_min_keep_ratio": [0.20, 1.00, 0.50, 1.00],
        }
    )
    result = {
        "prediction": np.asarray([0, 3, 4, 0]),
        "route_confidence": np.asarray([0.95, 0.95, 0.95, 0.95]),
        "fallback": np.asarray([False, False, False, False]),
    }
    scanned = np.asarray([[[True, True, True, False]]])
    safe = np.asarray([[[True, False, True, False]]])
    risk_table = DownstreamHeadRiskTable(scanned, safe, {})

    grouped, routes, diagnostics = apply_grouped_route_fallback(
        frame,
        result,
        risk_table,
    )

    assert grouped["prediction"].tolist() == [2, 6, 4, 6]
    assert grouped["fallback"].tolist() == [False, True, False, True]
    assert routes["grouped_keep_ratio"].tolist() == [0.25, 1.0, 0.5, 1.0]
    assert diagnostics["downstream_unknown_fallback_rate"] == pytest.approx(0.25)
    assert diagnostics["downstream_unsafe_fallback_rate"] == pytest.approx(0.25)
    assert diagnostics["maximum_groups_per_request_timestep_layer"] == 3
    assert diagnostics["unknown_or_unsafe_sparse_count"] == 0


def test_evaluate_grouped_split_bootstraps_post_quantization_route() -> None:
    frame = pd.DataFrame(
        {
            "request_key": ["request"] * 8,
            "split": ["validation"] * 8,
            "source_episode_index": [0] * 8,
            "dit_index": np.arange(8),
            "layer_index": [0] * 8,
            "head_index": [0] * 8,
            "oracle_min_keep_ratio": [0.20] * 8,
        }
    )
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    _quality_columns(frame, frame["oracle_min_keep_ratio"].to_numpy())
    risk_table = DownstreamHeadRiskTable(
        np.ones((8, 1, 1), dtype=bool),
        np.ones((8, 1, 1), dtype=bool),
        {},
    )
    bundle = {
        "estimator": AggressiveEstimator(),
        "confidence_calibrator": IdentityCalibrator(),
        "policy": RoutePolicy(confidence_threshold=0.8, promotion_buckets=0),
    }

    metrics, routes = evaluate_grouped_split(
        frame,
        bundle,
        risk_table,
        bootstrap_repeats=5,
    )

    assert metrics["false_sparse_rate"] == 0.0
    assert metrics["mass_p05_at_least_0_9_rate"] == 1.0
    assert metrics["mean_grouped_keep_ratio"] == pytest.approx(0.25)
    assert metrics["maximum_groups_per_request_timestep_layer"] == 1
    assert metrics["bootstrap"]["repeats"] == 5
    assert (
        metrics["bootstrap"]["metrics"]["false_sparse_rate"]["ci95_high"]
        == 0.0
    )
    assert routes["grouped_keep_ratio"].tolist() == [0.25] * 8
