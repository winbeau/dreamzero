import numpy as np
import pytest

from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    RoutePolicy,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
    DynamicM1GroupedRouter,
    build_grouped_budget_table,
    grouped_route_metrics,
    quantize_grouped_budgets,
)


class FeatureDrivenEstimator:
    classes_ = np.arange(len(BUDGET_BUCKETS))

    def predict_proba(self, features):
        classes = features["class_index"].to_numpy(dtype=np.int64)
        confidence = features["confidence"].to_numpy(dtype=np.float64)
        probabilities = np.repeat(
            ((1.0 - confidence) / (len(BUDGET_BUCKETS) - 1))[:, None],
            len(BUDGET_BUCKETS),
            axis=1,
        )
        probabilities[np.arange(len(features)), classes] = confidence
        return probabilities


class IdentityCalibrator:
    def predict(self, confidence):
        return np.asarray(confidence)


def _bundle():
    return {
        "estimator": FeatureDrivenEstimator(),
        "confidence_calibrator": IdentityCalibrator(),
        "feature_columns": (
            "class_index",
            "confidence",
            "previous_support_turnover_max",
            "previous_vv_output_change_relative_l2_max",
            "history_two_available",
        ),
        "budget_buckets": BUDGET_BUCKETS,
        "policy": RoutePolicy(confidence_threshold=0.8, promotion_buckets=0),
    }


def _features():
    return {
        "class_index": np.asarray([[0, 3, 4, 0], [3, 0, 0, 0]]),
        "confidence": np.asarray([[0.95, 0.95, 0.50, 0.95], [0.95] * 4]),
        "previous_support_turnover_max": np.full((2, 4), 0.10),
        "previous_vv_output_change_relative_l2_max": np.full((2, 4), 0.02),
        "history_two_available": np.ones((2, 4)),
    }


def test_grouped_budget_quantization_is_upward_and_fixed_shape() -> None:
    values = np.asarray([0.10, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00])

    assert quantize_grouped_budgets(values).tolist() == [
        0.25,
        0.25,
        0.25,
        0.50,
        0.50,
        0.75,
        1.00,
    ]


def test_grouped_router_separates_confidence_unknown_and_unsafe_fallback() -> None:
    router = DynamicM1GroupedRouter(_bundle())
    scanned = np.ones((2, 4), dtype=bool)
    scanned[0, 3] = False
    safe = np.ones((2, 4), dtype=bool)
    safe[1, 0] = False

    decision = router.route_step(
        _features(),
        dit_index=0,
        downstream_scanned=scanned,
        downstream_safe=safe,
    )

    assert decision.keep_ratios.tolist() == [
        [0.25, 0.50, 1.00, 1.00],
        [1.00, 0.25, 0.25, 0.25],
    ]
    assert decision.classifier_fallback.tolist() == [
        [False, False, True, False],
        [False, False, False, False],
    ]
    assert decision.downstream_unknown_fallback.tolist() == [
        [False, False, False, True],
        [False, False, False, False],
    ]
    assert decision.downstream_unsafe_fallback.tolist() == [
        [False, False, False, False],
        [True, False, False, False],
    ]
    assert decision.categories.tolist() == [
        ["stable", "slow-changing", "uncertain", "uncertain"],
        ["uncertain", "stable", "stable", "stable"],
    ]
    groups = decision.execution_groups_for_layer(0)
    assert [group["history_keep_ratio"] for group in groups] == [1.0, 0.5, 0.25]
    assert groups[0]["head_indices"] == (2, 3)
    assert groups[0]["fallback_heads"] == 2
    assert decision.summary()["maximum_groups_per_layer"] == 3


def test_grouped_router_defaults_unscanned_heads_to_dense() -> None:
    decision = DynamicM1GroupedRouter(_bundle()).route_step(
        _features(),
        dit_index=0,
    )

    assert np.all(decision.keep_ratios == 1.0)
    assert np.all(decision.downstream_unknown_fallback)
    assert np.all(decision.categories == "uncertain")


def test_grouped_router_consumes_task_disjoint_risk_table() -> None:
    scanned = np.zeros((8, 2, 4), dtype=bool)
    safe = np.zeros_like(scanned)
    scanned[0, 0, :2] = True
    safe[0, 0, :2] = True
    risk_table = DownstreamHeadRiskTable(scanned, safe, {"split": "validation"})

    decision = DynamicM1GroupedRouter(_bundle()).route_step(
        _features(),
        dit_index=0,
        downstream_risk_table=risk_table,
    )

    assert decision.keep_ratios[0].tolist() == [0.25, 0.50, 1.0, 1.0]
    assert np.all(decision.keep_ratios[1] == 1.0)
    with pytest.raises(ValueError, match="either downstream_risk_table"):
        DynamicM1GroupedRouter(_bundle()).route_step(
            _features(),
            dit_index=0,
            downstream_risk_table=risk_table,
            downstream_scanned=np.ones((2, 4), dtype=bool),
            downstream_safe=np.ones((2, 4), dtype=bool),
        )


def test_grouped_router_predictable_late_uses_prequantized_m1_budget() -> None:
    router = DynamicM1GroupedRouter(_bundle())
    decision = router.route_step(
        _features(),
        dit_index=5,
        downstream_scanned=np.ones((2, 4), dtype=bool),
        downstream_safe=np.ones((2, 4), dtype=bool),
    )

    assert decision.raw_keep_ratios[0, 1] == pytest.approx(0.35)
    assert decision.keep_ratios[0, 1] == pytest.approx(0.50)
    assert decision.categories[0, 1] == "predictable-late"
    assert decision.allow_linear_extrapolation[0, 1]
    assert decision.refresh_frequency[0, 1] == 4


def test_grouped_budget_table_couples_q_and_k_to_at_most_four_shapes() -> None:
    router = DynamicM1GroupedRouter(_bundle())
    decisions = [
        router.route_step(
            _features(),
            dit_index=dit_index,
            downstream_scanned=np.ones((2, 4), dtype=bool),
            downstream_safe=np.ones((2, 4), dtype=bool),
        )
        for dit_index in range(8)
    ]

    table = build_grouped_budget_table(decisions)

    assert table.num_dit_steps == 8
    assert table.num_layers == 2
    assert table.num_heads == 4
    assert table.num_groups <= 4
    assert table.head_current_keep_ratios == table.head_keep_ratios


def test_grouped_route_metrics_measure_post_quantization_false_sparse() -> None:
    router = DynamicM1GroupedRouter(_bundle())
    decision = router.route_step(
        _features(),
        dit_index=0,
        downstream_scanned=np.ones((2, 4), dtype=bool),
        downstream_safe=np.ones((2, 4), dtype=bool),
    )
    truth = np.asarray([[0.35, 0.50, 1.0, 0.20], [0.35, 0.35, 0.20, 0.20]])

    metrics = grouped_route_metrics(decision, truth)

    assert metrics["false_sparse_count"] == 2
    assert metrics["false_sparse_rate"] == pytest.approx(0.25)
    assert metrics["unknown_or_unsafe_sparse_count"] == 0
