"""Risk-controlled M1 routing into fixed-shape Packed-M2 head groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    RoutePolicy,
)
from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedHeadGroupBudgetTable,
)

EXECUTOR_BUDGET_BUCKETS = np.asarray(
    (0.25, 0.50, 0.75, 1.00),
    dtype=np.float64,
)


def quantize_grouped_budgets(values: np.ndarray) -> np.ndarray:
    """Conservatively map M1's seven buckets to four executor shapes."""

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("M1 keep ratios must be finite")
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("M1 keep ratios must lie in (0, 1]")
    indices = np.searchsorted(
        EXECUTOR_BUDGET_BUCKETS,
        values - 1e-12,
        side="left",
    )
    return EXECUTOR_BUDGET_BUCKETS[
        np.clip(indices, 0, len(EXECUTOR_BUDGET_BUCKETS) - 1)
    ]


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DownstreamHeadRiskTable:
    """Task-disjoint downstream coverage and conservative safety by head."""

    scanned: np.ndarray
    safe: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        scanned = np.asarray(self.scanned, dtype=bool)
        safe = np.asarray(self.safe, dtype=bool)
        if scanned.ndim != 3 or not all(scanned.shape):
            raise ValueError("Downstream risk table must have [DiT, layer, head] shape")
        if safe.shape != scanned.shape:
            raise ValueError("Downstream scanned and safe cubes must align")
        if np.any(safe & ~scanned):
            raise ValueError("An unscanned downstream head cannot be marked safe")
        object.__setattr__(self, "scanned", _readonly(scanned))
        object.__setattr__(self, "safe", _readonly(safe))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_dit_steps(self) -> int:
        return int(self.scanned.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.scanned.shape[1])

    @property
    def num_heads(self) -> int:
        return int(self.scanned.shape[2])

    def masks(self, dit_index: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= dit_index < self.num_dit_steps:
            raise IndexError("dit_index is outside the downstream risk table")
        return self.scanned[dit_index], self.safe[dit_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "shape": list(self.scanned.shape),
            "metadata": dict(self.metadata),
            "scanned": self.scanned.tolist(),
            "safe": self.safe.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DownstreamHeadRiskTable:
        return cls(
            scanned=np.asarray(payload["scanned"], dtype=bool),
            safe=np.asarray(payload["safe"], dtype=bool),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DownstreamHeadRiskTable:
        import json

        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class GroupedM1StepDecision:
    """One request's fixed-group decision for a real DiT evaluation."""

    dit_index: int
    raw_keep_ratios: np.ndarray
    keep_ratios: np.ndarray
    route_confidence: np.ndarray
    classifier_fallback: np.ndarray
    downstream_unknown_fallback: np.ndarray
    downstream_unsafe_fallback: np.ndarray
    fallback: np.ndarray
    categories: np.ndarray
    refresh_frequency: np.ndarray
    allow_linear_extrapolation: np.ndarray

    def __post_init__(self) -> None:
        if self.dit_index < 0:
            raise ValueError("dit_index must be non-negative")
        shape = np.asarray(self.keep_ratios).shape
        if len(shape) != 2 or not all(shape):
            raise ValueError("Grouped M1 decisions must have [layer, head] shape")
        fields = (
            "raw_keep_ratios",
            "route_confidence",
            "classifier_fallback",
            "downstream_unknown_fallback",
            "downstream_unsafe_fallback",
            "fallback",
            "categories",
            "refresh_frequency",
            "allow_linear_extrapolation",
        )
        for field_name in fields:
            if np.asarray(getattr(self, field_name)).shape != shape:
                raise ValueError(f"{field_name} does not match keep-ratio shape")
        unique = set(np.unique(self.keep_ratios).tolist())
        if not unique.issubset(set(EXECUTOR_BUDGET_BUCKETS.tolist())):
            raise ValueError("Grouped M1 decision uses a non-executor budget")
        for field_name in ("raw_keep_ratios", "keep_ratios", "route_confidence"):
            object.__setattr__(
                self,
                field_name,
                _readonly(np.asarray(getattr(self, field_name), dtype=np.float64)),
            )
        for field_name in (
            "classifier_fallback",
            "downstream_unknown_fallback",
            "downstream_unsafe_fallback",
            "fallback",
            "allow_linear_extrapolation",
        ):
            object.__setattr__(
                self,
                field_name,
                _readonly(np.asarray(getattr(self, field_name), dtype=bool)),
            )
        object.__setattr__(
            self,
            "categories",
            _readonly(np.asarray(self.categories, dtype=object)),
        )
        object.__setattr__(
            self,
            "refresh_frequency",
            _readonly(np.asarray(self.refresh_frequency, dtype=np.int64)),
        )

    @property
    def num_layers(self) -> int:
        return int(self.keep_ratios.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.keep_ratios.shape[1])

    def execution_groups_for_layer(
        self,
        layer_index: int,
    ) -> tuple[dict[str, object], ...]:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError("layer_index is outside the Grouped M1 decision")
        groups = []
        for ratio in sorted(np.unique(self.keep_ratios[layer_index]), reverse=True):
            heads = np.flatnonzero(self.keep_ratios[layer_index] == ratio)
            groups.append(
                {
                    "head_indices": tuple(int(index) for index in heads),
                    "history_keep_ratio": float(ratio),
                    "current_keep_ratio": float(ratio),
                    "minimum_route_confidence": float(
                        np.min(self.route_confidence[layer_index, heads])
                    ),
                    "fallback_heads": int(
                        np.sum(self.fallback[layer_index, heads])
                    ),
                }
            )
        if len(groups) > len(EXECUTOR_BUDGET_BUCKETS):
            raise RuntimeError("Grouped M1 produced too many execution groups")
        return tuple(groups)

    def summary(self) -> dict[str, object]:
        category, category_count = np.unique(self.categories, return_counts=True)
        group_counts = np.asarray(
            [
                len(self.execution_groups_for_layer(layer_index))
                for layer_index in range(self.num_layers)
            ],
            dtype=np.int64,
        )
        return {
            "dit_index": self.dit_index,
            "shape": [self.num_layers, self.num_heads],
            "mean_keep_ratio": float(np.mean(self.keep_ratios)),
            "dense_head_fraction": float(np.mean(self.keep_ratios == 1.0)),
            "classifier_fallback_rate": float(
                np.mean(self.classifier_fallback)
            ),
            "downstream_unknown_fallback_rate": float(
                np.mean(self.downstream_unknown_fallback)
            ),
            "downstream_unsafe_fallback_rate": float(
                np.mean(self.downstream_unsafe_fallback)
            ),
            "combined_fallback_rate": float(np.mean(self.fallback)),
            "maximum_groups_per_layer": int(group_counts.max()),
            "mean_groups_per_layer": float(group_counts.mean()),
            "category_counts": {
                str(name): int(count)
                for name, count in zip(category, category_count, strict=True)
            },
        }


class DynamicM1GroupedRouter:
    """Apply a calibrated M1 bundle and enforce downstream-risk fallback."""

    def __init__(
        self,
        bundle: Mapping[str, Any],
        *,
        require_downstream_coverage: bool = True,
    ) -> None:
        required = {
            "estimator",
            "feature_columns",
            "budget_buckets",
            "policy",
        }
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"M1 bundle is missing fields: {sorted(missing)}")
        bundle_buckets = np.asarray(bundle["budget_buckets"], dtype=np.float64)
        if not np.array_equal(bundle_buckets, BUDGET_BUCKETS):
            raise ValueError("M1 bundle budget buckets do not match runtime")
        feature_columns = tuple(str(name) for name in bundle["feature_columns"])
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("M1 bundle feature columns must be non-empty and unique")
        policy = bundle["policy"]
        if isinstance(policy, Mapping):
            policy = RoutePolicy(
                confidence_threshold=float(policy["confidence_threshold"]),
                promotion_buckets=int(policy["promotion_buckets"]),
            )
        if not isinstance(policy, RoutePolicy):
            raise TypeError("M1 bundle policy must be a RoutePolicy")
        if not 0.0 <= policy.confidence_threshold <= 1.0:
            raise ValueError("M1 confidence threshold must lie in [0, 1]")
        if policy.promotion_buckets < 0:
            raise ValueError("M1 promotion buckets must be non-negative")

        self.estimator = bundle["estimator"]
        self.calibrator = bundle.get("confidence_calibrator")
        self.feature_columns = feature_columns
        self.policy = policy
        self.require_downstream_coverage = bool(require_downstream_coverage)

    @classmethod
    def from_joblib(
        cls,
        path: str | Path,
        *,
        require_downstream_coverage: bool = True,
    ) -> DynamicM1GroupedRouter:
        import joblib

        return cls(
            joblib.load(path),
            require_downstream_coverage=require_downstream_coverage,
        )

    def _feature_frame(
        self,
        features: Mapping[str, np.ndarray] | np.ndarray,
    ) -> tuple[pd.DataFrame, tuple[int, int], dict[str, np.ndarray]]:
        feature_arrays: dict[str, np.ndarray] = {}
        if isinstance(features, Mapping):
            missing = set(self.feature_columns) - set(features)
            if missing:
                raise ValueError(f"M1 runtime features are missing: {sorted(missing)}")
            shape = None
            for name in self.feature_columns:
                value = np.asarray(features[name], dtype=np.float64)
                if value.ndim != 2:
                    raise ValueError(f"M1 feature {name} must have [layer, head] shape")
                if shape is None:
                    shape = value.shape
                elif value.shape != shape:
                    raise ValueError("M1 runtime feature shapes do not align")
                feature_arrays[name] = value
            assert shape is not None
            matrix = np.column_stack(
                [feature_arrays[name].reshape(-1) for name in self.feature_columns]
            )
        else:
            array = np.asarray(features, dtype=np.float64)
            if array.ndim != 3 or array.shape[-1] != len(self.feature_columns):
                raise ValueError(
                    "M1 runtime feature tensor must have [layer, head, feature] shape"
                )
            shape = (int(array.shape[0]), int(array.shape[1]))
            matrix = array.reshape(-1, array.shape[-1])
            feature_arrays = {
                name: array[..., index]
                for index, name in enumerate(self.feature_columns)
            }
        return (
            pd.DataFrame(matrix, columns=self.feature_columns),
            (int(shape[0]), int(shape[1])),
            feature_arrays,
        )

    @staticmethod
    def _downstream_masks(
        shape: tuple[int, int],
        *,
        downstream_scanned: np.ndarray | None,
        downstream_safe: np.ndarray | None,
        require_coverage: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if downstream_scanned is None and downstream_safe is None:
            if require_coverage:
                return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
            return np.ones(shape, dtype=bool), np.ones(shape, dtype=bool)
        if downstream_scanned is None or downstream_safe is None:
            raise ValueError(
                "downstream_scanned and downstream_safe must be supplied together"
            )
        scanned = np.asarray(downstream_scanned, dtype=bool)
        safe = np.asarray(downstream_safe, dtype=bool)
        if scanned.shape != shape or safe.shape != shape:
            raise ValueError("downstream risk masks do not match M1 route shape")
        return scanned, safe

    def route_step(
        self,
        features: Mapping[str, np.ndarray] | np.ndarray,
        *,
        dit_index: int,
        downstream_risk_table: DownstreamHeadRiskTable | None = None,
        downstream_scanned: np.ndarray | None = None,
        downstream_safe: np.ndarray | None = None,
    ) -> GroupedM1StepDecision:
        feature_frame, shape, feature_arrays = self._feature_frame(features)
        probabilities = np.asarray(
            self.estimator.predict_proba(feature_frame),
            dtype=np.float64,
        )
        classes = np.asarray(self.estimator.classes_, dtype=np.int64)
        if probabilities.shape != (len(feature_frame), len(classes)):
            raise ValueError("M1 estimator returned invalid probability shape")
        if len(np.unique(classes)) != len(classes):
            raise ValueError("M1 estimator classes must be unique")
        if np.any(classes < 0) or np.any(classes >= len(BUDGET_BUCKETS)):
            raise ValueError("M1 estimator returned an invalid class index")
        aligned = np.zeros((len(feature_frame), len(BUDGET_BUCKETS)), dtype=np.float64)
        aligned[:, classes] = probabilities
        if np.any(aligned < 0.0) or np.any(aligned > 1.0):
            raise ValueError("M1 estimator probabilities must lie in [0, 1]")
        if not np.all(np.isfinite(aligned)) or not np.allclose(
            aligned.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("M1 estimator probabilities are invalid")

        raw_prediction = np.argmax(aligned, axis=1)
        raw_confidence = np.max(aligned, axis=1)
        route_confidence = (
            np.asarray(self.calibrator.predict(raw_confidence), dtype=np.float64)
            if self.calibrator is not None
            else raw_confidence
        )
        if route_confidence.shape != raw_confidence.shape or not np.all(
            np.isfinite(route_confidence)
        ):
            raise ValueError("M1 confidence calibrator returned invalid values")
        if np.any(route_confidence < 0.0) or np.any(route_confidence > 1.0):
            raise ValueError("M1 route confidence must lie in [0, 1]")

        promoted = np.minimum(
            len(BUDGET_BUCKETS) - 1,
            raw_prediction + self.policy.promotion_buckets,
        )
        raw_keep = BUDGET_BUCKETS[promoted].reshape(shape)
        confidence = route_confidence.reshape(shape)
        classifier_fallback = confidence < self.policy.confidence_threshold
        if downstream_risk_table is not None:
            if downstream_scanned is not None or downstream_safe is not None:
                raise ValueError(
                    "Pass either downstream_risk_table or explicit masks, not both"
                )
            downstream_scanned, downstream_safe = downstream_risk_table.masks(
                dit_index
            )
        scanned, safe = self._downstream_masks(
            shape,
            downstream_scanned=downstream_scanned,
            downstream_safe=downstream_safe,
            require_coverage=self.require_downstream_coverage,
        )
        downstream_unknown = ~scanned
        downstream_unsafe = scanned & ~safe
        fallback = classifier_fallback | downstream_unknown | downstream_unsafe

        effective = raw_keep.copy()
        effective[fallback] = 1.0
        grouped = quantize_grouped_budgets(effective)

        turnover = np.asarray(
            feature_arrays.get(
                "previous_support_turnover_max",
                np.full(shape, np.inf),
            )
        )
        vv_change = np.asarray(
            feature_arrays.get(
                "previous_vv_output_change_relative_l2_max",
                np.full(shape, np.inf),
            )
        )
        history_two = np.asarray(
            feature_arrays.get("history_two_available", np.zeros(shape))
        )
        categories = np.full(shape, "slow-changing", dtype=object)
        categories[grouped >= 0.75] = "critical"
        stable = (grouped <= 0.25) & (turnover <= 0.20)
        categories[stable] = "stable"
        predictable = (
            (dit_index >= 5)
            & (raw_keep <= 0.35)
            & (turnover <= 0.20)
            & (vv_change <= 0.05)
            & (history_two > 0.5)
        )
        categories[predictable] = "predictable-late"
        categories[fallback] = "uncertain"
        refresh = np.full(shape, 2, dtype=np.int64)
        refresh[categories == "critical"] = 1
        refresh[categories == "stable"] = 5
        refresh[categories == "predictable-late"] = 4
        refresh[categories == "uncertain"] = 1

        return GroupedM1StepDecision(
            dit_index=dit_index,
            raw_keep_ratios=raw_keep,
            keep_ratios=grouped,
            route_confidence=confidence,
            classifier_fallback=classifier_fallback,
            downstream_unknown_fallback=downstream_unknown,
            downstream_unsafe_fallback=downstream_unsafe,
            fallback=fallback,
            categories=categories,
            refresh_frequency=refresh,
            allow_linear_extrapolation=(categories == "predictable-late"),
        )


def build_grouped_budget_table(
    decisions: Sequence[GroupedM1StepDecision],
    *,
    num_dit_steps: int = 8,
    name: str = "dynamic_m1_grouped",
) -> DynamicPackedHeadGroupBudgetTable:
    """Build a Q/K-coupled four-shape executor table from routed steps."""

    if len(decisions) != num_dit_steps:
        raise ValueError(
            f"Expected {num_dit_steps} routed DiT decisions, got {len(decisions)}"
        )
    by_dit = {decision.dit_index: decision for decision in decisions}
    if set(by_dit) != set(range(num_dit_steps)) or len(by_dit) != len(decisions):
        raise ValueError("Grouped M1 decisions must cover each DiT index exactly once")
    shapes = {decision.keep_ratios.shape for decision in decisions}
    if len(shapes) != 1:
        raise ValueError("Grouped M1 decisions do not share layer/head geometry")
    cube = tuple(
        tuple(
            tuple(float(value) for value in layer)
            for layer in by_dit[dit_index].keep_ratios
        )
        for dit_index in range(num_dit_steps)
    )
    return DynamicPackedHeadGroupBudgetTable(
        head_keep_ratios=cube,
        head_current_keep_ratios=cube,
        name=name,
    )


def grouped_route_metrics(
    decision: GroupedM1StepDecision,
    oracle_min_keep_ratio: np.ndarray,
) -> dict[str, object]:
    """Evaluate post-quantization risk without substituting local for downstream."""

    truth = np.asarray(oracle_min_keep_ratio, dtype=np.float64)
    if truth.shape != decision.keep_ratios.shape:
        raise ValueError("Oracle truth shape does not match Grouped M1 decision")
    if not np.all(np.isfinite(truth)):
        raise ValueError("Oracle truth must be finite")
    false_sparse = decision.keep_ratios + 1e-12 < truth
    critical = truth >= 0.75
    return {
        **decision.summary(),
        "false_sparse_rate": float(np.mean(false_sparse)),
        "false_sparse_count": int(np.sum(false_sparse)),
        "critical_false_sparse_rate": float(
            np.mean(false_sparse[critical]) if np.any(critical) else 0.0
        ),
        "unknown_or_unsafe_sparse_count": int(
            np.sum(
                (decision.downstream_unknown_fallback
                 | decision.downstream_unsafe_fallback)
                & (decision.keep_ratios < 1.0)
            )
        ),
    }
