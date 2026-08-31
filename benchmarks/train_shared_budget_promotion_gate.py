"""Train a causal Sparse-versus-Dense gate for a shared Packed budget table.

The base table remains the fastest measured quality candidate.  This gate sees
only state and Packed-M1 signals available after the first two mandatory real
DiT evaluations and predicts whether the remaining DiTs may use that table or
must fall back to Dense.  Final-action replay supplies the binary supervision;
current-call Dense attention, future DiTs, actions, and offline trajectory
annotations are never model inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.train_dynamic_m1_classifier import (
    BASE_COLUMNS,
    PACKED_PROXY_INPUT_COLUMNS,
    PRIOR_KEYS,
    add_deployment_features,
    sequential_predict,
)
from benchmarks.train_request_level_m1_gate import paired_profile_quality


SEED = 20260831
ROUTE_NAMES = ("shared_sparse", "dense_fallback")
ROUTE_COSTS = np.asarray((0.0, 1.0), dtype=np.float64)
SHARED_GATE_FEATURE_COLUMNS = (
    "state_l2",
    "state_abs_mean",
    "m1_route_keep_mean",
    "m1_route_keep_p95",
    "m1_dense_route_rate",
    "m1_fallback_rate",
    "m1_confidence_p05",
    "m1_confidence_mean",
    "m1_probability_entropy_mean",
    "m1_probability_entropy_p95",
    "m1_expected_keep_mean",
    "m1_critical_probability_mean",
    "packed_turnover_mean",
    "packed_turnover_p95",
    "packed_route_entropy_mean",
    "packed_route_mass_mean",
    "packed_route_mass_p05",
    "packed_action_change_mean",
    "packed_action_change_p95",
    "packed_action_cosine_p05",
    "packed_cfg_disagreement_mean",
    "packed_cfg_disagreement_p95",
    "packed_signature_norm_mean",
    "packed_signature_norm_p95",
    "candidate_m1_route_keep_mean",
    "candidate_m1_dense_route_rate",
    "candidate_m1_fallback_rate",
    "candidate_packed_turnover_p95",
    "candidate_packed_action_change_p95",
)


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    terms = probabilities * np.log(np.maximum(probabilities, 1e-12))
    return -terms.sum(axis=1) / np.log(probabilities.shape[1])


def _quantile(values: pd.Series, probability: float) -> float:
    return float(values.quantile(probability))


def _aggregate_candidate_region(decision: pd.DataFrame) -> pd.DataFrame:
    candidate = decision.loc[decision["layer_index"].between(1, 20)]
    if candidate["request_key"].nunique() != decision["request_key"].nunique():
        raise ValueError("candidate layer region does not cover every request")
    return candidate.groupby("request_key", sort=False).agg(
        candidate_m1_route_keep_mean=("m1_route_keep", "mean"),
        candidate_m1_dense_route_rate=("m1_dense_route", "mean"),
        candidate_m1_fallback_rate=("m1_fallback", "mean"),
        candidate_packed_turnover_p95=(
            "previous_packed_route_support_turnover_max",
            lambda values: _quantile(values, 0.95),
        ),
        candidate_packed_action_change_p95=(
            "previous_packed_action_output_change_relative_l2_max",
            lambda values: _quantile(values, 0.95),
        ),
    )


def aggregate_shared_gate_features(
    frame: pd.DataFrame,
    result: dict[str, np.ndarray],
    budget_buckets: np.ndarray,
) -> pd.DataFrame:
    """Build one causal feature row per request at the DiT-2 decision point."""

    frame = frame.copy()
    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    budget_buckets = np.asarray(budget_buckets, dtype=np.float64)
    if probabilities.shape != (len(frame), len(budget_buckets)):
        raise ValueError("M1 probability matrix does not align with feature rows")
    frame["m1_route_keep"] = budget_buckets[result["prediction"]]
    frame["m1_dense_route"] = (
        np.asarray(result["prediction"]) == len(budget_buckets) - 1
    ).astype(np.float64)
    frame["m1_fallback"] = np.asarray(result["fallback"], dtype=np.float64)
    frame["m1_confidence"] = np.asarray(
        result["route_confidence"], dtype=np.float64
    )
    frame["m1_probability_entropy"] = _normalized_entropy(probabilities)
    frame["m1_expected_keep"] = probabilities @ budget_buckets
    frame["m1_critical_probability"] = probabilities[
        :, budget_buckets >= 0.75
    ].sum(axis=1)
    frame["packed_action_acceleration"] = np.abs(
        frame["previous_packed_action_output_change_relative_l2_max"]
        - frame["previous_two_packed_action_output_change_relative_l2_max"]
    )

    decision = frame.loc[frame["dit_index"] == 2].copy()
    if decision.empty:
        raise ValueError("shared gate requires state before the third real DiT")
    if decision["request_key"].nunique() != frame["request_key"].nunique():
        raise ValueError("DiT-2 state does not cover every request")

    grouped = decision.groupby("request_key", sort=False)
    features = grouped.agg(
        split=("split", "first"),
        source_episode_index=("source_episode_index", "first"),
        state_l2=("state_l2", "first"),
        state_abs_mean=("state_abs_mean", "first"),
        m1_route_keep_mean=("m1_route_keep", "mean"),
        m1_route_keep_p95=("m1_route_keep", lambda values: _quantile(values, 0.95)),
        m1_dense_route_rate=("m1_dense_route", "mean"),
        m1_fallback_rate=("m1_fallback", "mean"),
        m1_confidence_p05=("m1_confidence", lambda values: _quantile(values, 0.05)),
        m1_confidence_mean=("m1_confidence", "mean"),
        m1_probability_entropy_mean=("m1_probability_entropy", "mean"),
        m1_probability_entropy_p95=(
            "m1_probability_entropy",
            lambda values: _quantile(values, 0.95),
        ),
        m1_expected_keep_mean=("m1_expected_keep", "mean"),
        m1_critical_probability_mean=("m1_critical_probability", "mean"),
        packed_turnover_mean=(
            "previous_packed_route_support_turnover_max",
            "mean",
        ),
        packed_turnover_p95=(
            "previous_packed_route_support_turnover_max",
            lambda values: _quantile(values, 0.95),
        ),
        packed_route_entropy_mean=(
            "previous_packed_route_normalized_entropy_mean",
            "mean",
        ),
        packed_route_mass_mean=("previous_packed_route_max_mass_mean", "mean"),
        packed_route_mass_p05=(
            "previous_packed_route_max_mass_mean",
            lambda values: _quantile(values, 0.05),
        ),
        packed_action_change_mean=(
            "previous_packed_action_output_change_relative_l2_max",
            "mean",
        ),
        packed_action_change_p95=(
            "previous_packed_action_output_change_relative_l2_max",
            lambda values: _quantile(values, 0.95),
        ),
        packed_two_action_change_mean=(
            "previous_two_packed_action_output_change_relative_l2_max",
            "mean",
        ),
        packed_two_action_change_p95=(
            "previous_two_packed_action_output_change_relative_l2_max",
            lambda values: _quantile(values, 0.95),
        ),
        packed_action_cosine_p05=(
            "previous_packed_action_output_change_cosine_min",
            lambda values: _quantile(values, 0.05),
        ),
        packed_cfg_disagreement_mean=(
            "previous_packed_cfg_disagreement_relative_l2",
            "mean",
        ),
        packed_cfg_disagreement_p95=(
            "previous_packed_cfg_disagreement_relative_l2",
            lambda values: _quantile(values, 0.95),
        ),
        packed_signature_norm_mean=(
            "previous_packed_action_output_signature_norm",
            "mean",
        ),
        packed_signature_norm_p95=(
            "previous_packed_action_output_signature_norm",
            lambda values: _quantile(values, 0.95),
        ),
        packed_action_acceleration_p95=(
            "packed_action_acceleration",
            lambda values: _quantile(values, 0.95),
        ),
    )
    features = features.join(_aggregate_candidate_region(decision), how="left")
    features.reset_index(inplace=True)
    missing = sorted(set(SHARED_GATE_FEATURE_COLUMNS) - set(features.columns))
    if missing:
        raise RuntimeError(f"shared gate aggregation omitted features: {missing}")
    return features


def prepare_shared_gate_features(
    oracle_table: Path,
    m1_bundle: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = tuple(m1_bundle["feature_columns"])
    if "previous_packed_route_support_turnover_max" not in feature_columns:
        raise ValueError("shared promotion gate requires a Packed-proxy M1 bundle")
    columns = list(dict.fromkeys((*BASE_COLUMNS, *PACKED_PROXY_INPUT_COLUMNS)))
    frame = pd.read_parquet(oracle_table, columns=columns)
    prior_table = m1_bundle["prior_table"]
    frame = frame.merge(
        prior_table,
        on=list(PRIOR_KEYS),
        how="left",
        validate="many_to_one",
    )
    if frame[
        [
            "prior_budget_mean_tlh",
            "prior_budget_std_tlh",
            "prior_critical_rate_tlh",
        ]
    ].isna().any().any():
        raise ValueError("M1 prior table does not cover shared-gate rows")
    frame = add_deployment_features(frame)
    result = sequential_predict(
        m1_bundle["estimator"],
        frame,
        m1_bundle.get("confidence_calibrator"),
        m1_bundle["policy"],
        feature_columns=feature_columns,
    )
    return aggregate_shared_gate_features(
        frame,
        result,
        np.asarray(m1_bundle["budget_buckets"], dtype=np.float64),
    )


def candidate_estimators() -> dict[str, tuple[Any, str]]:
    return {
        "cost_sensitive_logistic": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(C=0.25, max_iter=2000, random_state=SEED),
            ),
            "logisticregression__sample_weight",
        ),
        "cost_sensitive_gradient_boosting": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=160,
                    max_leaf_nodes=7,
                    min_samples_leaf=5,
                    l2_regularization=4.0,
                    random_state=SEED,
                ),
            ),
            "histgradientboostingclassifier__sample_weight",
        ),
        "cost_sensitive_small_mlp": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(24,),
                    alpha=0.02,
                    learning_rate_init=0.003,
                    max_iter=800,
                    random_state=SEED,
                ),
            ),
            "mlpclassifier__sample_weight",
        ),
    }


def dense_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if probabilities.shape != (len(features), len(classes)):
        raise ValueError("shared gate estimator returned invalid probabilities")
    dense_columns = np.flatnonzero(classes == 1)
    if len(dense_columns) != 1:
        raise ValueError("shared gate estimator did not learn both route classes")
    return probabilities[:, dense_columns[0]]


def episode_cross_validated_route(
    estimator: Any,
    features: pd.DataFrame,
    truth: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    *,
    weight_parameter: str,
    false_sparse_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose each fold threshold without observing its held-out episode."""

    truth = np.asarray(truth, dtype=np.int64)
    groups = np.asarray(groups)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    prediction = np.full(len(features), -1, dtype=np.int64)
    fold_metrics = []
    splitter = LeaveOneGroupOut()
    for train_indices, held_out_indices in splitter.split(features, truth, groups):
        fold_estimator = clone(estimator)
        fold_estimator.fit(
            features.iloc[train_indices],
            truth[train_indices],
            **{weight_parameter: sample_weight[train_indices]},
        )
        train_probability = dense_probability(
            fold_estimator, features.iloc[train_indices]
        )
        threshold, _train_prediction, _train_metrics = choose_dense_threshold(
            train_probability,
            truth[train_indices],
            false_sparse_limit=false_sparse_limit,
        )
        held_out_probability = dense_probability(
            fold_estimator, features.iloc[held_out_indices]
        )
        held_out_prediction = (held_out_probability > threshold).astype(np.int64)
        prediction[held_out_indices] = held_out_prediction
        fold_metrics.append(
            {
                "source_episode_index": int(groups[held_out_indices[0]]),
                **route_metrics(truth[held_out_indices], held_out_prediction),
            }
        )
    if np.any(prediction < 0):
        raise RuntimeError("episode cross-validation omitted request rows")
    return prediction, {
        **route_metrics(truth, prediction),
        "fold_count": len(fold_metrics),
        "folds_with_false_sparse": int(
            sum(metric["false_sparse_count"] > 0 for metric in fold_metrics)
        ),
        "folds": fold_metrics,
    }


def route_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    false_sparse = (truth == 1) & (prediction == 0)
    return {
        "request_count": int(len(truth)),
        "unsafe_request_count": int(np.sum(truth == 1)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=(0, 1)).tolist(),
        "false_sparse_rate": float(np.mean(false_sparse)),
        "false_sparse_count": int(np.sum(false_sparse)),
        "sparse_route_rate": float(np.mean(prediction == 0)),
        "dense_fallback_rate": float(np.mean(prediction == 1)),
        "mean_route_cost": float(np.mean(ROUTE_COSTS[prediction])),
    }


def choose_dense_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    false_sparse_limit: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(
        np.concatenate(
            (
                [-1e-12],
                probabilities,
                np.nextafter(probabilities, np.inf),
                [1.0],
            )
        )
    )
    feasible = []
    for threshold in candidates:
        prediction = (probabilities > threshold).astype(np.int64)
        metrics = route_metrics(truth, prediction)
        if metrics["false_sparse_rate"] < false_sparse_limit:
            feasible.append(
                (
                    (
                        metrics["mean_route_cost"],
                        -metrics["macro_f1"],
                    ),
                    float(threshold),
                    prediction,
                    metrics,
                )
            )
    if not feasible:
        raise RuntimeError("Dense fallback unexpectedly failed the safety constraint")
    _score, threshold, prediction, metrics = min(feasible, key=lambda item: item[0])
    return threshold, prediction, metrics


def realized_route_metrics(
    labels: pd.DataFrame,
    prediction: np.ndarray,
    *,
    cosine_threshold: float,
    relative_l2_threshold: float,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    sparse = prediction == 0
    cosine = np.ones(len(labels), dtype=np.float64)
    relative_l2 = np.zeros(len(labels), dtype=np.float64)
    latency = labels["dense_latency_seconds"].to_numpy(dtype=np.float64).copy()
    cosine[sparse] = labels.loc[sparse, "cosine"]
    relative_l2[sparse] = labels.loc[sparse, "relative_l2"]
    latency[sparse] = labels.loc[sparse, "profile_latency_seconds"]
    dense_latency = labels["dense_latency_seconds"].to_numpy(dtype=np.float64)
    quality_pass = (cosine >= cosine_threshold) & (
        relative_l2 <= relative_l2_threshold
    )
    return {
        "action_cosine_mean": float(cosine.mean()),
        "action_cosine_min": float(cosine.min()),
        "action_relative_l2_mean": float(relative_l2.mean()),
        "action_relative_l2_max": float(relative_l2.max()),
        "quality_failure_count": int(np.sum(~quality_pass)),
        "mixed_e2e_speedup": float(dense_latency.sum() / latency.sum()),
        "strictly_faster_fraction": float(np.mean(latency < dense_latency)),
        "worst_request_key": str(
            labels.iloc[int(np.argmax(relative_l2))]["request_key"]
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _quality_labels(
    dense_path: Path,
    sparse_path: Path,
    *,
    cosine_threshold: float,
    relative_l2_threshold: float,
) -> pd.DataFrame:
    labels = paired_profile_quality(
        load_json(dense_path),
        load_json(sparse_path),
        cosine_threshold=cosine_threshold,
        relative_l2_threshold=relative_l2_threshold,
    )
    labels["target_route_index"] = (~labels["quality_pass"]).astype(np.int64)
    labels["target_route"] = np.asarray(ROUTE_NAMES, dtype=object)[
        labels["target_route_index"]
    ]
    return labels


def train_shared_gate(args: argparse.Namespace) -> dict[str, Any]:
    m1_bundle = joblib.load(args.m1_bundle)
    features = prepare_shared_gate_features(args.oracle_table, m1_bundle)
    split_frames = {}
    for split in ("train", "validation", "test"):
        labels = _quality_labels(
            getattr(args, f"dense_{split}"),
            getattr(args, f"sparse_{split}"),
            cosine_threshold=args.cosine_threshold,
            relative_l2_threshold=args.relative_l2_threshold,
        )
        split_features = features.loc[features["split"] == split]
        split_frames[split] = labels.merge(
            split_features,
            on=["request_key", "split", "source_episode_index"],
            how="inner",
            validate="one_to_one",
        )
        if len(split_frames[split]) != len(labels):
            raise ValueError(f"missing shared-gate features for split {split}")

    train = split_frames["train"]
    validation = split_frames["validation"]
    test = split_frames["test"]
    train_truth = train["target_route_index"].to_numpy(dtype=np.int64)
    validation_truth = validation["target_route_index"].to_numpy(dtype=np.int64)
    test_truth = test["target_route_index"].to_numpy(dtype=np.int64)
    sample_weight = np.where(train_truth == 1, args.underprediction_cost, 1.0)

    model_results = {}
    fitted = {}
    for name, (estimator, weight_parameter) in candidate_estimators().items():
        _cv_prediction, cv_metrics = episode_cross_validated_route(
            estimator,
            train[list(SHARED_GATE_FEATURE_COLUMNS)],
            train_truth,
            train["source_episode_index"].to_numpy(dtype=np.int64),
            sample_weight,
            weight_parameter=weight_parameter,
            false_sparse_limit=args.false_sparse_limit,
        )
        estimator.fit(
            train[list(SHARED_GATE_FEATURE_COLUMNS)],
            train_truth,
            **{weight_parameter: sample_weight},
        )
        validation_probability = dense_probability(
            estimator, validation[list(SHARED_GATE_FEATURE_COLUMNS)]
        )
        threshold, validation_prediction, validation_metrics = (
            choose_dense_threshold(
                validation_probability,
                validation_truth,
                false_sparse_limit=args.false_sparse_limit,
            )
        )
        test_probability = dense_probability(
            estimator, test[list(SHARED_GATE_FEATURE_COLUMNS)]
        )
        test_prediction = (test_probability > threshold).astype(np.int64)
        model_results[name] = {
            "train_episode_cross_validation": cv_metrics,
            "threshold": threshold,
            "validation": validation_metrics,
            "validation_realized": realized_route_metrics(
                validation,
                validation_prediction,
                cosine_threshold=args.cosine_threshold,
                relative_l2_threshold=args.relative_l2_threshold,
            ),
            "test": route_metrics(test_truth, test_prediction),
            "test_realized": realized_route_metrics(
                test,
                test_prediction,
                cosine_threshold=args.cosine_threshold,
                relative_l2_threshold=args.relative_l2_threshold,
            ),
        }
        fitted[name] = (estimator, threshold)

    selected = min(
        model_results,
        key=lambda name: (
            model_results[name]["validation"]["false_sparse_count"],
            model_results[name]["train_episode_cross_validation"][
                "false_sparse_count"
            ],
            model_results[name]["validation"]["mean_route_cost"],
            model_results[name]["train_episode_cross_validation"][
                "mean_route_cost"
            ],
            -model_results[name]["validation"]["macro_f1"],
        ),
    )
    estimator, threshold = fitted[selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "model_name": selected,
            "estimator": estimator,
            "dense_probability_threshold": threshold,
            "feature_columns": SHARED_GATE_FEATURE_COLUMNS,
            "route_names": ROUTE_NAMES,
            "decision_dit_index": 2,
            "source_m1_bundle": str(args.m1_bundle),
            "base_budget_table": str(args.base_budget_table),
            "quality_thresholds": {
                "cosine": args.cosine_threshold,
                "relative_l2": args.relative_l2_threshold,
            },
        },
        args.output_dir / "shared_budget_promotion_gate.joblib",
        compress=3,
    )
    features.to_parquet(args.output_dir / "request_features.parquet", index=False)
    for split, frame in split_frames.items():
        frame.to_parquet(
            args.output_dir / f"{split}_labels_and_features.parquet", index=False
        )
    selected_result = model_results[selected]
    safety_gates_passed = bool(
        selected_result["train_episode_cross_validation"]["false_sparse_rate"]
        < args.false_sparse_limit
        and selected_result["train_episode_cross_validation"][
            "folds_with_false_sparse"
        ]
        == 0
        and selected_result["validation"]["false_sparse_rate"]
        < args.false_sparse_limit
        and selected_result["test"]["false_sparse_rate"]
        < args.false_sparse_limit
        and selected_result["validation_realized"]["quality_failure_count"] == 0
        and selected_result["test_realized"]["quality_failure_count"] == 0
    )
    performance_gates_passed = bool(
        selected_result["validation_realized"]["mixed_e2e_speedup"]
        >= args.minimum_mixed_speedup
        and selected_result["test_realized"]["mixed_e2e_speedup"]
        >= args.minimum_mixed_speedup
        and selected_result["validation_realized"]["strictly_faster_fraction"]
        >= args.minimum_strictly_faster_fraction
        and selected_result["test_realized"]["strictly_faster_fraction"]
        >= args.minimum_strictly_faster_fraction
    )
    passed = safety_gates_passed and performance_gates_passed
    summary = {
        "selected_model": selected,
        "feature_columns": list(SHARED_GATE_FEATURE_COLUMNS),
        "route_names": list(ROUTE_NAMES),
        "decision_dit_index": 2,
        "split_requests": {
            split: len(frame) for split, frame in split_frames.items()
        },
        "split_unsafe_requests": {
            split: int(frame["target_route_index"].sum())
            for split, frame in split_frames.items()
        },
        "underprediction_cost": args.underprediction_cost,
        "false_sparse_limit": args.false_sparse_limit,
        "minimum_mixed_speedup": args.minimum_mixed_speedup,
        "minimum_strictly_faster_fraction": (
            args.minimum_strictly_faster_fraction
        ),
        "models": model_results,
        "safety_gates_passed": safety_gates_passed,
        "performance_gates_passed": performance_gates_passed,
        "passed": passed,
        "reason": (
            "all safety and performance gates passed"
            if passed
            else (
                "shared promotion gate failed episode-level safety"
                if not safety_gates_passed
                else "shared promotion gate is safe but misses performance targets"
            )
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-table", type=Path, required=True)
    parser.add_argument("--m1-bundle", type=Path, required=True)
    parser.add_argument("--base-budget-table", type=Path, required=True)
    parser.add_argument("--dense-train", type=Path, required=True)
    parser.add_argument("--sparse-train", type=Path, required=True)
    parser.add_argument("--dense-validation", type=Path, required=True)
    parser.add_argument("--sparse-validation", type=Path, required=True)
    parser.add_argument("--dense-test", type=Path, required=True)
    parser.add_argument("--sparse-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    parser.add_argument("--relative-l2-threshold", type=float, default=0.05)
    parser.add_argument("--false-sparse-limit", type=float, default=0.01)
    parser.add_argument("--underprediction-cost", type=float, default=50.0)
    parser.add_argument("--minimum-mixed-speedup", type=float, default=1.35)
    parser.add_argument(
        "--minimum-strictly-faster-fraction",
        type=float,
        default=0.95,
    )
    args = parser.parse_args()
    if args.underprediction_cost <= 1.0:
        parser.error("--underprediction-cost must exceed one")
    if args.minimum_mixed_speedup <= 1.0:
        parser.error("--minimum-mixed-speedup must exceed one")
    if not 0.0 < args.minimum_strictly_faster_fraction <= 1.0:
        parser.error("--minimum-strictly-faster-fraction must lie in (0, 1]")
    summary = train_shared_gate(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
