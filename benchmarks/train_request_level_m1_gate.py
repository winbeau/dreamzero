"""Train a request-level safety gate for dynamic Packed-M2 profiles.

The per-head M1 classifier controls local ``(timestep, layer, head)`` budgets,
but local attention fidelity does not guarantee final-action fidelity.  This
module aggregates deployment-time M1 outputs and historical VV dynamics into
one feature row per request, then learns a cost-sensitive three-way route:

``balanced -> conservative -> dense``.

Profile labels come exclusively from paired real DreamZero replays.  A request
is assigned the fastest profile that satisfies both final-action gates.  The
validation split chooses one-sided probability thresholds with a hard
false-sparse constraint; the test split is evaluated without retuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.train_dynamic_m1_classifier import (
    BASE_COLUMNS,
    PRIOR_KEYS,
    add_deployment_features,
    sequential_predict,
)


SEED = 20260830
PROFILE_NAMES = ("balanced", "conservative", "dense")
PROFILE_COSTS = np.asarray((0.70, 0.85, 1.00), dtype=np.float64)
STAGE_CODES = {"early": 0.0, "middle": 0.5, "late": 1.0}

# These are all observable immediately after the first two mandatory real DiT
# evaluations.  The gate reads the M1 state for DiT index 2, whose historical
# features come from DiT indices 0 and 1.  Current-call Dense attention
# statistics, later DiT state, and final actions are deliberately absent.
REQUEST_FEATURE_COLUMNS = (
    "trajectory_stage_code",
    "trajectory_fraction",
    "trajectory_length_log",
    "state_l2",
    "state_abs_mean",
    "m1_route_keep_mean",
    "m1_route_keep_p95",
    "m1_raw_keep_mean",
    "m1_dense_route_rate",
    "m1_fallback_rate",
    "m1_confidence_p05",
    "m1_confidence_mean",
    "m1_probability_entropy_mean",
    "m1_probability_entropy_p95",
    "m1_expected_keep_mean",
    "m1_critical_probability_mean",
    "history_turnover_mean",
    "history_turnover_p95",
    "history_vv_change_mean",
    "history_vv_change_p95",
    "history_two_vv_change_mean",
    "history_two_vv_change_p95",
    "history_vv_acceleration_p95",
    "history_attention_entropy_mean",
    "history_attention_mass_mean",
    "history_qa_qv_correlation_p05",
    "decision_late_layer_route_keep_mean",
    "decision_late_layer_dense_route_rate",
    "decision_late_layer_fallback_rate",
    "decision_late_layer_vv_change_p95",
)


def _measured_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in report["records"] if record["phase"] == "measured"]
    if not records:
        raise ValueError("report contains no measured records")
    return records


def paired_profile_quality(
    dense_report: dict[str, Any],
    profile_report: dict[str, Any],
    *,
    cosine_threshold: float = 0.999,
    relative_l2_threshold: float = 0.05,
) -> pd.DataFrame:
    """Return one aligned final-action quality row per paired request."""

    dense_records = _measured_records(dense_report)
    profile_records = _measured_records(profile_report)
    if dense_report.get("seed") != profile_report.get("seed"):
        raise ValueError("dense and profile reports use different request seeds")
    if len(dense_records) != len(profile_records):
        raise ValueError("dense and profile reports have different request counts")

    rows = []
    for dense, profile in zip(dense_records, profile_records, strict=True):
        if dense.get("request_key") != profile.get("request_key"):
            raise ValueError("dense and profile request keys do not align")
        dense_action = np.asarray(dense["action"], dtype=np.float64).reshape(-1)
        profile_action = np.asarray(profile["action"], dtype=np.float64).reshape(-1)
        if dense_action.shape != profile_action.shape:
            raise ValueError("dense and profile action shapes do not align")
        dense_norm = float(np.linalg.norm(dense_action))
        profile_norm = float(np.linalg.norm(profile_action))
        denominator = dense_norm * profile_norm
        if denominator <= 1e-12:
            cosine = 1.0 if dense_norm <= 1e-12 and profile_norm <= 1e-12 else 0.0
        else:
            cosine = float(np.dot(dense_action, profile_action) / denominator)
        relative_l2 = float(
            np.linalg.norm(profile_action - dense_action) / max(dense_norm, 1e-12)
        )
        rows.append(
            {
                "request_key": dense["request_key"],
                "split": dense.get("split"),
                "source_episode_index": dense.get("source_episode_index"),
                "trajectory_stage": dense.get("trajectory_stage"),
                "cosine": cosine,
                "relative_l2": relative_l2,
                "quality_pass": bool(
                    cosine >= cosine_threshold
                    and relative_l2 <= relative_l2_threshold
                ),
                "dense_latency_seconds": float(dense["latency_seconds"]),
                "profile_latency_seconds": float(profile["latency_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def build_profile_labels(
    dense_report: dict[str, Any],
    balanced_report: dict[str, Any],
    conservative_report: dict[str, Any],
    *,
    cosine_threshold: float = 0.999,
    relative_l2_threshold: float = 0.05,
) -> pd.DataFrame:
    """Label each request with its fastest quality-safe execution profile."""

    balanced = paired_profile_quality(
        dense_report,
        balanced_report,
        cosine_threshold=cosine_threshold,
        relative_l2_threshold=relative_l2_threshold,
    ).rename(
        columns={
            "cosine": "balanced_cosine",
            "relative_l2": "balanced_relative_l2",
            "quality_pass": "balanced_quality_pass",
            "profile_latency_seconds": "balanced_latency_seconds",
        }
    )
    conservative = paired_profile_quality(
        dense_report,
        conservative_report,
        cosine_threshold=cosine_threshold,
        relative_l2_threshold=relative_l2_threshold,
    ).rename(
        columns={
            "cosine": "conservative_cosine",
            "relative_l2": "conservative_relative_l2",
            "quality_pass": "conservative_quality_pass",
            "profile_latency_seconds": "conservative_latency_seconds",
        }
    )
    duplicate_columns = {
        "split",
        "source_episode_index",
        "trajectory_stage",
        "dense_latency_seconds",
    }
    conservative = conservative.drop(columns=list(duplicate_columns))
    labels = balanced.merge(
        conservative,
        on="request_key",
        how="inner",
        validate="one_to_one",
    )
    target = np.full(len(labels), 2, dtype=np.int64)
    target[labels["conservative_quality_pass"].to_numpy(dtype=bool)] = 1
    target[labels["balanced_quality_pass"].to_numpy(dtype=bool)] = 0
    labels["target_profile_index"] = target
    labels["target_profile"] = np.asarray(PROFILE_NAMES, dtype=object)[target]
    return labels


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    terms = probabilities * np.log(np.maximum(probabilities, 1e-12))
    return -terms.sum(axis=1) / np.log(probabilities.shape[1])


def _quantile(series: pd.Series, probability: float) -> float:
    return float(series.quantile(probability))


def _aggregate_region(frame: pd.DataFrame, mask: np.ndarray, prefix: str) -> pd.DataFrame:
    subset = frame.loc[mask]
    grouped = subset.groupby("request_key", sort=False)
    return grouped.agg(
        **{
            f"{prefix}_route_keep_mean": ("m1_route_keep", "mean"),
            f"{prefix}_dense_route_rate": ("m1_dense_route", "mean"),
            f"{prefix}_fallback_rate": ("m1_fallback", "mean"),
            f"{prefix}_vv_change_p95": (
                "previous_vv_output_change_relative_l2_max",
                lambda values: _quantile(values, 0.95),
            ),
        }
    )


def aggregate_request_features(
    frame: pd.DataFrame,
    result: dict[str, np.ndarray],
    budget_buckets: np.ndarray,
) -> pd.DataFrame:
    """Collapse per-head M1 state into deployment-safe request features."""

    frame = frame.copy()
    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    budget_buckets = np.asarray(budget_buckets, dtype=np.float64)
    frame["m1_route_keep"] = budget_buckets[result["prediction"]]
    frame["m1_raw_keep"] = budget_buckets[result["raw_prediction"]]
    frame["m1_dense_route"] = (result["prediction"] == len(budget_buckets) - 1).astype(
        np.float64
    )
    frame["m1_fallback"] = np.asarray(result["fallback"], dtype=np.float64)
    frame["m1_confidence"] = result["route_confidence"]
    frame["m1_probability_entropy"] = _normalized_entropy(probabilities)
    frame["m1_expected_keep"] = probabilities @ budget_buckets
    frame["m1_critical_probability"] = probabilities[
        :, budget_buckets >= 0.75
    ].sum(axis=1)
    frame["history_vv_acceleration"] = np.abs(
        frame["previous_vv_output_change_relative_l2_max"]
        - frame["previous_two_vv_output_change_relative_l2_max"]
    )

    decision = frame.loc[frame["dit_index"] == 2].copy()
    if decision.empty:
        raise ValueError("request gate requires the third real DiT state (dit_index=2)")
    request_count = frame["request_key"].nunique()
    if decision["request_key"].nunique() != request_count:
        raise ValueError("dit_index=2 does not cover every request")

    grouped = decision.groupby("request_key", sort=False)
    features = grouped.agg(
        split=("split", "first"),
        source_episode_index=("source_episode_index", "first"),
        trajectory_stage=("trajectory_stage", "first"),
        trajectory_fraction=("trajectory_fraction", "first"),
        trajectory_length=("trajectory_length", "first"),
        state_l2=("state_l2", "first"),
        state_abs_mean=("state_abs_mean", "first"),
        m1_route_keep_mean=("m1_route_keep", "mean"),
        m1_route_keep_p95=("m1_route_keep", lambda values: _quantile(values, 0.95)),
        m1_raw_keep_mean=("m1_raw_keep", "mean"),
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
        history_turnover_mean=("previous_support_turnover_max", "mean"),
        history_turnover_p95=(
            "previous_support_turnover_max",
            lambda values: _quantile(values, 0.95),
        ),
        history_vv_change_mean=(
            "previous_vv_output_change_relative_l2_max",
            "mean",
        ),
        history_vv_change_p95=(
            "previous_vv_output_change_relative_l2_max",
            lambda values: _quantile(values, 0.95),
        ),
        history_two_vv_change_mean=(
            "previous_two_vv_output_change_relative_l2_max",
            "mean",
        ),
        history_two_vv_change_p95=(
            "previous_two_vv_output_change_relative_l2_max",
            lambda values: _quantile(values, 0.95),
        ),
        history_vv_acceleration_p95=(
            "history_vv_acceleration",
            lambda values: _quantile(values, 0.95),
        ),
        history_attention_entropy_mean=("previous_normalized_entropy_mean", "mean"),
        history_attention_mass_mean=("previous_max_attention_mass_mean", "mean"),
        history_qa_qv_correlation_p05=(
            "previous_qa_qv_key_importance_correlation_mean",
            lambda values: _quantile(values, 0.05),
        ),
    )
    features["trajectory_stage_code"] = features["trajectory_stage"].map(STAGE_CODES)
    features["trajectory_length_log"] = np.log1p(features["trajectory_length"])

    late_layer_mask = decision["layer_index"].to_numpy() >= 27
    late_layer = _aggregate_region(
        decision,
        late_layer_mask,
        "decision_late_layer",
    )
    features = features.join(late_layer, how="left")
    features.reset_index(inplace=True)
    missing = sorted(set(REQUEST_FEATURE_COLUMNS) - set(features.columns))
    if missing:
        raise RuntimeError(f"request aggregation omitted features: {missing}")
    return features


def prepare_request_features(
    oracle_table: Path,
    m1_bundle: dict[str, Any],
    *,
    splits: Iterable[str],
) -> pd.DataFrame:
    """Run the portable per-head M1 bundle and aggregate requested splits."""

    splits = tuple(splits)
    columns = list(dict.fromkeys(BASE_COLUMNS))
    frame = pd.read_parquet(
        oracle_table,
        columns=columns,
        filters=[("split", "in", list(splits))],
    )
    if frame.empty:
        raise ValueError(f"Oracle table contains no rows for splits {splits}")
    prior_table = m1_bundle["prior_table"]
    frame = frame.merge(
        prior_table,
        on=list(PRIOR_KEYS),
        how="left",
        validate="many_to_one",
    )
    prior_columns = (
        "prior_budget_mean_tlh",
        "prior_budget_std_tlh",
        "prior_critical_rate_tlh",
    )
    if frame[list(prior_columns)].isna().any().any():
        raise ValueError("M1 prior table does not cover all Oracle rows")
    frame = add_deployment_features(frame)
    result = sequential_predict(
        m1_bundle["estimator"],
        frame,
        m1_bundle["confidence_calibrator"],
        m1_bundle["policy"],
    )
    return aggregate_request_features(
        frame,
        result,
        np.asarray(m1_bundle["budget_buckets"], dtype=np.float64),
    )


def candidate_estimators() -> dict[str, tuple[Any, str]]:
    """Return compact request-level candidates and their fit parameters."""

    return {
        "cost_sensitive_logistic": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
            "logisticregression__sample_weight",
        ),
        "cost_sensitive_gradient_boosting": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=160,
                    max_leaf_nodes=15,
                    min_samples_leaf=5,
                    l2_regularization=3.0,
                    random_state=SEED,
                ),
            ),
            "histgradientboostingclassifier__sample_weight",
        ),
    }


def aligned_profile_probabilities(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    aligned = np.zeros((len(features), len(PROFILE_NAMES)), dtype=np.float64)
    aligned[:, np.asarray(estimator.classes_, dtype=np.int64)] = probabilities
    return aligned


def route_from_risk_thresholds(
    probabilities: np.ndarray,
    *,
    balanced_risk_threshold: float,
    conservative_risk_threshold: float,
) -> np.ndarray:
    """Route using one-sided probabilities of requiring a safer profile."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    balanced_risk = probabilities[:, 1:].sum(axis=1)
    conservative_risk = probabilities[:, 2]
    prediction = np.full(len(probabilities), 2, dtype=np.int64)
    prediction[conservative_risk <= conservative_risk_threshold] = 1
    prediction[balanced_risk <= balanced_risk_threshold] = 0
    return prediction


def route_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    false_sparse = prediction < truth
    return {
        "request_count": int(len(truth)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=np.arange(len(PROFILE_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            truth,
            prediction,
            labels=np.arange(len(PROFILE_NAMES)),
        ).tolist(),
        "false_sparse_rate": float(np.mean(false_sparse)),
        "false_sparse_count": int(false_sparse.sum()),
        "balanced_route_rate": float(np.mean(prediction == 0)),
        "conservative_route_rate": float(np.mean(prediction == 1)),
        "dense_fallback_rate": float(np.mean(prediction == 2)),
        "mean_profile_cost": float(np.mean(PROFILE_COSTS[prediction])),
    }


def choose_risk_thresholds(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    false_sparse_limit: float,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Choose the cheapest validation route under a hard safety constraint."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    balanced_risk = probabilities[:, 1:].sum(axis=1)
    conservative_risk = probabilities[:, 2]

    def candidates(values: np.ndarray) -> np.ndarray:
        return np.unique(
            np.concatenate(
                (
                    [-1e-12],
                    values,
                    np.nextafter(values, np.inf),
                    [1.0],
                )
            )
        )

    feasible = []
    for balanced_threshold in candidates(balanced_risk):
        for conservative_threshold in candidates(conservative_risk):
            prediction = route_from_risk_thresholds(
                probabilities,
                balanced_risk_threshold=float(balanced_threshold),
                conservative_risk_threshold=float(conservative_threshold),
            )
            metrics = route_metrics(truth, prediction)
            if metrics["false_sparse_rate"] < false_sparse_limit:
                score = (
                    metrics["mean_profile_cost"],
                    metrics["dense_fallback_rate"],
                    -metrics["macro_f1"],
                )
                feasible.append(
                    (
                        score,
                        float(balanced_threshold),
                        float(conservative_threshold),
                        prediction,
                        metrics,
                    )
                )
    if not feasible:
        raise RuntimeError("Dense fallback unexpectedly failed request-level safety gate")
    _, balanced_threshold, conservative_threshold, prediction, metrics = min(
        feasible, key=lambda item: item[0]
    )
    thresholds = {
        "balanced_risk_threshold": balanced_threshold,
        "conservative_risk_threshold": conservative_threshold,
    }
    return thresholds, prediction, metrics


def evaluate_realized_route(
    labels: pd.DataFrame,
    prediction: np.ndarray,
    *,
    cosine_threshold: float,
    relative_l2_threshold: float,
) -> dict[str, Any]:
    """Evaluate the mixed route using actions/latencies from paired replays."""

    prediction = np.asarray(prediction, dtype=np.int64)
    if len(labels) != len(prediction):
        raise ValueError("labels and route prediction lengths differ")
    cosine = np.ones(len(labels), dtype=np.float64)
    relative_l2 = np.zeros(len(labels), dtype=np.float64)
    latency = labels["dense_latency_seconds"].to_numpy(dtype=np.float64).copy()
    balanced = prediction == 0
    conservative = prediction == 1
    cosine[balanced] = labels.loc[balanced, "balanced_cosine"]
    relative_l2[balanced] = labels.loc[balanced, "balanced_relative_l2"]
    latency[balanced] = labels.loc[balanced, "balanced_latency_seconds"]
    cosine[conservative] = labels.loc[conservative, "conservative_cosine"]
    relative_l2[conservative] = labels.loc[
        conservative, "conservative_relative_l2"
    ]
    latency[conservative] = labels.loc[
        conservative, "conservative_latency_seconds"
    ]
    dense_latency = labels["dense_latency_seconds"].to_numpy(dtype=np.float64)
    quality_pass = (cosine >= cosine_threshold) & (
        relative_l2 <= relative_l2_threshold
    )
    return {
        "action_cosine_mean": float(cosine.mean()),
        "action_cosine_min": float(cosine.min()),
        "action_relative_l2_mean": float(relative_l2.mean()),
        "action_relative_l2_max": float(relative_l2.max()),
        "quality_pass_rate": float(quality_pass.mean()),
        "quality_failure_count": int((~quality_pass).sum()),
        "mixed_e2e_speedup": float(dense_latency.sum() / latency.sum()),
        "strictly_faster_fraction": float(np.mean(latency < dense_latency)),
        "worst_request_key": str(
            labels.iloc[int(np.argmax(relative_l2))]["request_key"]
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _profile_paths(args: argparse.Namespace, split: str) -> tuple[Path, Path, Path]:
    return (
        getattr(args, f"dense_{split}"),
        getattr(args, f"balanced_{split}"),
        getattr(args, f"conservative_{split}"),
    )


def train_request_gate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = joblib.load(args.m1_bundle)
    features = prepare_request_features(
        args.oracle_table,
        bundle,
        splits=("train", "validation", "test"),
    )
    labels = {}
    for split in ("train", "validation", "test"):
        dense_path, balanced_path, conservative_path = _profile_paths(args, split)
        labels[split] = build_profile_labels(
            load_json(dense_path),
            load_json(balanced_path),
            load_json(conservative_path),
            cosine_threshold=args.cosine_threshold,
            relative_l2_threshold=args.relative_l2_threshold,
        )

    split_frames = {}
    for split in ("train", "validation", "test"):
        split_features = features.loc[features["split"] == split]
        split_frames[split] = labels[split].merge(
            split_features,
            on=[
                "request_key",
                "split",
                "source_episode_index",
                "trajectory_stage",
            ],
            how="inner",
            validate="one_to_one",
        )
        if len(split_frames[split]) != len(labels[split]):
            raise ValueError(f"missing request features for split {split}")

    train = split_frames["train"]
    validation = split_frames["validation"]
    test = split_frames["test"]
    train_truth = train["target_profile_index"].to_numpy(dtype=np.int64)
    validation_truth = validation["target_profile_index"].to_numpy(dtype=np.int64)
    test_truth = test["target_profile_index"].to_numpy(dtype=np.int64)
    sample_weight = 1.0 + args.underprediction_cost * (train_truth / 2.0) ** 2

    model_results = {}
    fitted = {}
    for name, (estimator, sample_weight_parameter) in candidate_estimators().items():
        fit_kwargs = {sample_weight_parameter: sample_weight}
        estimator.fit(train[list(REQUEST_FEATURE_COLUMNS)], train_truth, **fit_kwargs)
        validation_probabilities = aligned_profile_probabilities(
            estimator, validation[list(REQUEST_FEATURE_COLUMNS)]
        )
        thresholds, validation_prediction, validation_metrics = choose_risk_thresholds(
            validation_probabilities,
            validation_truth,
            false_sparse_limit=args.false_sparse_limit,
        )
        test_probabilities = aligned_profile_probabilities(
            estimator, test[list(REQUEST_FEATURE_COLUMNS)]
        )
        test_prediction = route_from_risk_thresholds(test_probabilities, **thresholds)
        test_metrics = route_metrics(test_truth, test_prediction)
        model_results[name] = {
            "thresholds": thresholds,
            "validation": validation_metrics,
            "validation_realized": evaluate_realized_route(
                validation,
                validation_prediction,
                cosine_threshold=args.cosine_threshold,
                relative_l2_threshold=args.relative_l2_threshold,
            ),
            "test": test_metrics,
            "test_realized": evaluate_realized_route(
                test,
                test_prediction,
                cosine_threshold=args.cosine_threshold,
                relative_l2_threshold=args.relative_l2_threshold,
            ),
        }
        fitted[name] = (estimator, thresholds, test_prediction)

    feasible = [
        name
        for name, result in model_results.items()
        if result["validation"]["false_sparse_rate"] < args.false_sparse_limit
    ]
    selected = min(
        feasible,
        key=lambda name: (
            model_results[name]["validation"]["mean_profile_cost"],
            model_results[name]["validation"]["dense_fallback_rate"],
            -model_results[name]["validation"]["macro_f1"],
        ),
    )
    estimator, thresholds, test_prediction = fitted[selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "model_name": selected,
            "estimator": estimator,
            "thresholds": thresholds,
            "feature_columns": REQUEST_FEATURE_COLUMNS,
            "profile_names": PROFILE_NAMES,
            "quality_thresholds": {
                "cosine": args.cosine_threshold,
                "relative_l2": args.relative_l2_threshold,
            },
            "source_m1_bundle": str(args.m1_bundle),
        },
        args.output_dir / "request_level_m1_gate.joblib",
        compress=3,
    )
    features.to_parquet(args.output_dir / "request_features.parquet", index=False)
    for split, frame in split_frames.items():
        frame.to_parquet(args.output_dir / f"{split}_labels_and_features.parquet", index=False)
    summary = {
        "selected_model": selected,
        "feature_columns": list(REQUEST_FEATURE_COLUMNS),
        "profile_names": list(PROFILE_NAMES),
        "split_requests": {
            split: int(len(frame)) for split, frame in split_frames.items()
        },
        "split_episodes": {
            split: int(frame["source_episode_index"].nunique())
            for split, frame in split_frames.items()
        },
        "quality_thresholds": {
            "cosine": args.cosine_threshold,
            "relative_l2": args.relative_l2_threshold,
        },
        "false_sparse_limit": args.false_sparse_limit,
        "models": model_results,
        "passed": bool(
            model_results[selected]["test"]["false_sparse_rate"]
            < args.false_sparse_limit
            and model_results[selected]["test_realized"]["quality_failure_count"] == 0
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-table", type=Path, required=True)
    parser.add_argument("--m1-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--dense-{split}", type=Path, required=True)
        parser.add_argument(f"--balanced-{split}", type=Path, required=True)
        parser.add_argument(f"--conservative-{split}", type=Path, required=True)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    parser.add_argument("--relative-l2-threshold", type=float, default=0.05)
    parser.add_argument("--false-sparse-limit", type=float, default=0.01)
    parser.add_argument("--underprediction-cost", type=float, default=40.0)
    args = parser.parse_args()
    print(json.dumps(train_request_gate(args), indent=2))


if __name__ == "__main__":
    main()
