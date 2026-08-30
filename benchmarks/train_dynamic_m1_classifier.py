"""Train and calibrate dynamic M1 budget classifiers on task-disjoint Oracle data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from groot.vla.model.dreamzero.modules.dynamic_m1_classifier import (
    BUDGET_BUCKETS,
    MappedGMMClassifier,
)


SEED = 20260830
PRIOR_KEYS = ("dit_index", "layer_index", "head_index")
QUALITY_PREFIXES = (
    "worst_mass_p05",
    "worst_output_cosine_p05",
    "worst_output_relative_l2_p95",
)
HISTORY_COLUMNS = (
    "previous_support_turnover_max",
    "previous_vv_output_change_relative_l2_max",
    "previous_two_vv_output_change_relative_l2_max",
    "previous_normalized_entropy_mean",
    "previous_max_attention_mass_mean",
    "previous_qa_qv_key_importance_correlation_mean",
)
BASE_COLUMNS = (
    "request_key",
    "split",
    "source_episode_index",
    "trajectory_stage",
    "trajectory_fraction",
    "trajectory_length",
    "length_bucket",
    "instruction_index",
    "state_l2",
    "state_abs_mean",
    "action_l2",
    "action_std",
    "action_temporal_delta_l2",
    "dit_index",
    "scheduler_index",
    "diffusion_timestep",
    "layer_index",
    "head_index",
    "timestep_position",
    "layer_depth",
    "oracle_min_keep_ratio",
    "previous_oracle_min_keep_ratio",
    "previous_two_oracle_min_keep_ratio",
    *HISTORY_COLUMNS,
)
FEATURE_COLUMNS = (
    "timestep_position",
    "layer_depth",
    "head_position",
    "scheduler_position",
    "diffusion_timestep_scaled",
    "trajectory_stage_code",
    "trajectory_fraction",
    "trajectory_length_log",
    "length_bucket_code",
    "instruction_position",
    "state_l2",
    "state_abs_mean",
    "action_l2",
    "action_std",
    "action_temporal_delta_l2",
    "history_one_available",
    "history_two_available",
    *HISTORY_COLUMNS,
    "vv_change_acceleration",
    "prior_budget_mean_tlh",
    "prior_budget_std_tlh",
    "prior_critical_rate_tlh",
)


@dataclass(frozen=True)
class RoutePolicy:
    confidence_threshold: float
    promotion_buckets: int


def _ratio_suffix(ratio: float) -> str:
    return f"r{int(round(ratio * 100)):03d}"


def required_columns() -> list[str]:
    columns = list(BASE_COLUMNS)
    for prefix in QUALITY_PREFIXES:
        columns.extend(f"{prefix}_{_ratio_suffix(ratio)}" for ratio in BUDGET_BUCKETS)
    return list(dict.fromkeys(columns))


def budget_indices(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    distances = np.abs(values[:, None] - BUDGET_BUCKETS[None, :])
    indices = np.argmin(distances, axis=1)
    if np.any(distances[np.arange(len(values)), indices] > 1e-5):
        bad = np.unique(values[distances[np.arange(len(values)), indices] > 1e-5])
        raise ValueError(f"Values outside fixed budget buckets: {bad[:10]}")
    return indices.astype(np.int64)


def add_train_only_priors(
    train: pd.DataFrame, evaluation_frames: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    """Add cross-task Oracle priors without leaking a train row into its own prior."""

    train = train.copy()
    train["_target"] = train["oracle_min_keep_ratio"].astype(np.float64)
    train["_target_square"] = train["_target"] ** 2
    train["_critical"] = (train["_target"] >= 0.75).astype(np.float64)
    aggregations = {
        "prior_count": ("_target", "size"),
        "prior_sum": ("_target", "sum"),
        "prior_square_sum": ("_target_square", "sum"),
        "prior_critical_sum": ("_critical", "sum"),
    }
    global_stats = train.groupby(list(PRIOR_KEYS), sort=False).agg(**aggregations).reset_index()
    episode_stats = (
        train.groupby(["source_episode_index", *PRIOR_KEYS], sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    prior_table = global_stats[list(PRIOR_KEYS)].copy()
    prior_count = global_stats["prior_count"].clip(lower=1.0)
    prior_mean = global_stats["prior_sum"] / prior_count
    prior_variance = global_stats["prior_square_sum"] / prior_count - prior_mean**2
    prior_table["prior_budget_mean_tlh"] = prior_mean
    prior_table["prior_budget_std_tlh"] = np.sqrt(prior_variance.clip(lower=0.0))
    prior_table["prior_critical_rate_tlh"] = (
        global_stats["prior_critical_sum"] / prior_count
    )

    def finalize(frame: pd.DataFrame, leave_episode_out: bool) -> pd.DataFrame:
        frame = frame.copy()
        frame["_original_order"] = np.arange(len(frame), dtype=np.int64)
        frame = frame.merge(global_stats, on=list(PRIOR_KEYS), how="left", sort=False)
        if leave_episode_out:
            frame = frame.merge(
                episode_stats,
                on=["source_episode_index", *PRIOR_KEYS],
                how="left",
                sort=False,
                suffixes=("", "_episode"),
            )
            for name in (
                "prior_count",
                "prior_sum",
                "prior_square_sum",
                "prior_critical_sum",
            ):
                frame[name] = frame[name] - frame[f"{name}_episode"].fillna(0.0)
                frame.drop(columns=f"{name}_episode", inplace=True)
        count = frame["prior_count"].clip(lower=1.0)
        mean = frame["prior_sum"] / count
        variance = frame["prior_square_sum"] / count - mean**2
        frame["prior_budget_mean_tlh"] = mean
        frame["prior_budget_std_tlh"] = np.sqrt(variance.clip(lower=0.0))
        frame["prior_critical_rate_tlh"] = frame["prior_critical_sum"] / count
        frame.sort_values("_original_order", inplace=True)
        frame.drop(
            columns=[
                "_original_order",
                "prior_count",
                "prior_sum",
                "prior_square_sum",
                "prior_critical_sum",
                "_target",
                "_target_square",
                "_critical",
            ],
            errors="ignore",
            inplace=True,
        )
        frame.reset_index(drop=True, inplace=True)
        return frame

    return (
        finalize(train, True),
        [finalize(frame, False) for frame in evaluation_frames],
        prior_table,
    )


def add_deployment_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["head_position"] = frame["head_index"] / 39.0
    frame["scheduler_position"] = frame["scheduler_index"] / 15.0
    frame["diffusion_timestep_scaled"] = frame["diffusion_timestep"] / 1000.0
    frame["trajectory_stage_code"] = frame["trajectory_stage"].map(
        {"early": 0.0, "middle": 0.5, "late": 1.0}
    )
    frame["length_bucket_code"] = frame["length_bucket"].map(
        {"short": 0.0, "middle": 0.5, "long": 1.0}
    )
    frame["trajectory_length_log"] = np.log1p(frame["trajectory_length"])
    frame["instruction_position"] = frame["instruction_index"] / 2.0
    frame["history_one_available"] = (frame["dit_index"] >= 1).astype(np.float64)
    frame["history_two_available"] = (frame["dit_index"] >= 2).astype(np.float64)
    frame["vv_change_acceleration"] = np.abs(
        frame["previous_vv_output_change_relative_l2_max"]
        - frame["previous_two_vv_output_change_relative_l2_max"]
    )
    # Route-state columns are populated recursively during deployment.  They
    # stay outside FEATURE_COLUMNS until out-of-fold route histories exist;
    # fitting on Oracle previous budgets would create teacher-forcing leakage.
    frame["previous_route_budget"] = 1.0
    frame["previous_two_route_budget"] = 1.0
    return frame


def stratified_cap(frame: pd.DataFrame, labels: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(frame) <= maximum:
        return np.arange(len(frame), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        count = max(1, int(round(maximum * len(candidates) / len(labels))))
        selected.append(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
    result = np.concatenate(selected)
    if len(result) > maximum:
        result = rng.choice(result, size=maximum, replace=False)
    return np.sort(result)


def candidate_estimators() -> dict[str, tuple[object, bool]]:
    imputer = lambda: SimpleImputer(strategy="median", add_indicator=True)
    return {
        "original_gmm": (
            make_pipeline(
                imputer(),
                StandardScaler(),
                MappedGMMClassifier(n_components=3, random_state=SEED),
            ),
            False,
        ),
        "supervised_logistic": (
            make_pipeline(
                imputer(),
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=SEED,
                ),
            ),
            False,
        ),
        "gradient_boosting": (
            make_pipeline(
                imputer(),
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=160,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
            False,
        ),
        "small_mlp": (
            make_pipeline(
                imputer(),
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    alpha=1e-3,
                    batch_size=2048,
                    early_stopping=True,
                    max_iter=120,
                    random_state=SEED,
                ),
            ),
            False,
        ),
        "cost_sensitive_gradient_boosting": (
            make_pipeline(
                imputer(),
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=200,
                    max_leaf_nodes=31,
                    l2_regularization=2.0,
                    random_state=SEED,
                ),
            ),
            True,
        ),
    }


def aligned_probabilities(estimator, features: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    aligned = np.zeros((len(features), len(BUDGET_BUCKETS)), dtype=np.float64)
    aligned[:, classes] = probabilities
    return aligned


def sequential_predict(
    estimator,
    frame: pd.DataFrame,
    calibrator: IsotonicRegression | None = None,
    policy: RoutePolicy | None = None,
) -> dict[str, np.ndarray]:
    """Predict in real DiT order so budget history is deployment-faithful."""

    working = frame.copy()
    state_keys = pd.MultiIndex.from_frame(working[["request_key", "layer_index", "head_index"]])
    state_codes, uniques = pd.factorize(state_keys, sort=False)
    previous = np.ones(len(uniques), dtype=np.float64)
    previous_two = np.ones(len(uniques), dtype=np.float64)
    raw_prediction = np.empty(len(working), dtype=np.int64)
    routed_prediction = np.empty(len(working), dtype=np.int64)
    raw_confidence = np.empty(len(working), dtype=np.float64)
    route_confidence = np.empty(len(working), dtype=np.float64)
    fallback = np.zeros(len(working), dtype=bool)
    probabilities = np.empty((len(working), len(BUDGET_BUCKETS)), dtype=np.float64)

    for dit_index in range(8):
        row_indices = np.flatnonzero(working["dit_index"].to_numpy() == dit_index)
        codes = state_codes[row_indices]
        working.loc[row_indices, "previous_route_budget"] = previous[codes]
        working.loc[row_indices, "previous_two_route_budget"] = previous_two[codes]
        step_probabilities = aligned_probabilities(
            estimator, working.loc[row_indices, FEATURE_COLUMNS]
        )
        step_raw = np.argmax(step_probabilities, axis=1)
        step_raw_confidence = np.max(step_probabilities, axis=1)
        step_route_confidence = (
            calibrator.predict(step_raw_confidence)
            if calibrator is not None
            else step_raw_confidence
        )
        step_route = step_raw.copy()
        step_fallback = np.zeros(len(row_indices), dtype=bool)
        if policy is not None:
            step_route = np.minimum(
                len(BUDGET_BUCKETS) - 1,
                step_route + policy.promotion_buckets,
            )
            step_fallback = step_route_confidence < policy.confidence_threshold
            step_route[step_fallback] = len(BUDGET_BUCKETS) - 1
        probabilities[row_indices] = step_probabilities
        raw_prediction[row_indices] = step_raw
        routed_prediction[row_indices] = step_route
        raw_confidence[row_indices] = step_raw_confidence
        route_confidence[row_indices] = step_route_confidence
        fallback[row_indices] = step_fallback
        old_previous = previous[codes].copy()
        previous[codes] = BUDGET_BUCKETS[step_route]
        previous_two[codes] = old_previous

    return {
        "probabilities": probabilities,
        "raw_prediction": raw_prediction,
        "prediction": routed_prediction,
        "raw_confidence": raw_confidence,
        "route_confidence": route_confidence,
        "fallback": fallback,
    }


def expected_calibration_error(
    confidence: np.ndarray, outcomes: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if np.any(mask):
            error += np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(outcomes[mask]))
    return float(error if len(confidence) else np.nan)


def selected_quality(frame: pd.DataFrame, prediction: np.ndarray, prefix: str) -> np.ndarray:
    matrix = frame[
        [f"{prefix}_{_ratio_suffix(ratio)}" for ratio in BUDGET_BUCKETS]
    ].to_numpy(dtype=np.float64)
    return matrix[np.arange(len(frame)), prediction]


def route_metrics(
    frame: pd.DataFrame,
    truth: np.ndarray,
    result: dict[str, np.ndarray],
) -> dict[str, object]:
    prediction = result["prediction"]
    false_sparse = prediction < truth
    critical = truth >= budget_indices([0.75])[0]
    mass = selected_quality(frame, prediction, "worst_mass_p05")
    cosine = selected_quality(frame, prediction, "worst_output_cosine_p05")
    relative_l2 = selected_quality(frame, prediction, "worst_output_relative_l2_p95")
    safe_outcome = prediction >= truth
    return {
        "row_count": len(frame),
        "macro_f1": float(
            f1_score(truth, prediction, labels=np.arange(len(BUDGET_BUCKETS)), average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=np.arange(len(BUDGET_BUCKETS))
        ).tolist(),
        "false_sparse_rate": float(np.mean(false_sparse)),
        "critical_false_sparse_rate": float(
            np.mean(false_sparse[critical]) if np.any(critical) else 0.0
        ),
        "mean_keep_ratio": float(np.mean(BUDGET_BUCKETS[prediction])),
        "dense_route_rate": float(np.mean(prediction == len(BUDGET_BUCKETS) - 1)),
        "confidence_fallback_rate": float(np.mean(result["fallback"])),
        "mass_p05_at_least_0_9_rate": float(np.mean(mass >= 0.9)),
        "local_output_gate_rate": float(np.mean((cosine >= 0.999) & (relative_l2 <= 0.05))),
        "route_confidence_ece": expected_calibration_error(
            result["route_confidence"], safe_outcome
        ),
    }


def calibrate_confidence(
    raw_result: dict[str, np.ndarray], truth: np.ndarray
) -> IsotonicRegression:
    safe = (raw_result["raw_prediction"] >= truth).astype(np.float64)
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        raw_result["raw_confidence"], safe
    )


def choose_policy(
    estimator,
    validation: pd.DataFrame,
    truth: np.ndarray,
    calibrator: IsotonicRegression,
    *,
    false_sparse_limit: float,
    mass_gate_rate: float,
) -> tuple[RoutePolicy, dict[str, np.ndarray], dict[str, object]]:
    raw = sequential_predict(estimator, validation)
    calibrated = calibrator.predict(raw["raw_confidence"])
    thresholds = np.unique(
        np.concatenate(
            (
                [0.0],
                np.quantile(calibrated, np.linspace(0.05, 0.95, 10)),
                [0.99, 1.000001],
            )
        )
    )
    feasible = []
    for promotion in range(len(BUDGET_BUCKETS)):
        for threshold in thresholds:
            policy = RoutePolicy(float(threshold), promotion)
            prediction = np.minimum(
                len(BUDGET_BUCKETS) - 1,
                raw["raw_prediction"] + promotion,
            )
            fallback = calibrated < threshold
            prediction = prediction.copy()
            prediction[fallback] = len(BUDGET_BUCKETS) - 1
            result = {
                **raw,
                "prediction": prediction,
                "route_confidence": calibrated,
                "fallback": fallback,
            }
            metrics = route_metrics(validation, truth, result)
            if (
                metrics["false_sparse_rate"] < false_sparse_limit
                and metrics["mass_p05_at_least_0_9_rate"] >= mass_gate_rate
            ):
                score = (
                    metrics["mean_keep_ratio"],
                    metrics["confidence_fallback_rate"],
                    metrics["false_sparse_rate"],
                )
                feasible.append((score, policy, result, metrics))
    if not feasible:
        raise RuntimeError("Dense fallback candidate unexpectedly failed M1 gates")
    _, policy, result, metrics = min(feasible, key=lambda item: item[0])
    return policy, result, metrics


def bootstrap_test_metrics(
    frame: pd.DataFrame,
    truth: np.ndarray,
    result: dict[str, np.ndarray],
    repeats: int,
) -> dict[str, dict[str, float]]:
    episodes = frame["source_episode_index"].to_numpy()
    unique = np.unique(episodes)
    by_episode = {episode: np.flatnonzero(episodes == episode) for episode in unique}
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in ("false_sparse_rate", "mean_keep_ratio", "mass_p05_at_least_0_9_rate")}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_episode[episode] for episode in sampled])
        subset_result = {key: value[indices] for key, value in result.items()}
        subset_metrics = route_metrics(frame.iloc[indices], truth[indices], subset_result)
        for name in values:
            values[name].append(subset_metrics[name])
    return {
        name: {
            "mean": float(np.mean(samples)),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
        }
        for name, samples in values.items()
    }


def route_semantics(frame: pd.DataFrame, result: dict[str, np.ndarray]) -> dict[str, object]:
    prediction = result["prediction"]
    fallback = result["fallback"]
    previous_turnover = frame["previous_support_turnover_max"].to_numpy()
    previous_vv_l2 = frame["previous_vv_output_change_relative_l2_max"].to_numpy()
    late = frame["dit_index"].to_numpy() >= 5
    categories = np.full(len(frame), "slow-changing", dtype=object)
    categories[prediction >= budget_indices([0.75])[0]] = "critical"
    stable = (prediction <= budget_indices([0.25])[0]) & (previous_turnover <= 0.20)
    categories[stable] = "stable"
    predictable = (
        late
        & (prediction <= budget_indices([0.35])[0])
        & (previous_turnover <= 0.20)
        & (previous_vv_l2 <= 0.05)
        & (frame["history_two_available"].to_numpy() > 0.5)
    )
    categories[predictable] = "predictable-late"
    categories[fallback] = "uncertain"
    counts = pd.Series(categories).value_counts().to_dict()
    return {
        "category_counts": {str(key): int(value) for key, value in counts.items()},
        "category_rules": {
            "critical": "routed budget >= 75%",
            "stable": "budget <=25% and previous support turnover <=0.20",
            "predictable-late": "DiT>=5, budget<=35%, two-step history, turnover<=0.20, previous VV relative L2<=0.05",
            "slow-changing": "remaining confident routes",
            "uncertain": "confidence-triggered Dense fallback",
        },
        "outputs": {
            "anchor_profile": "nested prefix selected by routed budget bucket",
            "refresh_frequency": {
                "critical": 1,
                "stable": 5,
                "slow-changing": 2,
                "predictable-late": 4,
                "uncertain": 1,
            },
            "allow_linear_extrapolation": "predictable-late only; sentinel gate remains mandatory",
        },
    }


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.read_parquet(args.input_table, columns=required_columns())
    frame["split"] = frame["split"].replace({"validation": "val"})
    split_frames = {
        split: frame.loc[frame["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    if any(part.empty for part in split_frames.values()):
        raise ValueError("Expected non-empty task-disjoint train/val/test splits")
    split_frames["train"], evaluations, prior_table = add_train_only_priors(
        split_frames["train"], [split_frames["val"], split_frames["test"]]
    )
    split_frames["val"], split_frames["test"] = evaluations
    split_frames["train"] = add_deployment_features(split_frames["train"])
    split_frames["val"] = add_deployment_features(split_frames["val"])
    split_frames["test"] = add_deployment_features(split_frames["test"])
    labels = {
        split: budget_indices(part["oracle_min_keep_ratio"])
        for split, part in split_frames.items()
    }

    available = candidate_estimators()
    selected_names = list(available) if args.models == ["all"] else args.models
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_results = {}
    fitted = {}
    for name in selected_names:
        estimator, cost_sensitive = available[name]
        cap = args.mlp_train_rows if name == "small_mlp" else args.max_train_rows
        indices = stratified_cap(split_frames["train"], labels["train"], cap)
        fit_features = split_frames["train"].iloc[indices][list(FEATURE_COLUMNS)]
        fit_labels = labels["train"][indices]
        fit_kwargs = {}
        if cost_sensitive:
            weights = 1.0 + args.underprediction_cost * (
                fit_labels / (len(BUDGET_BUCKETS) - 1)
            ) ** 2
            fit_kwargs["histgradientboostingclassifier__sample_weight"] = weights
        estimator.fit(fit_features, fit_labels, **fit_kwargs)
        raw_validation = sequential_predict(estimator, split_frames["val"])
        calibrator = calibrate_confidence(raw_validation, labels["val"])
        policy, validation_result, validation_metrics = choose_policy(
            estimator,
            split_frames["val"],
            labels["val"],
            calibrator,
            false_sparse_limit=args.false_sparse_limit,
            mass_gate_rate=args.mass_gate_rate,
        )
        test_result = sequential_predict(
            estimator, split_frames["test"], calibrator, policy
        )
        test_metrics = route_metrics(split_frames["test"], labels["test"], test_result)
        raw_safe = raw_validation["raw_prediction"] >= labels["val"]
        model_results[name] = {
            "train_rows": int(len(indices)),
            "validation": validation_metrics,
            "test": test_metrics,
            "policy": {
                "confidence_threshold": policy.confidence_threshold,
                "promotion_buckets": policy.promotion_buckets,
            },
            "validation_raw_safe_confidence_ece": expected_calibration_error(
                raw_validation["raw_confidence"], raw_safe
            ),
            "test_bootstrap_200": bootstrap_test_metrics(
                split_frames["test"],
                labels["test"],
                test_result,
                args.bootstrap_repeats,
            ),
            "test_route_semantics": route_semantics(split_frames["test"], test_result),
        }
        fitted[name] = (estimator, calibrator, policy, test_result)
        (args.output_dir / f"{name}_metrics.json").write_text(
            json.dumps(model_results[name], indent=2) + "\n"
        )

    feasible = [
        name
        for name in selected_names
        if model_results[name]["validation"]["false_sparse_rate"]
        < args.false_sparse_limit
        and model_results[name]["validation"]["mass_p05_at_least_0_9_rate"]
        >= args.mass_gate_rate
        and model_results[name]["validation"]["macro_f1"] >= args.minimum_macro_f1
        and (
            not args.require_confidence_fallback
            or model_results[name]["validation"]["confidence_fallback_rate"] > 0.0
        )
    ]
    if not feasible:
        raise RuntimeError("No M1 candidate passed validation routing gates")
    best_name = min(
        feasible,
        key=lambda name: (
            model_results[name]["validation"]["false_sparse_rate"],
            model_results[name]["validation"]["mean_keep_ratio"],
            -model_results[name]["validation"]["macro_f1"],
        ),
    )
    best_estimator, best_calibrator, best_policy, best_test_result = fitted[best_name]
    bundle = {
        "model_name": best_name,
        "estimator": best_estimator,
        "confidence_calibrator": best_calibrator,
        "policy": best_policy,
        "feature_columns": FEATURE_COLUMNS,
        "budget_buckets": BUDGET_BUCKETS,
        "prior_table": prior_table,
        "schema_version": 1,
    }
    joblib.dump(bundle, args.output_dir / "selected_m1_bundle.joblib", compress=3)
    prior_table.to_parquet(args.output_dir / "m1_prior_table.parquet", index=False)
    test_metrics = model_results[best_name]["test"]
    statistical_gates = (
        test_metrics["false_sparse_rate"] < args.false_sparse_limit
        and test_metrics["mass_p05_at_least_0_9_rate"] >= args.mass_gate_rate
        and test_metrics["macro_f1"] >= args.minimum_macro_f1
        and (
            not args.require_confidence_fallback
            or test_metrics["confidence_fallback_rate"] > 0.0
        )
    )
    summary = {
        "input_table": str(args.input_table),
        "feature_columns": list(FEATURE_COLUMNS),
        "forbidden_current_dense_features": [
            "support_turnover_max",
            "vv_output_change_relative_l2_max",
            "normalized_entropy_mean",
            "max_attention_mass_mean",
            "qa_qv_key_importance_correlation_mean",
            "worst_mass/output metrics",
        ],
        "split_rows": {key: len(value) for key, value in split_frames.items()},
        "split_episodes": {
            key: int(value["source_episode_index"].nunique())
            for key, value in split_frames.items()
        },
        "models": model_results,
        "selected_model": best_name,
        "selection_gates": {
            "false_sparse_limit": args.false_sparse_limit,
            "mass_gate_rate": args.mass_gate_rate,
            "minimum_macro_f1": args.minimum_macro_f1,
            "require_confidence_fallback": args.require_confidence_fallback,
            "ordering": "validation false-sparse, mean keep ratio, macro-F1",
        },
        "prior_table": str(args.output_dir / "m1_prior_table.parquet"),
        "prior_table_rows": len(prior_table),
        "statistical_gates_passed": bool(statistical_gates),
        "final_action_cosine_gate": "pending actual DreamZero policy replay",
        "passed": False,
        "reason": (
            "Classifier statistics are only one M1 gate; final action cosine >=0.999 "
            "must be measured after dynamic routing integration."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Candidate names or 'all'",
    )
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--mlp-train-rows", type=int, default=200_000)
    parser.add_argument("--underprediction-cost", type=float, default=20.0)
    parser.add_argument("--false-sparse-limit", type=float, default=0.01)
    parser.add_argument("--mass-gate-rate", type=float, default=0.95)
    parser.add_argument("--minimum-macro-f1", type=float, default=0.50)
    parser.add_argument(
        "--require-confidence-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    args = parser.parse_args()
    summary = train_and_evaluate(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
