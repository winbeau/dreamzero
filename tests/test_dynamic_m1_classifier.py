from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.train_dynamic_m1_classifier import (
    BASE_COLUMNS,
    QUALITY_PREFIXES,
    RoutePolicy,
    add_train_only_priors,
    budget_indices,
    route_metrics,
    sequential_predict,
    train_and_evaluate,
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


def test_small_task_disjoint_training_pipeline(tmp_path):
    rows = []
    split_episodes = {"train": (0, 1), "val": (2,), "test": (3,)}
    for split, episodes in split_episodes.items():
        for episode in episodes:
            request_key = f"{split}-{episode}"
            previous = {}
            previous_two = {}
            for dit_index in range(8):
                for layer_index in range(2):
                    for head_index in range(2):
                        state_key = (layer_index, head_index)
                        target_index = (dit_index + layer_index + head_index) % 7
                        target = BUDGET_BUCKETS[target_index]
                        row = {
                            "request_key": request_key,
                            "split": split,
                            "source_episode_index": episode,
                            "trajectory_stage": ("early", "middle", "late")[episode % 3],
                            "trajectory_fraction": episode / 3.0,
                            "trajectory_length": 100 + episode,
                            "length_bucket": ("short", "middle", "long")[episode % 3],
                            "instruction_index": episode % 3,
                            "state_l2": 1.0 + episode,
                            "state_abs_mean": 0.1 + episode,
                            "action_l2": 2.0 + episode,
                            "action_std": 0.2 + episode,
                            "action_temporal_delta_l2": 0.3 + episode,
                            "dit_index": dit_index,
                            "scheduler_index": (0, 1, 2, 6, 10, 13, 14, 15)[dit_index],
                            "diffusion_timestep": 1000 - 100 * dit_index,
                            "layer_index": layer_index,
                            "head_index": head_index,
                            "timestep_position": dit_index / 7.0,
                            "layer_depth": layer_index / 39.0,
                            "oracle_min_keep_ratio": target,
                            "previous_oracle_min_keep_ratio": previous.get(state_key, np.nan),
                            "previous_two_oracle_min_keep_ratio": previous_two.get(
                                state_key, np.nan
                            ),
                            "previous_support_turnover_max": (
                                np.nan if dit_index == 0 else 0.1
                            ),
                            "previous_vv_output_change_relative_l2_max": (
                                np.nan if dit_index == 0 else 0.02
                            ),
                            "previous_two_vv_output_change_relative_l2_max": (
                                np.nan if dit_index < 2 else 0.03
                            ),
                            "previous_normalized_entropy_mean": (
                                np.nan if dit_index == 0 else 0.5
                            ),
                            "previous_max_attention_mass_mean": (
                                np.nan if dit_index == 0 else 0.2
                            ),
                            "previous_qa_qv_key_importance_correlation_mean": (
                                np.nan if dit_index == 0 else 0.8
                            ),
                        }
                        for ratio in BUDGET_BUCKETS:
                            suffix = f"r{int(round(ratio * 100)):03d}"
                            safe = ratio >= target
                            row[f"worst_mass_p05_{suffix}"] = 0.95 if safe else 0.5
                            row[f"worst_output_cosine_p05_{suffix}"] = (
                                1.0 if safe else 0.9
                            )
                            row[f"worst_output_relative_l2_p95_{suffix}"] = (
                                0.0 if safe else 0.2
                            )
                        rows.append(row)
                        previous_two[state_key] = previous.get(state_key, 1.0)
                        previous[state_key] = target
    table = pd.DataFrame(rows)
    assert set(BASE_COLUMNS).issubset(table.columns)
    assert all(
        any(column.startswith(prefix) for column in table.columns)
        for prefix in QUALITY_PREFIXES
    )
    input_path = tmp_path / "m1.parquet"
    output_dir = tmp_path / "output"
    table.to_parquet(input_path)

    summary = train_and_evaluate(
        Namespace(
            input_table=input_path,
            output_dir=output_dir,
            models=["gradient_boosting"],
            max_train_rows=0,
            mlp_train_rows=0,
            underprediction_cost=20.0,
            false_sparse_limit=0.01,
            mass_gate_rate=0.95,
            bootstrap_repeats=5,
        )
    )

    assert summary["selected_model"] == "gradient_boosting"
    assert summary["statistical_gates_passed"]
    assert not summary["passed"]
    assert (output_dir / "selected_m1_bundle.joblib").is_file()
