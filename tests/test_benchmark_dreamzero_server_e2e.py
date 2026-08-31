import json
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.benchmark_dreamzero_server_e2e import make_request, summarize
from benchmarks.benchmark_dreamzero_server_droid import (
    build_request_plan,
    history_frame_groups,
    split_state,
)
from benchmarks.benchmark_downstream_head_sensitivity_droid import (
    RETURN_VIDEO_KEY,
    action_sensitivity_metrics,
    intervention_control,
    run_chain,
    run_target,
    validate_downstream_trace,
    video_sensitivity_metrics,
)
from benchmarks.benchmark_downstream_head_sensitivity_grid_droid import (
    load_candidates,
    summarize_candidate_records,
    validate_resume_video_schema,
)
from benchmarks.analyze_downstream_oracle_alignment import (
    average_tie_ranks,
    finite_correlation,
)
from benchmarks.compare_dreamzero_server_e2e import compare_reports
from benchmarks.summarize_dreamzero_server_log import summarize_log
from benchmarks.validate_downstream_exactness import validate_exactness_report


def test_make_request_is_deterministic_and_has_server_shapes():
    first = make_request(request_index=2, seed=7, session_id="test")
    second = make_request(request_index=2, seed=7, session_id="test")

    for key in (
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        assert first[key].shape == (180, 320, 3)
        assert first[key].dtype == np.uint8
        np.testing.assert_array_equal(first[key], second[key])

    assert first["observation/joint_position"].shape == (7,)
    assert first["observation/cartesian_position"].shape == (6,)
    assert first["observation/gripper_position"].shape == (1,)


def test_summarize_reports_expected_statistics():
    result = summarize([1.0, 2.0, 3.0, 4.0])
    assert result["mean_seconds"] == pytest.approx(2.5)
    assert result["median_seconds"] == pytest.approx(2.5)
    assert result["min_seconds"] == pytest.approx(1.0)
    assert result["max_seconds"] == pytest.approx(4.0)


def test_downstream_action_sensitivity_metrics_are_paired() -> None:
    baseline = np.asarray([[1.0, 0.0]])
    intervened = np.asarray([[0.8, 0.2]])

    metrics = action_sensitivity_metrics(baseline, intervened)

    assert metrics["action_cosine"] == pytest.approx(0.9701425)
    assert metrics["action_relative_l2"] == pytest.approx(np.sqrt(0.08))
    assert metrics["action_max_abs"] == pytest.approx(0.2)


def test_downstream_video_sensitivity_metrics_are_paired() -> None:
    baseline = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    intervened = np.asarray([[[0.8, 0.2], [0.1, 0.9]]])

    metrics = video_sensitivity_metrics(baseline, intervened)

    assert metrics["video_cosine"] == pytest.approx(1.7 / np.sqrt(3.0))
    assert metrics["video_relative_l2"] == pytest.approx(np.sqrt(0.1) / np.sqrt(2))
    assert metrics["video_max_abs"] == pytest.approx(0.2)

    with pytest.raises(ValueError, match="shapes do not match"):
        video_sensitivity_metrics(np.zeros((2, 4)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="must be finite"):
        video_sensitivity_metrics(np.asarray([1.0]), np.asarray([np.nan]))


def test_downstream_run_target_requests_and_parses_video_latent() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.request = None

        def infer(self, observation):
            self.request = observation.copy()
            return {
                "action": np.asarray([[1.0, 2.0]]),
                "video": np.asarray([[[3.0, 4.0]]]),
                "downstream_trace": {
                    "configured": False,
                    "applied_count": 0,
                },
            }

    client = FakeClient()
    result = run_target(
        client,
        {"frame": 7},
        target_control=None,
        return_video=True,
    )

    assert client.request[RETURN_VIDEO_KEY] is True
    assert client.request["dynamic_downstream_head_intervention"] == {
        "enabled": False
    }
    np.testing.assert_array_equal(result.action, np.asarray([[1.0, 2.0]]))
    np.testing.assert_array_equal(result.video, np.asarray([[[3.0, 4.0]]]))
    assert result.downstream_trace == {
        "configured": False,
        "applied_count": 0,
    }
    assert result.latency_seconds >= 0.0


def test_downstream_trace_requires_exactly_one_target_application() -> None:
    control = intervention_control(
        dit_index=0,
        layer_index=39,
        head_indices=(12, 13, 14, 15),
        scale=1.0,
        query_scope="all",
    )
    trace = {
        "configured": True,
        "dit_index": 0,
        "layer_index": 39,
        "head_indices": [12, 13, 14, 15],
        "scale": 1.0,
        "cfg_branches": ["conditional"],
        "query_scope": "all",
        "applied_count": 1,
    }

    validate_downstream_trace(trace, expected_control=control)
    trace["applied_count"] = 2
    with pytest.raises(ValueError, match="trace mismatch"):
        validate_downstream_trace(trace, expected_control=control)


def test_downstream_intervention_control_is_conditional_only() -> None:
    assert intervention_control(
        dit_index=2,
        layer_index=7,
        head_indices=(1, 4),
        scale=0.0,
        query_scope="register",
    ) == {
        "enabled": True,
        "dit_index": 2,
        "layer_index": 7,
        "head_indices": [1, 4],
        "scale": 0.0,
        "cfg_branches": ["conditional"],
        "query_scope": "register",
    }


def test_downstream_run_chain_disables_history_and_controls_only_target() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.requests = []
            self.resets = []

        def infer(self, observation):
            self.requests.append(observation.copy())
            return np.asarray([[len(self.requests), 0.0]])

        def reset(self, reset_info):
            self.resets.append(reset_info.copy())

    client = FakeClient()
    observations = [{"frame": index} for index in range(4)]
    control = intervention_control(
        dit_index=0,
        layer_index=39,
        head_indices=(14,),
        scale=0.0,
        query_scope="all",
    )

    action, history_latencies, target_latency = run_chain(
        client,
        observations,
        target_control=control,
        session_id="paired",
    )

    assert np.array_equal(action, np.asarray([[4, 0.0]]))
    assert len(history_latencies) == 3
    assert target_latency >= 0.0
    assert [
        request["dynamic_downstream_head_intervention"]
        for request in client.requests[:-1]
    ] == [{"enabled": False}] * 3
    assert client.requests[-1]["dynamic_downstream_head_intervention"] == control
    assert client.resets == [{"session_id": "paired"}]


def test_downstream_grid_loads_unique_candidates(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "label": "early-critical",
                        "dit_index": 0,
                        "layer_index": 20,
                        "head_indices": [1, 4],
                        "query_scope": "all",
                    }
                ]
            }
        )
    )

    assert load_candidates(path) == [
        {
            "label": "early-critical",
            "control": {
                "enabled": True,
                "dit_index": 0,
                "layer_index": 20,
                "head_indices": [1, 4],
                "scale": 0.0,
                "cfg_branches": ["conditional"],
                "query_scope": "all",
            },
        }
    ]


def test_downstream_grid_summary_records_worst_and_stage_means() -> None:
    records = [
        {
            "request_key": "early",
            "trajectory_stage": "early",
            "action_cosine": 0.99,
            "action_relative_l2": 0.04,
            "action_max_abs": 0.2,
        },
        {
            "request_key": "late",
            "trajectory_stage": "late",
            "action_cosine": 0.999,
            "action_relative_l2": 0.01,
            "action_max_abs": 0.1,
        },
    ]

    summary = summarize_candidate_records(records)

    assert summary["measured_requests"] == 2
    assert summary["action_cosine_mean"] == pytest.approx(0.9945)
    assert summary["action_relative_l2_mean"] == pytest.approx(0.025)
    assert summary["action_relative_l2_max"] == pytest.approx(0.04)
    assert summary["worst_request_key"] == "early"
    assert summary["stage_relative_l2_mean"] == {
        "early": pytest.approx(0.04),
        "late": pytest.approx(0.01),
    }


def test_downstream_grid_summary_includes_video_worst_case() -> None:
    records = [
        {
            "request_key": "early",
            "trajectory_stage": "early",
            "action_cosine": 0.999,
            "action_relative_l2": 0.01,
            "action_max_abs": 0.1,
            "video_cosine": 0.98,
            "video_relative_l2": 0.08,
            "video_max_abs": 0.4,
        },
        {
            "request_key": "late",
            "trajectory_stage": "late",
            "action_cosine": 0.9999,
            "action_relative_l2": 0.002,
            "action_max_abs": 0.02,
            "video_cosine": 0.995,
            "video_relative_l2": 0.02,
            "video_max_abs": 0.2,
        },
    ]

    summary = summarize_candidate_records(records)

    assert summary["video_cosine_min"] == pytest.approx(0.98)
    assert summary["video_relative_l2_mean"] == pytest.approx(0.05)
    assert summary["video_relative_l2_max"] == pytest.approx(0.08)
    assert summary["video_worst_request_key"] == "early"
    assert summary["stage_video_relative_l2_mean"] == {
        "early": pytest.approx(0.08),
        "late": pytest.approx(0.02),
    }


def test_downstream_resume_rejects_mismatched_video_schema() -> None:
    validate_resume_video_schema([], record_video_sensitivity=False)
    validate_resume_video_schema(
        [
            {
                "video_cosine": 1.0,
                "video_relative_l2": 0.0,
                "video_max_abs": 0.0,
            }
        ],
        record_video_sensitivity=True,
    )
    with pytest.raises(ValueError, match="partial video metrics"):
        validate_resume_video_schema(
            [{"video_cosine": 1.0}],
            record_video_sensitivity=True,
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_video_schema(
            [{"request_key": "action-only"}],
            record_video_sensitivity=True,
        )


def _exactness_report() -> dict:
    control = intervention_control(
        dit_index=0,
        layer_index=39,
        head_indices=(12, 13, 14, 15),
        scale=1.0,
        query_scope="all",
    )
    return {
        "record_video_sensitivity": True,
        "reuse_history_snapshot": True,
        "records": [
            {
                "request_key": "validation_early",
                "candidate_label": "scale1",
                "intervention": control,
                "action_shape": [1, 2],
                "baseline_action": [[1.0, 2.0]],
                "intervention_action": [[1.0, 2.0]],
                "action_cosine": 1.0,
                "action_relative_l2": 0.0,
                "action_max_abs": 0.0,
                "video_shape": [1, 16, 4, 44, 80],
                "video_cosine": 1.0,
                "video_relative_l2": 0.0,
                "video_max_abs": 0.0,
                "baseline_downstream_trace": {
                    "configured": False,
                    "applied_count": 0,
                },
                "intervention_downstream_trace": {
                    "configured": True,
                    "dit_index": 0,
                    "layer_index": 39,
                    "head_indices": [12, 13, 14, 15],
                    "scale": 1.0,
                    "cfg_branches": ["conditional"],
                    "query_scope": "all",
                    "applied_count": 1,
                },
            }
        ],
    }


def test_downstream_exactness_report_requires_all_gates() -> None:
    assert validate_exactness_report(
        _exactness_report(),
        expected_records=1,
    ) == {
        "exact": True,
        "records": 1,
        "requests": 1,
        "candidates": 1,
        "action_elementwise_exact": True,
        "video_difference_exact": True,
        "intervention_applied_once": True,
    }

    action_failure = deepcopy(_exactness_report())
    action_failure["records"][0]["intervention_action"][0][1] = 2.1
    with pytest.raises(ValueError, match="action arrays"):
        validate_exactness_report(action_failure)

    video_failure = deepcopy(_exactness_report())
    video_failure["records"][0]["video_relative_l2"] = 1e-8
    with pytest.raises(ValueError, match="video output is not exact"):
        validate_exactness_report(video_failure)

    trace_failure = deepcopy(_exactness_report())
    trace_failure["records"][0]["intervention_downstream_trace"][
        "applied_count"
    ] = 0
    with pytest.raises(ValueError, match="trace mismatch"):
        validate_exactness_report(trace_failure)


def test_downstream_alignment_correlations_handle_ties() -> None:
    np.testing.assert_array_equal(
        average_tie_ranks(np.asarray([2.0, 1.0, 2.0, 4.0])),
        np.asarray([1.5, 0.0, 1.5, 3.0]),
    )
    assert finite_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(
        -1.0
    )
    assert finite_correlation(
        [1.0, 1.0, 2.0],
        [4.0, 4.0, 5.0],
        rank=True,
    ) == pytest.approx(1.0)
    assert finite_correlation([1.0, 1.0], [2.0, 3.0]) is None


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        summarize([])


def _report(label, seed, latencies):
    return {
        "label": label,
        "seed": seed,
        "records": [
            {
                "request_index": index,
                "phase": "measured",
                "latency_seconds": latency,
                "action_shape": [24, 8],
            }
            for index, latency in enumerate(latencies, start=2)
        ],
    }


def test_compare_reports_computes_paired_speedup_and_ci():
    dense = _report("dense", 7, [2.0, 4.0, 6.0])
    sparse = _report("sparse", 7, [1.0, 2.0, 3.0])

    result = compare_reports(dense, sparse, bootstrap_samples=100, bootstrap_seed=9)

    assert result["mean_latency_speedup"] == pytest.approx(2.0)
    assert result["p50_latency_speedup"] == pytest.approx(2.0)
    assert result["paired_geometric_mean_speedup"] == pytest.approx(2.0)
    assert result["paired_geometric_mean_speedup_ci95"] == pytest.approx([2.0, 2.0])
    assert result["sparse_faster_fraction"] == pytest.approx(1.0)


def test_compare_reports_rejects_mismatched_seed():
    dense = _report("dense", 7, [2.0])
    sparse = _report("sparse", 8, [1.0])

    with pytest.raises(ValueError, match="different request seeds"):
        compare_reports(dense, sparse, bootstrap_samples=10)


def test_compare_reports_includes_paired_action_quality_and_worst_request():
    dense = _report("dense", 7, [2.0, 2.0])
    sparse = _report("sparse", 7, [1.0, 1.0])
    dense["records"][0]["action"] = [[1.0, 0.0]]
    dense["records"][1]["action"] = [[1.0, 0.0]]
    sparse["records"][0]["action"] = [[1.0, 0.0]]
    sparse["records"][1]["action"] = [[0.8, 0.2]]

    result = compare_reports(dense, sparse, bootstrap_samples=10)

    assert result["action_cosine_mean"] < 1.0
    assert result["action_cosine_min"] == pytest.approx(0.9701425)
    assert result["action_relative_l2_mean"] == pytest.approx(0.14142136)
    assert result["action_relative_l2_max"] == pytest.approx(0.28284271)
    assert result["worst_action_request_index"] == 3


def test_droid_request_plan_and_history_match_oracle_protocol():
    manifest = {
        "selections": [
            {
                "split": "test",
                "subset_episode_index": 2,
                "source_episode_index": 17,
                "length": 100,
                "tasks": ["first", "second", "third"],
                "trajectory_stages": [
                    {"name": "early", "fraction": 0.1},
                    {"name": "middle", "fraction": 0.5},
                ],
            }
        ]
    }
    plan = build_request_plan(
        manifest, splits={"test"}, stages={"middle"}, max_requests=None
    )
    assert [request["request_key"] for request in plan] == [
        "test_subset002_source000017_middle"
    ]
    assert history_frame_groups(
        base_step=50, trajectory_length=100, history_blocks=3
    ) == [[38], [39, 40, 41, 42], [43, 44, 45, 46], [47, 48, 49, 50]]


def test_split_droid_state_uses_modality_slices():
    joint, cartesian, gripper = split_state(np.arange(14, dtype=np.float64))
    np.testing.assert_array_equal(joint, np.arange(7, 14))
    np.testing.assert_array_equal(cartesian, np.arange(6))
    np.testing.assert_array_equal(gripper, np.asarray([6.0]))


def test_summarize_log_uses_latest_run_and_drops_warmup():
    rows = []
    for value in (9.0, 8.0, 3.0, 1.0, 2.0):
        rows.append(
            "Time taken: Total "
            f"{value:.2f} seconds, Text Encoder 0.10 seconds, "
            "Image Encoder 0.20 seconds, VAE 0.30 seconds, "
            "KV Cache Creation 0.40 seconds, Diffusion 0.50 seconds, "
            "DIT Compute Steps 8 steps, Scheduler 0.60 seconds"
        )
        rows.append(
            "Inference Time: Total "
            f"{value + 0.1:.2f} seconds, Transform: 0.01 seconds, "
            f"Model: {value:.2f} seconds, Untransform: 0.09 seconds"
        )

    result = summarize_log("\n".join(rows), total_requests=3, warmup_requests=1)

    assert result["measured_requests"] == 2
    assert result["dit_steps"] == [8]
    assert result["stage_mean_seconds"]["total"] == pytest.approx(1.5)
    assert result["inference_mean_seconds"]["model"] == pytest.approx(1.5)
