import numpy as np
import pytest

from benchmarks.benchmark_dreamzero_server_e2e import make_request, summarize
from benchmarks.compare_dreamzero_server_e2e import compare_reports
from benchmarks.summarize_dreamzero_server_log import summarize_log


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
