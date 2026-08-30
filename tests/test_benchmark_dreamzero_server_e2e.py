import numpy as np
import pytest

from benchmarks.benchmark_dreamzero_server_e2e import make_request, summarize
from benchmarks.compare_dreamzero_server_e2e import compare_reports


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
