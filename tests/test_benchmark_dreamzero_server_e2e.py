import numpy as np
import pytest

from benchmarks.benchmark_dreamzero_server_e2e import make_request, summarize


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
