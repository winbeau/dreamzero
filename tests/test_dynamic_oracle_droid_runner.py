import numpy as np

from benchmarks.collect_dynamic_attention_oracle_droid import (
    _condition_summary,
    _pad_leading_video_frames,
)


def test_boundary_video_padding_restores_four_frame_ar_block():
    frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    data_point = {
        "video.exterior_image_1_left": frames.copy(),
        "state.joint_position": np.zeros((1, 7)),
    }

    _pad_leading_video_frames(data_point, 4)

    padded = data_point["video.exterior_image_1_left"]
    assert padded.shape == (4, 2, 2, 3)
    assert np.array_equal(padded[0], frames[0])
    assert np.array_equal(padded[1:], frames)


def test_condition_summary_uses_real_state_and_action_values():
    data_point = {
        "state.joint_position": np.asarray([[3.0, 4.0]]),
        "action.joint_position": np.asarray([[0.0, 0.0], [3.0, 4.0]]),
    }

    summary = _condition_summary(data_point)

    assert summary["state_l2"] == 5.0
    assert summary["state_abs_mean"] == 3.5
    assert summary["action_l2"] == 5.0
    assert summary["action_temporal_delta_l2"] == 5.0
    assert summary["action_std"] > 0.0


def test_condition_summary_can_be_attached_to_request_metadata():
    metadata = {"request_key": "request"}
    data_point = {
        "state.joint_position": np.asarray([[3.0, 4.0]]),
        "action.joint_position": np.asarray([[0.0, 0.0], [3.0, 4.0]]),
    }

    metadata.update(_condition_summary(data_point))

    assert metadata["request_key"] == "request"
    assert metadata["state_l2"] == 5.0
