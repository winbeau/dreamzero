import numpy as np

from benchmarks.collect_dynamic_attention_oracle_droid import (
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
