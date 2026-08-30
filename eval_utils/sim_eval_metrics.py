from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# Conservative center-in-container proxies for the three official DROID scenes.
# The final videos remain the authoritative audit artifact.
SCENE_SUCCESS_SPECS = {
    1: {
        "source_asset": "rubiks_cube",
        "target_asset": "_24_bowl",
        "max_xy_distance_m": 0.08,
        "min_z_offset_m": -0.02,
        "max_z_offset_m": 0.10,
    },
    2: {
        "source_asset": "_10_potted_meat_can",
        "target_asset": "_25_mug",
        "max_xy_distance_m": 0.055,
        "min_z_offset_m": -0.02,
        "max_z_offset_m": 0.12,
    },
    3: {
        "source_asset": "_11_banana",
        "target_asset": "small_KLT_visual_collision",
        "max_xy_distance_m": 0.16,
        "min_z_offset_m": -0.02,
        "max_z_offset_m": 0.18,
    },
}


def evaluate_position_success(
    scene: int,
    source_position: Sequence[float],
    target_position: Sequence[float],
) -> dict[str, object]:
    if scene not in SCENE_SUCCESS_SPECS:
        raise ValueError(f"Scene {scene} is not supported")

    source = np.asarray(source_position, dtype=np.float64).reshape(3)
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    spec = SCENE_SUCCESS_SPECS[scene]

    xy_distance_m = float(np.linalg.norm(source[:2] - target[:2]))
    z_offset_m = float(source[2] - target[2])
    success = (
        xy_distance_m <= spec["max_xy_distance_m"]
        and spec["min_z_offset_m"] <= z_offset_m <= spec["max_z_offset_m"]
    )
    return {
        "success": bool(success),
        "xy_distance_m": xy_distance_m,
        "z_offset_m": z_offset_m,
        "source_position_m": source.tolist(),
        "target_position_m": target.tolist(),
        **spec,
    }
