import pytest

from eval_utils.sim_eval_metrics import evaluate_position_success


@pytest.mark.parametrize(
    ("scene", "source", "target"),
    [
        (1, (0.41, 0.17, 0.11), (0.405, 0.174, 0.078)),
        (2, (0.44, -0.08, 0.13), (0.439, -0.078, 0.096)),
        (3, (0.50, 0.20, 0.17), (0.496, 0.198, 0.125)),
    ],
)
def test_scene_success_proxy_accepts_centered_objects(scene, source, target):
    assert evaluate_position_success(scene, source, target)["success"] is True


@pytest.mark.parametrize(
    ("scene", "source", "target"),
    [
        (1, (0.36, -0.08, 0.103), (0.405, 0.174, 0.078)),
        (2, (0.454, 0.062, 0.099), (0.439, -0.078, 0.096)),
        (3, (0.422, -0.111, 0.096), (0.496, 0.198, 0.125)),
    ],
)
def test_scene_success_proxy_rejects_off_target_initial_states(scene, source, target):
    assert evaluate_position_success(scene, source, target)["success"] is False


def test_scene_success_proxy_rejects_unknown_scene():
    with pytest.raises(ValueError, match="not supported"):
        evaluate_position_success(4, (0, 0, 0), (0, 0, 0))
