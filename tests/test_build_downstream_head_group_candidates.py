import pytest

from benchmarks.build_downstream_head_group_candidates import build_candidates


def test_build_candidates_covers_each_fixed_head_group_per_cell() -> None:
    candidates = build_candidates(
        dit_indices=(0, 7),
        layer_indices=(0, 39),
        num_heads=8,
        group_size=4,
    )

    assert len(candidates) == 8
    assert candidates[0] == {
        "label": "d0_l0_h00_03_all",
        "dit_index": 0,
        "layer_index": 0,
        "head_indices": [0, 1, 2, 3],
        "scale": 0.0,
        "query_scope": "all",
    }
    assert candidates[-1]["label"] == "d7_l39_h04_07_all"
    for dit_index in (0, 7):
        for layer_index in (0, 39):
            groups = [
                candidate["head_indices"]
                for candidate in candidates
                if candidate["dit_index"] == dit_index
                and candidate["layer_index"] == layer_index
            ]
            assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dit_indices": ()},
        {"dit_indices": (0, 0)},
        {"layer_indices": ()},
        {"layer_indices": (1, 1)},
        {"num_heads": 7, "group_size": 4},
        {"query_scopes": ("invalid",)},
    ),
)
def test_build_candidates_rejects_invalid_fixed_shapes(kwargs) -> None:
    defaults = {
        "dit_indices": (0,),
        "layer_indices": (0,),
        "num_heads": 8,
        "group_size": 4,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        build_candidates(**defaults)
