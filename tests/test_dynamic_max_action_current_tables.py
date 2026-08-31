import pytest

from benchmarks.build_dynamic_max_action_current_tables import (
    candidate_tables,
    propagation_segment_layers,
)


def test_propagation_segment_entries_and_exits_match_packed_boundaries() -> None:
    entries, exits = propagation_segment_layers()

    assert entries == (1, 6, 11, 16, 21, 26, 31, 36)
    assert exits == (5, 10, 15, 20, 25, 30, 35, 38)


def test_max_action_current_candidates_have_fixed_dreamzero_shape() -> None:
    tables = {table.name: table for table in candidate_tables()}

    assert set(tables) == {
        "none",
        "all_middle",
        "segment_entries",
        "segment_exits",
        "early_dit_segment_entries",
        "late_dit_segment_entries",
    }
    assert all(table.num_dit_steps == 8 for table in tables.values())
    assert all(table.num_layers == 40 for table in tables.values())
    assert not any(any(row) for row in tables["none"].enabled_cells)
    assert tables["all_middle"].enabled(0, 1)
    assert not tables["all_middle"].enabled(0, 0)
    assert not tables["all_middle"].enabled(0, 39)
    assert tables["segment_entries"].enabled(0, 1)
    assert not tables["segment_entries"].enabled(0, 5)
    assert tables["segment_exits"].enabled(0, 5)
    assert not tables["segment_exits"].enabled(0, 6)
    assert tables["early_dit_segment_entries"].enabled(1, 36)
    assert not tables["early_dit_segment_entries"].enabled(2, 36)
    assert tables["late_dit_segment_entries"].enabled(4, 1)
    assert not tables["late_dit_segment_entries"].enabled(3, 1)


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"propagate_every": 0}, "propagate_every"),
        ({"dense_prefix_layers": 39}, "Packed middle"),
        ({"dense_suffix_layers": -1}, "non-negative"),
    ),
)
def test_invalid_segment_layout_is_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        propagation_segment_layers(**kwargs)
