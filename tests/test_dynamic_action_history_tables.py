from benchmarks.build_dynamic_action_history_tables import candidate_tables


def test_action_history_candidates_have_fixed_dreamzero_shape() -> None:
    tables = {table.name: table for table in candidate_tables()}

    assert set(tables) == {
        "none",
        "all_middle",
        "early_layers",
        "middle_layers",
        "late_layers",
        "early_dit_all_middle",
        "late_dit_all_middle",
        "late_dit_late_layers",
    }
    assert all(table.num_dit_steps == 8 for table in tables.values())
    assert all(table.num_layers == 40 for table in tables.values())
    assert not any(any(row) for row in tables["none"].enabled_cells)
    assert tables["all_middle"].enabled(0, 1)
    assert not tables["all_middle"].enabled(0, 0)
    assert not tables["all_middle"].enabled(0, 39)
    assert tables["late_dit_late_layers"].enabled(4, 28)
    assert not tables["late_dit_late_layers"].enabled(3, 28)
    assert not tables["late_dit_late_layers"].enabled(4, 27)
