from benchmarks.build_early_video_history_tables import candidate_tables
from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedBudgetTable,
)


def test_early_video_history_floor_preserves_current_and_other_layers() -> None:
    base = DynamicPackedBudgetTable.constant(
        num_dit_steps=8,
        num_layers=40,
        history_keep_ratio=0.20,
        current_keep_ratio=0.35,
        name="base",
    )
    tables = {table.name: table for table in candidate_tables(base)}

    assert set(tables) == {
        "base_early_history_50",
        "base_early_history_75",
        "base_early_history_100",
    }
    dense = tables["base_early_history_100"]
    assert dense.current_keep_ratios == base.current_keep_ratios
    assert dense.ratios(0, 0) == (0.20, 0.35)
    assert dense.ratios(0, 1) == (1.00, 0.35)
    assert dense.ratios(7, 13) == (1.00, 0.35)
    assert dense.ratios(7, 14) == (0.20, 0.35)
