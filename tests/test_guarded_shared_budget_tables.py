import numpy as np
import pandas as pd

from benchmarks.build_guarded_shared_budget_tables import build_guarded_tables


def test_guarded_tables_select_oracle_safe_cells_and_keep_dense_prefix() -> None:
    timestep = pd.DataFrame(
        {
            "dit_index": range(8),
            "oracle_mean": [8.0, 7.0, 6.0, 5.0, 4.0, 1.0, 3.0, 2.0],
        }
    )
    layer = pd.DataFrame(
        {
            "layer_index": range(40),
            "oracle_mean": np.arange(40, dtype=np.float64)[::-1],
        }
    )

    tables, summary = build_guarded_tables(
        timestep,
        layer,
        sparse_timestep_count=2,
        sparse_layer_count=3,
        dense_dit_prefix=2,
    )

    assert summary["selected_timesteps"] == [5, 7]
    assert summary["selected_layers"] == [37, 38, 39]
    assert summary["sparse_cell_count"] == 6
    history = np.asarray(tables["history_only"].history_keep_ratios)
    current = np.asarray(tables["history_only"].current_keep_ratios)
    assert np.all(history[:2] == 1.0)
    assert np.all(current == 1.0)
    assert np.all(history[np.ix_([5, 7], [37, 38, 39])] == 0.75)
    assert np.count_nonzero(history < 1.0) == 6


def test_joint_table_uses_same_guarded_cells_for_current_and_history() -> None:
    timestep = pd.DataFrame(
        {"dit_index": range(8), "oracle_mean": np.linspace(0.8, 0.1, 8)}
    )
    layer = pd.DataFrame(
        {"layer_index": range(40), "oracle_mean": np.linspace(0.1, 0.9, 40)}
    )

    tables, _ = build_guarded_tables(
        timestep,
        layer,
        sparse_timestep_count=3,
        sparse_layer_count=4,
    )

    history = np.asarray(tables["joint"].history_keep_ratios)
    current = np.asarray(tables["joint"].current_keep_ratios)
    assert np.array_equal(history, current)
    assert np.count_nonzero(current == 0.75) == 12
