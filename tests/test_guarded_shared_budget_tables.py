import numpy as np
import pandas as pd

from benchmarks.build_guarded_shared_budget_tables import (
    build_guarded_tables,
    lowest_sensitivity_segments,
)


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


def test_segment_selection_uses_worst_layer_risk_and_complete_segments() -> None:
    layer = pd.DataFrame(
        {
            "layer_index": range(40),
            "oracle_mean": np.full(40, 10.0),
        }
    )
    layer.loc[layer["layer_index"].between(6, 10), "oracle_mean"] = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    layer.loc[layer["layer_index"].between(16, 20), "oracle_mean"] = [
        0.2,
        0.2,
        0.2,
        0.2,
        0.6,
    ]

    segments = lowest_sensitivity_segments(
        layer,
        expected_count=40,
        selected_count=2,
        dense_prefix_layers=1,
        dense_suffix_layers=1,
        propagate_every=5,
    )

    assert segments == ((6, 7, 8, 9, 10), (16, 17, 18, 19, 20))


def test_guarded_segment_table_marks_only_complete_propagation_segments() -> None:
    timestep = pd.DataFrame(
        {"dit_index": range(8), "oracle_mean": np.linspace(0.8, 0.1, 8)}
    )
    layer = pd.DataFrame(
        {
            "layer_index": range(40),
            "oracle_mean": np.full(40, 10.0),
        }
    )
    layer.loc[layer["layer_index"].between(11, 15), "oracle_mean"] = 0.1

    tables, summary = build_guarded_tables(
        timestep,
        layer,
        sparse_timestep_count=2,
        sparse_segment_count=1,
        sparse_history_keep_ratio=0.75,
        sparse_current_keep_ratio=0.35,
    )

    assert summary["selection_mode"] == "propagation_segments"
    assert summary["selected_segments"] == [[11, 12, 13, 14, 15]]
    assert summary["selected_layers"] == [11, 12, 13, 14, 15]
    history = np.asarray(tables["joint"].history_keep_ratios)
    current = np.asarray(tables["joint"].current_keep_ratios)
    sparse_cells = np.ix_([6, 7], [11, 12, 13, 14, 15])
    assert np.all(history[sparse_cells] == 0.75)
    assert np.all(current[sparse_cells] == 0.35)
    assert np.count_nonzero(current < 1.0) == 10
