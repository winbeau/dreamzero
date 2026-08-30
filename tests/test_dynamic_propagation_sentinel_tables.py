import numpy as np

from benchmarks.build_dynamic_propagation_sentinel_tables import (
    build_sentinel_table,
    propagation_boundary_layers,
)
from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedBudgetTable,
)


def test_propagation_boundaries_include_regular_and_final_segments() -> None:
    assert propagation_boundary_layers(
        num_layers=40,
        dense_prefix_layers=1,
        dense_suffix_layers=1,
        propagate_every=5,
    ) == (5, 10, 15, 20, 25, 30, 35, 38)


def test_sentinel_promotes_only_boundary_current_budgets() -> None:
    base = DynamicPackedBudgetTable.constant(
        num_dit_steps=2,
        num_layers=12,
        history_keep_ratio=0.20,
        current_keep_ratio=0.35,
        name="base",
    )
    table = build_sentinel_table(
        base,
        sentinel_current_keep_ratio=0.75,
        dense_prefix_layers=1,
        dense_suffix_layers=1,
        propagate_every=4,
    )

    history = np.asarray(table.history_keep_ratios)
    current = np.asarray(table.current_keep_ratios)
    assert np.array_equal(history, np.full((2, 12), 0.20))
    assert np.array_equal(current[:, (4, 8, 10)], np.full((2, 3), 0.75))
    non_boundary = (0, 1, 2, 3, 5, 6, 7, 9, 11)
    assert np.array_equal(current[:, non_boundary], np.full((2, 9), 0.35))


def test_sentinel_never_reduces_a_more_conservative_base_budget() -> None:
    base = DynamicPackedBudgetTable(
        history_keep_ratios=((0.20, 0.20, 0.20),),
        current_keep_ratios=((0.35, 1.00, 0.50),),
    )
    table = build_sentinel_table(
        base,
        sentinel_current_keep_ratio=0.75,
        dense_prefix_layers=0,
        dense_suffix_layers=0,
        propagate_every=2,
    )

    assert table.current_keep_ratios == ((0.35, 1.00, 0.75),)
