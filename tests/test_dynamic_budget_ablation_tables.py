import numpy as np
import pandas as pd

from benchmarks.build_dynamic_budget_ablation_tables import (
    build_tables,
    rank_assign,
)


def test_rank_assign_is_monotonic_in_sensitivity() -> None:
    scores = np.asarray([3.0, 1.0, 4.0, 2.0])
    assigned = rank_assign(
        scores,
        budgets=(0.20, 0.50),
        fractions=(0.50, 0.50),
    )

    assert assigned[2] >= assigned[0] >= assigned[3] >= assigned[1]


def test_ablation_tables_preserve_timestep_and_u_shaped_layer_order() -> None:
    timestep = pd.DataFrame(
        {"dit_index": range(8), "oracle_mean": np.linspace(0.70, 0.67, 8)}
    )
    layer_signal = np.concatenate(
        (np.linspace(0.8, 0.6, 20), np.linspace(0.6, 0.9, 20))
    )
    layer = pd.DataFrame(
        {"layer_index": range(40), "oracle_mean": layer_signal}
    )

    tables = build_tables(timestep, layer, profile="aggressive")

    timestep_only = np.asarray(tables["timestep_only"].history_keep_ratios)
    layer_only = np.asarray(tables["layer_only"].history_keep_ratios)
    joint = np.asarray(tables["timestep_layer"].history_keep_ratios)
    assert timestep_only[0, 0] > timestep_only[-1, 0]
    assert layer_only[0, 39] > layer_only[0, 15]
    assert joint.shape == (8, 40)
    assert set(np.unique(joint)).issubset({0.20, 0.35, 0.50, 0.75})
