import pandas as pd
import pytest

from benchmarks.build_dynamic_two_group_qkv_table import (
    build_two_group_qkv_table,
)


def _prior() -> pd.DataFrame:
    rows = []
    for dit_index in range(2):
        for layer_index in range(2):
            for head_index, mean in enumerate((0.95, 0.79, 0.80, 0.20)):
                rows.append(
                    {
                        "dit_index": dit_index,
                        "layer_index": layer_index,
                        "head_index": head_index,
                        "prior_budget_mean_tlh": mean,
                    }
                )
    return pd.DataFrame(rows)


def test_build_two_group_qkv_table_assigns_critical_and_normal_shapes():
    table = build_two_group_qkv_table(
        _prior(),
        num_dit_steps=2,
        num_layers=2,
        num_heads=4,
    )

    assert table.num_groups == 2
    assert table.execution_groups_for_layer(1, 1) == (
        ((0, 2), 1.0, 0.75),
        ((1, 3), 0.35, 0.35),
    )


def test_build_two_group_qkv_table_rejects_bad_threshold_and_grid():
    with pytest.raises(ValueError, match="critical_threshold"):
        build_two_group_qkv_table(
            _prior(),
            critical_threshold=1.1,
            num_dit_steps=2,
            num_layers=2,
            num_heads=4,
        )

    with pytest.raises(ValueError, match="dense grid"):
        build_two_group_qkv_table(
            _prior().iloc[:-1],
            num_dit_steps=2,
            num_layers=2,
            num_heads=4,
        )
