import itertools

import pandas as pd
import pytest

from benchmarks.build_dynamic_head_group_budget_table import (
    build_head_group_table,
)


def _prior() -> pd.DataFrame:
    rows = []
    values = (0.20, 0.40, 0.70, 0.90)
    for dit_index, layer_index, head_index in itertools.product(
        range(2), range(3), range(4)
    ):
        rows.append(
            {
                "dit_index": dit_index,
                "layer_index": layer_index,
                "head_index": head_index,
                "prior_budget_mean_tlh": values[head_index],
            }
        )
    return pd.DataFrame(rows)


def test_build_head_group_table_quantizes_up_and_groups_by_cell():
    table = build_head_group_table(
        _prior(),
        num_dit_steps=2,
        num_layers=3,
        num_heads=4,
    )

    assert table.ratios(0, 0) == (0.25, 0.50, 0.75, 1.0)
    assert table.groups_for_layer(1, 2) == (
        ((3,), 1.0),
        ((2,), 0.75),
        ((1,), 0.50),
        ((0,), 0.25),
    )


def test_build_head_group_table_rejects_incomplete_or_duplicate_grid():
    prior = _prior()
    with pytest.raises(ValueError, match="dense grid"):
        build_head_group_table(
            prior.iloc[:-1],
            num_dit_steps=2,
            num_layers=3,
            num_heads=4,
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_head_group_table(
            pd.concat((prior, prior.iloc[[0]]), ignore_index=True),
            num_dit_steps=2,
            num_layers=3,
            num_heads=4,
        )
