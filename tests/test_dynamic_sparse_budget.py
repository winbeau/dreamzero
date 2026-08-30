import json

import pytest

from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedHeadGroupBudgetTable,
    DynamicPackedBudgetTable,
    bucket_at_least,
)


def test_budget_table_indexes_fixed_dit_layer_shapes() -> None:
    table = DynamicPackedBudgetTable(
        history_keep_ratios=((0.75, 0.50, 1.0), (0.50, 0.35, 0.75)),
        current_keep_ratios=((0.50, 0.35, 1.0), (0.35, 0.20, 0.75)),
        name="u-shaped",
    )

    assert table.num_dit_steps == 2
    assert table.num_layers == 3
    assert table.ratios(1, 1) == (0.35, 0.20)
    assert table.maximum_current_ratio(0, range(3)) == 1.0
    assert table.history_ratios(1, (0, 1, 2)) == (0.35, 0.50, 0.75)


def test_constant_table_and_json_payload_round_trip() -> None:
    table = DynamicPackedBudgetTable.constant(
        num_dit_steps=8,
        num_layers=40,
        history_keep_ratio=0.20,
        current_keep_ratio=0.25,
    )
    restored = DynamicPackedBudgetTable.from_dict(table.to_dict())

    assert restored.history_keep_ratios == table.history_keep_ratios
    assert restored.current_keep_ratios == table.current_keep_ratios


def test_continuous_oracle_budget_quantizes_upward() -> None:
    assert bucket_at_least(0.11) == 0.20
    assert bucket_at_least(0.50) == 0.50
    assert bucket_at_least(0.91) == 1.0


def test_arbitrary_dynamic_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="fixed buckets"):
        DynamicPackedBudgetTable(
            history_keep_ratios=((0.3,),),
            current_keep_ratios=((0.2,),),
        )


def test_dynamic_head_group_budget_roundtrip_and_partition_validation(tmp_path):
    table = DynamicPackedHeadGroupBudgetTable(
        head_groups=((0, 2), (1, 3)),
        group_names=("critical", "normal"),
        history_keep_ratios=tuple(
            tuple((1.0, 0.35) for _ in range(3)) for _ in range(2)
        ),
        name="tiny",
    )

    assert table.num_dit_steps == 2
    assert table.num_layers == 3
    assert table.num_groups == 2
    assert table.num_heads == 4
    assert table.groups_for_layer(1, 2) == (
        ((0, 2), 1.0),
        ((1, 3), 0.35),
    )

    path = tmp_path / "head_groups.json"
    path.write_text(json.dumps(table.to_dict()))
    assert DynamicPackedHeadGroupBudgetTable.from_json(path) == table

    with pytest.raises(ValueError, match="must not overlap"):
        DynamicPackedHeadGroupBudgetTable(
            head_groups=((0, 1), (1, 2)),
            history_keep_ratios=(((1.0, 0.5),),),
        )

    with pytest.raises(ValueError, match="group count"):
        DynamicPackedHeadGroupBudgetTable(
            head_groups=((0,), (1,)),
            history_keep_ratios=(((1.0,),),),
        )
