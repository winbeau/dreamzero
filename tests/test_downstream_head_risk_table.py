from copy import deepcopy

import numpy as np
import pytest

from benchmarks.benchmark_downstream_head_sensitivity_droid import (
    intervention_control,
)
from benchmarks.build_downstream_head_risk_table import (
    build_downstream_head_risk_table,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
)


def _record(
    *,
    request_key: str,
    stage: str,
    heads: tuple[int, ...],
    safe: bool,
) -> dict:
    control = intervention_control(
        dit_index=0,
        layer_index=0,
        head_indices=heads,
        scale=0.0,
        query_scope="all",
    )
    return {
        "request_key": request_key,
        "candidate_label": f"heads-{heads[0]}-{heads[-1]}",
        "split": "validation",
        "trajectory_stage": stage,
        "intervention": control,
        "action_cosine": 0.9995 if safe else 0.998,
        "action_relative_l2": 0.02 if safe else 0.08,
        "video_cosine": 0.995 if safe else 0.90,
        "video_relative_l2": 0.04 if safe else 0.30,
        "baseline_downstream_trace": {
            "configured": False,
            "applied_count": 0,
        },
        "intervention_downstream_trace": {
            "configured": True,
            "dit_index": 0,
            "layer_index": 0,
            "head_indices": list(heads),
            "scale": 0.0,
            "cfg_branches": ["conditional"],
            "query_scope": "all",
            "applied_count": 1,
        },
    }


def _records() -> list[dict]:
    records = []
    for index, stage in enumerate(("early", "middle", "late")):
        records.append(
            _record(
                request_key=f"safe-{index}",
                stage=stage,
                heads=(0, 1),
                safe=True,
            )
        )
        records.append(
            _record(
                request_key=f"unsafe-{index}",
                stage=stage,
                heads=(2, 3),
                safe=index < 2,
            )
        )
    return records


def test_downstream_group_evidence_builds_safe_unsafe_and_unknown_masks() -> None:
    table, cells = build_downstream_head_risk_table(
        _records(),
        num_dit_steps=2,
        num_layers=2,
        num_heads=4,
        min_unique_requests=3,
        video_cosine_min=0.99,
        video_relative_l2_max=0.10,
    )

    assert table.scanned[0, 0].tolist() == [True, True, True, True]
    assert table.safe[0, 0].tolist() == [True, True, False, False]
    assert not np.any(table.scanned[0, 1])
    assert not np.any(table.scanned[1])
    assert len(cells) == 4
    assert cells[0]["unique_requests"] == 3
    assert cells[0]["trajectory_stages"] == ["early", "late", "middle"]
    assert cells[0]["safe"] is True
    assert cells[-1]["failed_rows"] == 1
    assert cells[-1]["safe"] is False
    assert table.metadata["scanned_head_fraction"] == pytest.approx(0.25)
    assert table.metadata["safe_head_fraction"] == pytest.approx(0.125)

    restored = DownstreamHeadRiskTable.from_dict(table.to_dict())
    np.testing.assert_array_equal(restored.scanned, table.scanned)
    np.testing.assert_array_equal(restored.safe, table.safe)


def test_downstream_risk_coverage_requires_requests_stages_and_split() -> None:
    records = _records()[:2]
    table, _ = build_downstream_head_risk_table(
        records,
        num_dit_steps=1,
        num_layers=1,
        num_heads=4,
        min_unique_requests=1,
        video_cosine_min=0.99,
        video_relative_l2_max=0.10,
    )

    assert not np.any(table.scanned)
    assert not np.any(table.safe)


def test_downstream_risk_rejects_nonremoval_or_untraced_evidence() -> None:
    scale_one = deepcopy(_records())
    scale_one[0]["intervention"]["scale"] = 1.0
    scale_one[0]["intervention_downstream_trace"]["scale"] = 1.0
    with pytest.raises(ValueError, match="scale-zero"):
        build_downstream_head_risk_table(
            scale_one,
            num_dit_steps=2,
            num_layers=2,
            num_heads=4,
            min_unique_requests=3,
            video_cosine_min=0.99,
            video_relative_l2_max=0.10,
        )

    untraced = deepcopy(_records())
    del untraced[0]["intervention_downstream_trace"]
    with pytest.raises(TypeError, match="trace must be a mapping"):
        build_downstream_head_risk_table(
            untraced,
            num_dit_steps=2,
            num_layers=2,
            num_heads=4,
            min_unique_requests=3,
            video_cosine_min=0.99,
            video_relative_l2_max=0.10,
        )
