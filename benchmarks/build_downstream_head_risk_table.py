"""Build conservative M1 fallback masks from task-disjoint downstream scans."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_downstream_head_sensitivity_droid import (
    validate_downstream_trace,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_group_router import (
    DownstreamHeadRiskTable,
)

METRIC_NAMES = (
    "action_cosine",
    "action_relative_l2",
    "video_cosine",
    "video_relative_l2",
)


def _validated_metric(record: dict[str, Any], name: str) -> float:
    if name not in record:
        raise ValueError(f"downstream record is missing {name}")
    value = float(record[name])
    if not math.isfinite(value):
        raise ValueError(f"downstream record has non-finite {name}")
    return value


def build_downstream_head_risk_table(
    records: list[dict[str, Any]],
    *,
    num_dit_steps: int = 8,
    num_layers: int = 40,
    num_heads: int = 40,
    required_splits: tuple[str, ...] = ("validation",),
    required_stages: tuple[str, ...] = ("early", "middle", "late"),
    min_unique_requests: int = 18,
    action_cosine_min: float = 0.999,
    action_relative_l2_max: float = 0.05,
    video_cosine_min: float,
    video_relative_l2_max: float,
    require_trace: bool = True,
) -> tuple[DownstreamHeadRiskTable, list[dict[str, Any]]]:
    """Aggregate group-removal evidence into conservative per-head masks.

    A scale-zero shared-group intervention is a conservative criticality probe:
    failure marks every head in that group unsafe. Passing evidence is accepted
    only after the requested split, trajectory-stage, and unique-request
    coverage gates all pass. Everything else remains unscanned and therefore
    routes Dense in ``DynamicM1GroupedRouter``.
    """

    if not records:
        raise ValueError("downstream risk table requires at least one record")
    if min(num_dit_steps, num_layers, num_heads, min_unique_requests) <= 0:
        raise ValueError("downstream risk dimensions and coverage must be positive")
    if not required_splits or len(set(required_splits)) != len(required_splits):
        raise ValueError("required_splits must be non-empty and unique")
    if not required_stages or len(set(required_stages)) != len(required_stages):
        raise ValueError("required_stages must be non-empty and unique")
    thresholds = (
        action_cosine_min,
        action_relative_l2_max,
        video_cosine_min,
        video_relative_l2_max,
    )
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("downstream safety thresholds must be finite")
    if not -1.0 <= action_cosine_min <= 1.0:
        raise ValueError("action_cosine_min must lie in [-1, 1]")
    if not -1.0 <= video_cosine_min <= 1.0:
        raise ValueError("video_cosine_min must lie in [-1, 1]")
    if action_relative_l2_max < 0.0 or video_relative_l2_max < 0.0:
        raise ValueError("relative-L2 limits must be non-negative")

    evidence: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    selected_records = 0
    pair_keys = set()
    for record in records:
        if str(record.get("split")) not in required_splits:
            continue
        selected_records += 1
        pair_key = (str(record["request_key"]), str(record["candidate_label"]))
        if pair_key in pair_keys:
            raise ValueError(f"duplicate downstream request/candidate row: {pair_key}")
        pair_keys.add(pair_key)
        intervention = record.get("intervention")
        if not isinstance(intervention, dict):
            raise TypeError("downstream record intervention must be a mapping")
        if float(intervention.get("scale", 0.0)) != 0.0:
            raise ValueError("downstream risk evidence must use scale-zero removal")
        dit_index = int(intervention["dit_index"])
        layer_index = int(intervention["layer_index"])
        head_indices = tuple(int(index) for index in intervention["head_indices"])
        if not 0 <= dit_index < num_dit_steps:
            raise ValueError("downstream evidence DiT index is outside the grid")
        if not 0 <= layer_index < num_layers:
            raise ValueError("downstream evidence layer index is outside the grid")
        if not head_indices or len(set(head_indices)) != len(head_indices):
            raise ValueError("downstream evidence heads must be non-empty and unique")
        if any(not 0 <= index < num_heads for index in head_indices):
            raise ValueError("downstream evidence head index is outside the grid")
        stage = str(record.get("trajectory_stage"))
        if stage not in required_stages:
            raise ValueError(f"unexpected downstream trajectory stage: {stage}")
        metrics = {name: _validated_metric(record, name) for name in METRIC_NAMES}
        if require_trace:
            validate_downstream_trace(
                record.get("baseline_downstream_trace"),
                expected_control=None,
            )
            validate_downstream_trace(
                record.get("intervention_downstream_trace"),
                expected_control=intervention,
            )
        safe = bool(
            metrics["action_cosine"] >= action_cosine_min
            and metrics["action_relative_l2"] <= action_relative_l2_max
            and metrics["video_cosine"] >= video_cosine_min
            and metrics["video_relative_l2"] <= video_relative_l2_max
        )
        normalized = {
            "request_key": str(record["request_key"]),
            "split": str(record["split"]),
            "trajectory_stage": stage,
            "candidate_label": str(record["candidate_label"]),
            "safe": safe,
            **metrics,
        }
        for head_index in head_indices:
            evidence[(dit_index, layer_index, head_index)].append(normalized)

    if selected_records == 0:
        raise ValueError("no downstream records match required_splits")

    scanned = np.zeros((num_dit_steps, num_layers, num_heads), dtype=bool)
    safe = np.zeros_like(scanned)
    cell_records = []
    for (dit_index, layer_index, head_index), head_records in sorted(evidence.items()):
        request_keys = {record["request_key"] for record in head_records}
        splits = {record["split"] for record in head_records}
        stages = {record["trajectory_stage"] for record in head_records}
        covered = bool(
            len(request_keys) >= min_unique_requests
            and set(required_splits).issubset(splits)
            and set(required_stages).issubset(stages)
        )
        cell_safe = bool(covered and all(record["safe"] for record in head_records))
        scanned[dit_index, layer_index, head_index] = covered
        safe[dit_index, layer_index, head_index] = cell_safe
        cell_records.append(
            {
                "dit_index": dit_index,
                "layer_index": layer_index,
                "head_index": head_index,
                "evidence_rows": len(head_records),
                "unique_requests": len(request_keys),
                "splits": sorted(splits),
                "trajectory_stages": sorted(stages),
                "scanned": covered,
                "safe": cell_safe,
                "action_cosine_min": float(
                    min(record["action_cosine"] for record in head_records)
                ),
                "action_relative_l2_max": float(
                    max(record["action_relative_l2"] for record in head_records)
                ),
                "video_cosine_min": float(
                    min(record["video_cosine"] for record in head_records)
                ),
                "video_relative_l2_max": float(
                    max(record["video_relative_l2"] for record in head_records)
                ),
                "failed_rows": int(
                    sum(not record["safe"] for record in head_records)
                ),
            }
        )

    metadata = {
        "evidence_type": "scale-zero shared-head-group removal proxy",
        "selected_record_rows": selected_records,
        "required_splits": list(required_splits),
        "required_stages": list(required_stages),
        "min_unique_requests": min_unique_requests,
        "require_trace": require_trace,
        "thresholds": {
            "action_cosine_min": action_cosine_min,
            "action_relative_l2_max": action_relative_l2_max,
            "video_cosine_min": video_cosine_min,
            "video_relative_l2_max": video_relative_l2_max,
        },
        "scanned_head_fraction": float(np.mean(scanned)),
        "safe_head_fraction": float(np.mean(safe)),
        "unsafe_scanned_head_fraction": float(np.mean(scanned & ~safe)),
        "unknown_head_fraction": float(np.mean(~scanned)),
    }
    return DownstreamHeadRiskTable(scanned, safe, metadata), cell_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-output", type=Path)
    parser.add_argument("--required-splits", nargs="+", default=["validation"])
    parser.add_argument(
        "--required-stages",
        nargs="+",
        default=["early", "middle", "late"],
    )
    parser.add_argument("--min-unique-requests", type=int, default=18)
    parser.add_argument("--action-cosine-min", type=float, default=0.999)
    parser.add_argument("--action-relative-l2-max", type=float, default=0.05)
    parser.add_argument("--video-cosine-min", type=float, required=True)
    parser.add_argument("--video-relative-l2-max", type=float, required=True)
    parser.add_argument(
        "--require-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.records.read_text().splitlines()
        if line.strip()
    ]
    table, cells = build_downstream_head_risk_table(
        records,
        required_splits=tuple(args.required_splits),
        required_stages=tuple(args.required_stages),
        min_unique_requests=args.min_unique_requests,
        action_cosine_min=args.action_cosine_min,
        action_relative_l2_max=args.action_relative_l2_max,
        video_cosine_min=args.video_cosine_min,
        video_relative_l2_max=args.video_relative_l2_max,
        require_trace=args.require_trace,
    )
    payload = table.to_dict()
    payload["metadata"]["records"] = str(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    cell_output = args.cell_output or args.output.with_suffix(".cells.jsonl")
    cell_output.parent.mkdir(parents=True, exist_ok=True)
    cell_output.write_text("".join(json.dumps(cell) + "\n" for cell in cells))
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
