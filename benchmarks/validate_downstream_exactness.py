from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_downstream_head_sensitivity_droid import (
    validate_downstream_trace,
)


def _require_exact_metric(record: dict[str, Any], prefix: str) -> None:
    cosine = float(record[f"{prefix}_cosine"])
    relative_l2 = float(record[f"{prefix}_relative_l2"])
    max_abs = float(record[f"{prefix}_max_abs"])
    if not all(math.isfinite(value) for value in (cosine, relative_l2, max_abs)):
        raise ValueError(f"{prefix} exactness metrics must be finite")
    if relative_l2 != 0.0 or max_abs != 0.0:
        raise ValueError(
            f"{prefix} output is not exact: relative_l2={relative_l2}, "
            f"max_abs={max_abs}"
        )
    if not math.isclose(cosine, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{prefix} exactness cosine is {cosine}, expected 1")


def validate_exactness_report(
    report: dict[str, Any],
    *,
    expected_records: int | None = None,
) -> dict[str, Any]:
    if report.get("record_video_sensitivity") is not True:
        raise ValueError("exactness report must record final-video sensitivity")
    if report.get("reuse_history_snapshot") is not True:
        raise ValueError("exactness report must reuse the Dense-history snapshot")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("exactness report contains no records")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(
            f"exactness report has {len(records)} records, "
            f"expected {expected_records}"
        )

    pair_keys = set()
    request_keys = set()
    candidate_labels = set()
    for record in records:
        control = record["intervention"]
        if float(control.get("scale", 0.0)) != 1.0:
            raise ValueError("exactness candidates must use intervention scale=1")
        pair_key = (record["request_key"], record["candidate_label"])
        if pair_key in pair_keys:
            raise ValueError(f"duplicate exactness pair: {pair_key}")
        pair_keys.add(pair_key)
        request_keys.add(record["request_key"])
        candidate_labels.add(record["candidate_label"])

        baseline_action = np.asarray(record["baseline_action"])
        intervention_action = np.asarray(record["intervention_action"])
        if baseline_action.shape != tuple(record["action_shape"]):
            raise ValueError("baseline action shape does not match action_shape")
        if not np.array_equal(baseline_action, intervention_action):
            raise ValueError("scale-one action arrays are not elementwise exact")
        if "video_shape" not in record or not record["video_shape"]:
            raise ValueError("exactness record is missing final-video shape")

        _require_exact_metric(record, "action")
        _require_exact_metric(record, "video")
        validate_downstream_trace(
            record["baseline_downstream_trace"],
            expected_control=None,
        )
        validate_downstream_trace(
            record["intervention_downstream_trace"],
            expected_control=control,
        )

    return {
        "exact": True,
        "records": len(records),
        "requests": len(request_keys),
        "candidates": len(candidate_labels),
        "action_elementwise_exact": True,
        "video_difference_exact": True,
        "intervention_applied_once": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail unless a scale-one downstream grid preserves final action "
            "and video exactly after Dense-history snapshot restore."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_records is not None and args.expected_records <= 0:
        parser.error("--expected-records must be positive")

    report = json.loads(args.report.read_text())
    summary = validate_exactness_report(
        report,
        expected_records=args.expected_records,
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
