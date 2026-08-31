"""Calibrate online flow-sentinel thresholds from paired validation replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TRACE_PREFIX = "Flow Sentinel Trace "


def parse_flow_sentinel_traces(path: Path) -> list[list[dict[str, Any]]]:
    traces = []
    for line in path.read_text(errors="replace").splitlines():
        position = line.find(TRACE_PREFIX)
        if position >= 0:
            traces.append(json.loads(line[position + len(TRACE_PREFIX) :]))
    return traces


def align_target_traces(
    report: dict[str, Any],
    traces: list[list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    records = [record for record in report["records"] if record["phase"] == "measured"]
    expected_calls = sum(1 + int(record["history_request_count"]) for record in records)
    if len(traces) < expected_calls:
        raise ValueError(
            f"server log contains {len(traces)} traces, expected at least {expected_calls}"
        )
    traces = traces[-expected_calls:]
    aligned = []
    cursor = 0
    for record in records:
        call_count = 1 + int(record["history_request_count"])
        request_traces = traces[cursor : cursor + call_count]
        if len(request_traces) != call_count:
            raise ValueError("flow-sentinel traces ended inside one request")
        target_trace = request_traces[-1]
        if len(target_trace) != 8:
            raise ValueError("each target must contain exactly eight real DiT records")
        if [int(item["dit_index"]) for item in target_trace] != list(range(8)):
            raise ValueError("target DiT trace is not ordered from 0 through 7")
        aligned.append((record, target_trace))
        cursor += call_count
    return aligned


def _trigger_matrix(
    cosine: np.ndarray,
    relative_l2: np.ndarray,
    *,
    minimum_cosine: float,
    maximum_relative_l2: float,
) -> np.ndarray:
    return (cosine < minimum_cosine) | (relative_l2 > maximum_relative_l2)


def choose_thresholds(
    cosine: np.ndarray,
    relative_l2: np.ndarray,
    quality_pass: np.ndarray,
) -> dict[str, Any]:
    """Minimize validation reruns while detecting every unsafe request."""

    if cosine.shape != relative_l2.shape or cosine.ndim != 2:
        raise ValueError("sentinel metrics must share shape [requests, checked_steps]")
    quality_pass = np.asarray(quality_pass, dtype=bool)
    if quality_pass.shape != (cosine.shape[0],):
        raise ValueError("quality labels must provide one value per request")
    cosine_candidates = np.unique(
        np.concatenate(([-1.0], cosine.ravel(), np.nextafter(cosine.ravel(), np.inf)))
    )
    l2_candidates = np.unique(
        np.concatenate((relative_l2.ravel(), np.nextafter(relative_l2.ravel(), -np.inf), [np.inf]))
    )
    feasible = []
    unsafe = ~quality_pass
    for minimum_cosine in cosine_candidates:
        for maximum_relative_l2 in l2_candidates:
            step_trigger = _trigger_matrix(
                cosine,
                relative_l2,
                minimum_cosine=float(minimum_cosine),
                maximum_relative_l2=float(maximum_relative_l2),
            )
            request_trigger = step_trigger.any(axis=1)
            false_negative = unsafe & ~request_trigger
            if false_negative.any():
                continue
            false_positive = quality_pass & request_trigger
            score = (
                int(step_trigger.sum()),
                int(request_trigger.sum()),
                int(false_positive.sum()),
                -float(maximum_relative_l2),
                float(minimum_cosine),
            )
            feasible.append(
                (
                    score,
                    float(minimum_cosine),
                    float(maximum_relative_l2),
                    step_trigger,
                    request_trigger,
                    false_positive,
                )
            )
    if not feasible:
        raise RuntimeError("no flow-sentinel threshold detects every unsafe request")
    (
        _,
        minimum_cosine,
        maximum_relative_l2,
        step_trigger,
        request_trigger,
        false_positive,
    ) = min(feasible, key=lambda item: item[0])
    return {
        "minimum_cosine": minimum_cosine,
        "maximum_relative_l2": maximum_relative_l2,
        "unsafe_request_count": int((~quality_pass).sum()),
        "detected_unsafe_request_count": int(((~quality_pass) & request_trigger).sum()),
        "safe_request_count": int(quality_pass.sum()),
        "false_positive_request_count": int(false_positive.sum()),
        "triggered_request_count": int(request_trigger.sum()),
        "triggered_step_count": int(step_trigger.sum()),
        "checked_step_count": int(step_trigger.size),
        "mean_dense_reruns_if_recomputed": float(step_trigger.sum(axis=1).mean()),
        "request_trigger": request_trigger.tolist(),
        "step_trigger": step_trigger.tolist(),
    }


def analyze(
    report: dict[str, Any],
    comparison: dict[str, Any],
    traces: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    aligned = align_target_traces(report, traces)
    if len(aligned) != int(comparison["paired_requests"]):
        raise ValueError("quality comparison and flow-sentinel trace counts differ")
    quality_cosine = np.asarray(
        comparison["action_cosine_per_request"], dtype=np.float64
    )
    quality_relative_l2 = np.asarray(
        comparison["action_relative_l2_per_request"], dtype=np.float64
    )
    quality_pass = (quality_cosine >= 0.999) & (quality_relative_l2 <= 0.05)
    checked_rows = []
    request_rows = []
    for request_index, (record, trace) in enumerate(aligned):
        checked = [item for item in trace if item.get("checked")]
        if len(checked) != 6:
            raise ValueError("flow sentinel must check DiT indices 2 through 7")
        checked_rows.append(checked)
        request_rows.append(
            {
                "request_key": record["request_key"],
                "trajectory_stage": record["trajectory_stage"],
                "quality_pass": bool(quality_pass[request_index]),
                "action_cosine": float(quality_cosine[request_index]),
                "action_relative_l2": float(quality_relative_l2[request_index]),
                "minimum_flow_cosine": float(
                    min(float(item["cosine"]) for item in checked)
                ),
                "maximum_flow_relative_l2": float(
                    max(float(item["relative_l2"]) for item in checked)
                ),
            }
        )
    flow_cosine = np.asarray(
        [[float(item["cosine"]) for item in checked] for checked in checked_rows]
    )
    flow_relative_l2 = np.asarray(
        [[float(item["relative_l2"]) for item in checked] for checked in checked_rows]
    )
    selected = choose_thresholds(flow_cosine, flow_relative_l2, quality_pass)
    for index, row in enumerate(request_rows):
        row["sentinel_triggered"] = bool(selected["request_trigger"][index])
        row["triggered_dit_indices"] = [
            int(checked_rows[index][step]["dit_index"])
            for step, triggered in enumerate(selected["step_trigger"][index])
            if triggered
        ]
    return {
        "paired_requests": len(aligned),
        "quality_pass_count": int(quality_pass.sum()),
        "quality_failure_count": int((~quality_pass).sum()),
        "selected_thresholds": selected,
        "requests": request_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.report.read_text()),
        json.loads(args.comparison.read_text()),
        parse_flow_sentinel_traces(args.server_log),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["selected_thresholds"], indent=2))


if __name__ == "__main__":
    main()
