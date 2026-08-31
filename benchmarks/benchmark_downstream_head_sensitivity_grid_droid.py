from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_downstream_head_sensitivity_droid import (
    action_sensitivity_metrics,
    intervention_control,
    run_chain_result,
    run_history,
    run_target,
    video_sensitivity_metrics,
)
from benchmarks.benchmark_dreamzero_server_droid import (
    DroidRequestReader,
    STAGE_TO_INSTRUCTION,
    build_request_plan,
)
from eval_utils.policy_client import WebsocketClientPolicy


VIDEO_METRIC_FIELDS = frozenset(
    {"video_cosine", "video_relative_l2", "video_max_abs"}
)


def validate_resume_video_schema(
    records: list[dict[str, Any]],
    *,
    record_video_sensitivity: bool,
) -> None:
    for record in records:
        present_video_fields = VIDEO_METRIC_FIELDS.intersection(record)
        if present_video_fields and present_video_fields != VIDEO_METRIC_FIELDS:
            raise ValueError("resume JSONL contains partial video metrics")
        if bool(present_video_fields) != record_video_sensitivity:
            raise ValueError(
                "resume JSONL video schema does not match "
                "--record-video-sensitivity"
            )


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate file must contain a non-empty candidate list")

    normalized = []
    labels = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("every downstream candidate must be a mapping")
        label = str(candidate["label"])
        if not label or label in labels:
            raise ValueError("candidate labels must be non-empty and unique")
        labels.add(label)
        control = intervention_control(
            dit_index=int(candidate["dit_index"]),
            layer_index=int(candidate["layer_index"]),
            head_indices=tuple(int(index) for index in candidate["head_indices"]),
            scale=float(candidate.get("scale", 0.0)),
            query_scope=str(candidate.get("query_scope", "all")),
        )
        normalized.append({"label": label, "control": control})
    return normalized


def summarize_candidate_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("candidate summary requires at least one record")
    worst = max(records, key=lambda record: record["action_relative_l2"])
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stage[record["trajectory_stage"]].append(record)
    summary = {
        "measured_requests": len(records),
        "action_cosine_mean": float(
            np.mean([record["action_cosine"] for record in records])
        ),
        "action_cosine_min": float(
            np.min([record["action_cosine"] for record in records])
        ),
        "action_relative_l2_mean": float(
            np.mean([record["action_relative_l2"] for record in records])
        ),
        "action_relative_l2_max": float(
            np.max([record["action_relative_l2"] for record in records])
        ),
        "action_max_abs_max": float(
            np.max([record["action_max_abs"] for record in records])
        ),
        "worst_request_key": worst["request_key"],
        "stage_relative_l2_mean": {
            stage: float(
                np.mean([record["action_relative_l2"] for record in stage_records])
            )
            for stage, stage_records in sorted(by_stage.items())
        },
    }
    record_video_sensitivity = bool(VIDEO_METRIC_FIELDS.intersection(records[0]))
    validate_resume_video_schema(
        records,
        record_video_sensitivity=record_video_sensitivity,
    )
    if record_video_sensitivity:
        video_worst = max(
            records,
            key=lambda record: record["video_relative_l2"],
        )
        summary.update(
            {
                "video_cosine_mean": float(
                    np.mean([record["video_cosine"] for record in records])
                ),
                "video_cosine_min": float(
                    np.min([record["video_cosine"] for record in records])
                ),
                "video_relative_l2_mean": float(
                    np.mean(
                        [record["video_relative_l2"] for record in records]
                    )
                ),
                "video_relative_l2_max": float(
                    np.max(
                        [record["video_relative_l2"] for record in records]
                    )
                ),
                "video_max_abs_max": float(
                    np.max([record["video_max_abs"] for record in records])
                ),
                "video_worst_request_key": video_worst["request_key"],
                "stage_video_relative_l2_mean": {
                    stage: float(
                        np.mean(
                            [
                                record["video_relative_l2"]
                                for record in stage_records
                            ]
                        )
                    )
                    for stage, stage_records in sorted(by_stage.items())
                },
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan multiple Dense attention interventions while sharing one "
            "same-noise baseline per real DROID request."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-history-snapshot", action="store_true")
    parser.add_argument("--record-video-sensitivity", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("validation",),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("early", "middle", "late"),
        default=("early", "middle", "late"),
    )
    parser.add_argument("--history-blocks", type=int, default=3)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")

    candidates = load_candidates(args.candidates)
    manifest = json.loads(args.manifest.read_text())
    plan = build_request_plan(
        manifest,
        splits=set(args.splits),
        stages=set(args.stages),
        max_requests=args.max_requests,
    )
    if not plan:
        raise ValueError("request plan is empty")

    jsonl_output = args.jsonl_output or args.output.with_suffix(".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if args.resume and jsonl_output.exists():
        records = [
            json.loads(line)
            for line in jsonl_output.read_text().splitlines()
            if line.strip()
        ]
    else:
        jsonl_output.write_text("")
    validate_resume_video_schema(
        records,
        record_video_sensitivity=args.record_video_sensitivity,
    )
    completed = {
        (record["request_key"], record["candidate_label"])
        for record in records
    }
    if len(completed) != len(records):
        raise ValueError("resume JSONL contains duplicate request/candidate rows")

    reader = DroidRequestReader(args.dataset_path)
    client = WebsocketClientPolicy(args.host, args.port)
    session_id = f"dreamzero-downstream-grid-{args.label}"
    client.reset({"session_id": session_id})
    try:
        for request_index, request in enumerate(plan):
            remaining_candidates = [
                candidate
                for candidate in candidates
                if (request["request_key"], candidate["label"]) not in completed
            ]
            if not remaining_candidates:
                continue
            baseline_observations = reader.observations(
                request,
                history_blocks=args.history_blocks,
                session_id=session_id,
            )
            if args.reuse_history_snapshot:
                baseline_history = run_history(
                    client,
                    baseline_observations[:-1],
                )
                client.snapshot({"request_key": request["request_key"]})
                baseline_result = run_target(
                    client,
                    baseline_observations[-1],
                    target_control=None,
                    return_video=args.record_video_sensitivity,
                )
            else:
                baseline_result, baseline_history = run_chain_result(
                    client,
                    baseline_observations,
                    target_control=None,
                    session_id=session_id,
                    return_video=args.record_video_sensitivity,
                )
            for candidate in remaining_candidates:
                if args.reuse_history_snapshot:
                    client.restore(
                        {
                            "request_key": request["request_key"],
                            "candidate_label": candidate["label"],
                        }
                    )
                    intervention_result = run_target(
                        client,
                        baseline_observations[-1],
                        target_control=candidate["control"],
                        return_video=args.record_video_sensitivity,
                    )
                    intervention_history = []
                else:
                    intervention_observations = reader.observations(
                        request,
                        history_blocks=args.history_blocks,
                        session_id=session_id,
                    )
                    intervention_result, intervention_history = run_chain_result(
                        client,
                        intervention_observations,
                        target_control=candidate["control"],
                        session_id=session_id,
                        return_video=args.record_video_sensitivity,
                    )
                baseline = baseline_result.action
                intervened = intervention_result.action
                video_metrics = {}
                video_shape = None
                if args.record_video_sensitivity:
                    if baseline_result.video is None:
                        raise RuntimeError("baseline video result is missing")
                    if intervention_result.video is None:
                        raise RuntimeError("intervention video result is missing")
                    video_shape = list(baseline_result.video.shape)
                    video_metrics = video_sensitivity_metrics(
                        baseline_result.video,
                        intervention_result.video,
                    )
                record = {
                    "request_index": request_index,
                    "request_key": request["request_key"],
                    "candidate_label": candidate["label"],
                    "split": request["split"],
                    "trajectory_stage": request["stage"]["name"],
                    "task": request["tasks"][
                        STAGE_TO_INSTRUCTION[request["stage"]["name"]]
                    ],
                    "intervention": candidate["control"],
                    "baseline_history_latency_seconds": baseline_history,
                    "intervention_history_latency_seconds": intervention_history,
                    "baseline_latency_seconds": baseline_result.latency_seconds,
                    "intervention_latency_seconds": (
                        intervention_result.latency_seconds
                    ),
                    "action_shape": list(baseline.shape),
                    "baseline_action": baseline.tolist(),
                    "intervention_action": intervened.tolist(),
                    **action_sensitivity_metrics(baseline, intervened),
                    **video_metrics,
                }
                if video_shape is not None:
                    record["video_shape"] = video_shape
                    record["baseline_downstream_trace"] = (
                        baseline_result.downstream_trace
                    )
                    record["intervention_downstream_trace"] = (
                        intervention_result.downstream_trace
                    )
                records.append(record)
                with jsonl_output.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in {
                                "baseline_action",
                                "intervention_action",
                                "baseline_history_latency_seconds",
                                "intervention_history_latency_seconds",
                            }
                        }
                    ),
                    flush=True,
                )
            if args.reuse_history_snapshot:
                client.restore({"request_key": request["request_key"]})
                client.reset({"session_id": session_id})
    finally:
        client._ws.close()

    candidate_summaries = {}
    for candidate in candidates:
        candidate_records = [
            record
            for record in records
            if record["candidate_label"] == candidate["label"]
        ]
        candidate_summaries[candidate["label"]] = {
            "intervention": candidate["control"],
            **summarize_candidate_records(candidate_records),
        }
    report = {
        "label": args.label,
        "seed": manifest["seed"],
        "dataset_path": str(args.dataset_path),
        "manifest": str(args.manifest),
        "candidate_file": str(args.candidates),
        "splits": args.splits,
        "stages": args.stages,
        "history_blocks": args.history_blocks,
        "reuse_history_snapshot": args.reuse_history_snapshot,
        "record_video_sensitivity": args.record_video_sensitivity,
        "target_latency_includes_video_transfer": (
            args.record_video_sensitivity
        ),
        "requests": len(plan),
        "candidates": len(candidates),
        "baseline_trajectory_count": len(plan),
        "intervention_trajectory_count": len(records),
        "candidate_summaries": candidate_summaries,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(candidate_summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
