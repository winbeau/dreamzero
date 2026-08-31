from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_dreamzero_server_droid import (
    DroidRequestReader,
    STAGE_TO_INSTRUCTION,
    build_request_plan,
)
from eval_utils.policy_client import WebsocketClientPolicy


CONTROL_KEY = "dynamic_downstream_head_intervention"
RETURN_VIDEO_KEY = "dynamic_downstream_return_video"


@dataclass(frozen=True)
class TargetResult:
    action: np.ndarray
    latency_seconds: float
    video: np.ndarray | None = None
    downstream_trace: dict[str, Any] | None = None


def _paired_sensitivity_metrics(
    baseline: np.ndarray,
    intervened: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    intervened_array = np.asarray(intervened, dtype=np.float64)
    if baseline_array.shape != intervened_array.shape:
        raise ValueError(f"paired {prefix} shapes do not match")
    if baseline_array.size == 0:
        raise ValueError(f"paired {prefix} arrays must be non-empty")
    if not np.all(np.isfinite(baseline_array)) or not np.all(
        np.isfinite(intervened_array)
    ):
        raise ValueError(f"paired {prefix} arrays must be finite")
    baseline_flat = baseline_array.reshape(-1)
    intervened_flat = intervened_array.reshape(-1)
    baseline_norm = float(np.linalg.norm(baseline_flat))
    intervened_norm = float(np.linalg.norm(intervened_flat))
    denominator = max(baseline_norm * intervened_norm, 1e-12)
    difference = intervened_flat - baseline_flat
    return {
        f"{prefix}_cosine": float(
            np.dot(baseline_flat, intervened_flat) / denominator
        ),
        f"{prefix}_relative_l2": float(
            np.linalg.norm(difference) / max(baseline_norm, 1e-12)
        ),
        f"{prefix}_max_abs": float(np.max(np.abs(difference))),
    }


def action_sensitivity_metrics(
    baseline: np.ndarray,
    intervened: np.ndarray,
) -> dict[str, float]:
    return _paired_sensitivity_metrics(
        baseline,
        intervened,
        prefix="action",
    )


def video_sensitivity_metrics(
    baseline: np.ndarray,
    intervened: np.ndarray,
) -> dict[str, float]:
    return _paired_sensitivity_metrics(
        baseline,
        intervened,
        prefix="video",
    )


def intervention_control(
    *,
    dit_index: int,
    layer_index: int,
    head_indices: tuple[int, ...],
    scale: float,
    query_scope: str,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "dit_index": dit_index,
        "layer_index": layer_index,
        "head_indices": list(head_indices),
        "scale": scale,
        "cfg_branches": ["conditional"],
        "query_scope": query_scope,
    }


def validate_downstream_trace(
    trace: dict[str, Any],
    *,
    expected_control: dict[str, Any] | None,
) -> None:
    if not isinstance(trace, dict):
        raise TypeError("downstream intervention trace must be a mapping")
    if expected_control is None:
        if trace.get("configured") is not False:
            raise ValueError("Dense baseline unexpectedly configured an intervention")
        if int(trace.get("applied_count", -1)) != 0:
            raise ValueError("Dense baseline unexpectedly applied an intervention")
        return

    expected = {
        "configured": True,
        "dit_index": int(expected_control["dit_index"]),
        "layer_index": int(expected_control["layer_index"]),
        "head_indices": [int(index) for index in expected_control["head_indices"]],
        "scale": float(expected_control.get("scale", 0.0)),
        "cfg_branches": list(expected_control.get("cfg_branches", ["conditional"])),
        "query_scope": str(expected_control.get("query_scope", "all")),
        "applied_count": 1,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": trace.get(key)}
        for key, expected_value in expected.items()
        if trace.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"downstream intervention trace mismatch: {mismatches}")


def run_chain(
    client: WebsocketClientPolicy,
    observations: list[dict[str, Any]],
    *,
    target_control: dict[str, Any] | None,
    session_id: str,
) -> tuple[np.ndarray, list[float], float]:
    result, history_latencies = run_chain_result(
        client,
        observations,
        target_control=target_control,
        session_id=session_id,
    )
    return result.action, history_latencies, result.latency_seconds


def run_chain_result(
    client: WebsocketClientPolicy,
    observations: list[dict[str, Any]],
    *,
    target_control: dict[str, Any] | None,
    session_id: str,
    return_video: bool = False,
) -> tuple[TargetResult, list[float]]:
    history_latencies = run_history(client, observations[:-1])
    result = run_target(
        client,
        observations[-1],
        target_control=target_control,
        return_video=return_video,
    )
    client.reset({"session_id": session_id})
    return result, history_latencies


def run_history(
    client: WebsocketClientPolicy,
    observations: list[dict[str, Any]],
) -> list[float]:
    history_latencies = []
    for observation in observations:
        observation[CONTROL_KEY] = {"enabled": False}
        started = time.perf_counter()
        client.infer(observation)
        history_latencies.append(time.perf_counter() - started)
    return history_latencies


def run_target(
    client: WebsocketClientPolicy,
    observation: dict[str, Any],
    *,
    target_control: dict[str, Any] | None,
    return_video: bool = False,
) -> TargetResult:
    target = dict(observation)
    target[CONTROL_KEY] = (
        {"enabled": False} if target_control is None else target_control
    )
    if return_video:
        target[RETURN_VIDEO_KEY] = True
    started = time.perf_counter()
    response = client.infer(target)
    target_latency = time.perf_counter() - started
    if return_video:
        if not isinstance(response, dict):
            raise TypeError("video-return request expected a mapping response")
        expected_response_fields = {"action", "video", "downstream_trace"}
        if set(response) != expected_response_fields:
            raise ValueError(
                "video-return response must contain exactly action, video, "
                "and downstream_trace"
            )
        downstream_trace = response["downstream_trace"]
        if not isinstance(downstream_trace, dict):
            raise TypeError("downstream intervention trace must be a mapping")
        validate_downstream_trace(
            downstream_trace,
            expected_control=target_control,
        )
        return TargetResult(
            action=np.asarray(response["action"]),
            latency_seconds=target_latency,
            video=np.asarray(response["video"]),
            downstream_trace=downstream_trace,
        )
    return TargetResult(
        action=np.asarray(response),
        latency_seconds=target_latency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure final-action sensitivity of one Dense attention head group "
            "with same-process, same-noise paired DROID trajectories."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dit-index", type=int, required=True)
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--head-indices", type=int, nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument(
        "--query-scope", choices=("all", "video", "register"), default="all"
    )
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
    parser.add_argument("--warmup-pairs", type=int, default=0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--record-video-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.warmup_pairs < 0:
        parser.error("--warmup-pairs must be non-negative")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")

    manifest = json.loads(args.manifest.read_text())
    plan = build_request_plan(
        manifest,
        splits=set(args.splits),
        stages=set(args.stages),
        max_requests=args.max_requests,
    )
    if len(plan) <= args.warmup_pairs:
        raise ValueError("request plan contains no measured pairs")

    reader = DroidRequestReader(args.dataset_path)
    client = WebsocketClientPolicy(args.host, args.port)
    control = intervention_control(
        dit_index=args.dit_index,
        layer_index=args.layer_index,
        head_indices=tuple(args.head_indices),
        scale=args.scale,
        query_scope=args.query_scope,
    )
    session_id = f"dreamzero-downstream-{args.label}"
    records = []
    try:
        for request_index, request in enumerate(plan):
            baseline_observations = reader.observations(
                request,
                history_blocks=args.history_blocks,
                session_id=session_id,
            )
            baseline_result, baseline_history = run_chain_result(
                client,
                baseline_observations,
                target_control=None,
                session_id=session_id,
                return_video=args.record_video_sensitivity,
            )
            intervention_observations = reader.observations(
                request,
                history_blocks=args.history_blocks,
                session_id=session_id,
            )
            intervention_result, intervention_history = run_chain_result(
                client,
                intervention_observations,
                target_control=control,
                session_id=session_id,
                return_video=args.record_video_sensitivity,
            )
            baseline = baseline_result.action
            intervened = intervention_result.action
            metrics = action_sensitivity_metrics(baseline, intervened)
            if args.record_video_sensitivity:
                if baseline_result.video is None:
                    raise RuntimeError("baseline video result is missing")
                if intervention_result.video is None:
                    raise RuntimeError("intervention video result is missing")
                metrics.update(
                    video_sensitivity_metrics(
                        baseline_result.video,
                        intervention_result.video,
                    )
                )
            phase = "warmup" if request_index < args.warmup_pairs else "measured"
            record = {
                "request_index": request_index,
                "request_key": request["request_key"],
                "phase": phase,
                "split": request["split"],
                "trajectory_stage": request["stage"]["name"],
                "task": request["tasks"][
                    STAGE_TO_INSTRUCTION[request["stage"]["name"]]
                ],
                "baseline_history_latency_seconds": baseline_history,
                "intervention_history_latency_seconds": intervention_history,
                "baseline_latency_seconds": baseline_result.latency_seconds,
                "intervention_latency_seconds": (
                    intervention_result.latency_seconds
                ),
                "action_shape": list(baseline.shape),
                "baseline_action": baseline.tolist(),
                "intervention_action": intervened.tolist(),
                **metrics,
            }
            if baseline_result.video is not None:
                record["video_shape"] = list(baseline_result.video.shape)
                record["baseline_downstream_trace"] = (
                    baseline_result.downstream_trace
                )
                record["intervention_downstream_trace"] = (
                    intervention_result.downstream_trace
                )
            records.append(record)
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
    finally:
        client._ws.close()

    measured = [record for record in records if record["phase"] == "measured"]
    report = {
        "label": args.label,
        "seed": manifest["seed"],
        "dataset_path": str(args.dataset_path),
        "manifest": str(args.manifest),
        "history_blocks": args.history_blocks,
        "record_video_sensitivity": args.record_video_sensitivity,
        "target_latency_includes_video_transfer": (
            args.record_video_sensitivity
        ),
        "warmup_pairs": args.warmup_pairs,
        "measured_pairs": len(measured),
        "intervention": control,
        "summary": {
            "action_cosine_mean": float(
                np.mean([record["action_cosine"] for record in measured])
            ),
            "action_cosine_min": float(
                np.min([record["action_cosine"] for record in measured])
            ),
            "action_relative_l2_mean": float(
                np.mean([record["action_relative_l2"] for record in measured])
            ),
            "action_relative_l2_max": float(
                np.max([record["action_relative_l2"] for record in measured])
            ),
            "action_max_abs_max": float(
                np.max([record["action_max_abs"] for record in measured])
            ),
        },
        "records": records,
    }
    if args.record_video_sensitivity:
        report["summary"].update(
            {
                "video_cosine_mean": float(
                    np.mean([record["video_cosine"] for record in measured])
                ),
                "video_cosine_min": float(
                    np.min([record["video_cosine"] for record in measured])
                ),
                "video_relative_l2_mean": float(
                    np.mean(
                        [record["video_relative_l2"] for record in measured]
                    )
                ),
                "video_relative_l2_max": float(
                    np.max(
                        [record["video_relative_l2"] for record in measured]
                    )
                ),
                "video_max_abs_max": float(
                    np.max([record["video_max_abs"] for record in measured])
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
