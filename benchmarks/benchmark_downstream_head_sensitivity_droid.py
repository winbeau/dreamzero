from __future__ import annotations

import argparse
import json
import sys
import time
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


def action_sensitivity_metrics(
    baseline: np.ndarray,
    intervened: np.ndarray,
) -> dict[str, float]:
    baseline_flat = np.asarray(baseline, dtype=np.float64).reshape(-1)
    intervened_flat = np.asarray(intervened, dtype=np.float64).reshape(-1)
    if baseline_flat.shape != intervened_flat.shape:
        raise ValueError("paired action shapes do not match")
    baseline_norm = float(np.linalg.norm(baseline_flat))
    intervened_norm = float(np.linalg.norm(intervened_flat))
    denominator = max(baseline_norm * intervened_norm, 1e-12)
    difference = intervened_flat - baseline_flat
    return {
        "action_cosine": float(
            np.dot(baseline_flat, intervened_flat) / denominator
        ),
        "action_relative_l2": float(
            np.linalg.norm(difference) / max(baseline_norm, 1e-12)
        ),
        "action_max_abs": float(np.max(np.abs(difference))),
    }


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


def run_chain(
    client: WebsocketClientPolicy,
    observations: list[dict[str, Any]],
    *,
    target_control: dict[str, Any] | None,
    session_id: str,
) -> tuple[np.ndarray, list[float], float]:
    history_latencies = []
    for observation in observations[:-1]:
        observation[CONTROL_KEY] = {"enabled": False}
        started = time.perf_counter()
        client.infer(observation)
        history_latencies.append(time.perf_counter() - started)

    target = observations[-1]
    target[CONTROL_KEY] = (
        {"enabled": False} if target_control is None else target_control
    )
    started = time.perf_counter()
    action = np.asarray(client.infer(target))
    target_latency = time.perf_counter() - started
    client.reset({"session_id": session_id})
    return action, history_latencies, target_latency


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
            baseline, baseline_history, baseline_latency = run_chain(
                client,
                baseline_observations,
                target_control=None,
                session_id=session_id,
            )
            intervention_observations = reader.observations(
                request,
                history_blocks=args.history_blocks,
                session_id=session_id,
            )
            intervened, intervention_history, intervention_latency = run_chain(
                client,
                intervention_observations,
                target_control=control,
                session_id=session_id,
            )
            metrics = action_sensitivity_metrics(baseline, intervened)
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
                "baseline_latency_seconds": baseline_latency,
                "intervention_latency_seconds": intervention_latency,
                "action_shape": list(baseline.shape),
                "baseline_action": baseline.tolist(),
                "intervention_action": intervened.tolist(),
                **metrics,
            }
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
