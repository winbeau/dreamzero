from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_utils.policy_client import WebsocketClientPolicy


def make_request(*, request_index: int, seed: int, session_id: str) -> dict[str, object]:
    images = []
    for camera_index in range(3):
        rng = np.random.default_rng(seed + 1009 * request_index + camera_index)
        images.append(rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8))

    return {
        "observation/exterior_image_0_left": images[0],
        "observation/exterior_image_1_left": images[1],
        "observation/wrist_image_left": images[2],
        "observation/joint_position": np.array(
            [0.0, -0.55, 0.0, -2.10, 0.0, 1.55, 0.78],
            dtype=np.float64,
        ),
        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
        "observation/gripper_position": np.array([0.0], dtype=np.float64),
        "prompt": "put the cube in the bowl",
        "session_id": session_id,
    }


def summarize(latencies: list[float]) -> dict[str, float]:
    values = np.asarray(latencies, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one measured latency is required")
    return {
        "mean_seconds": float(values.mean()),
        "median_seconds": float(np.median(values)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "p90_seconds": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the real DreamZero WebSocket policy end to end."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--measured-requests", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    if args.measured_requests <= 0:
        raise ValueError("measured_requests must be positive")

    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    session_id = f"dreamzero-e2e-{args.label}-{args.seed}"
    records = []
    measured_latencies = []
    total_requests = args.warmup_requests + args.measured_requests

    try:
        for request_index in range(total_requests):
            request = make_request(
                request_index=request_index,
                seed=args.seed,
                session_id=session_id,
            )
            start = time.perf_counter()
            response = client.infer(request)
            latency = time.perf_counter() - start
            action = np.asarray(response)
            phase = "warmup" if request_index < args.warmup_requests else "measured"
            record = {
                "request_index": request_index,
                "phase": phase,
                "latency_seconds": latency,
                "action_shape": list(action.shape),
                "action_dtype": str(action.dtype),
            }
            records.append(record)
            if phase == "measured":
                measured_latencies.append(latency)
            print(json.dumps(record), flush=True)
    finally:
        client._ws.close()

    report = {
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "seed": args.seed,
        "warmup_requests": args.warmup_requests,
        "measured_requests": args.measured_requests,
        "server_metadata": metadata,
        "summary": summarize(measured_latencies),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
