from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_dreamzero_server_e2e import summarize
from eval_utils.policy_client import WebsocketClientPolicy


CAMERA_PATHS = {
    "observation/exterior_image_0_left": "observation.images.exterior_image_1_left",
    "observation/exterior_image_1_left": "observation.images.exterior_image_2_left",
    "observation/wrist_image_left": "observation.images.wrist_image_left",
}
STAGE_TO_INSTRUCTION = {"early": 0, "middle": 1, "late": 2}


def build_request_plan(
    manifest: dict[str, Any],
    *,
    splits: set[str],
    stages: set[str],
    max_requests: int | None,
) -> list[dict[str, Any]]:
    plan = []
    for selection in manifest["selections"]:
        if selection["split"] not in splits:
            continue
        for stage in selection["trajectory_stages"]:
            if stage["name"] not in stages:
                continue
            request_key = (
                f"{selection['split']}_subset{selection['subset_episode_index']:03d}_"
                f"source{selection['source_episode_index']:06d}_{stage['name']}"
            )
            plan.append({**selection, "stage": stage, "request_key": request_key})
    if max_requests is not None:
        plan = plan[:max_requests]
    return plan


def history_frame_groups(
    *, base_step: int, trajectory_length: int, history_blocks: int
) -> list[list[int]]:
    if history_blocks < 0:
        raise ValueError("history_blocks must be non-negative")
    if trajectory_length <= 0:
        raise ValueError("trajectory_length must be positive")

    def clipped(values: list[int]) -> list[int]:
        return [min(max(value, 0), trajectory_length - 1) for value in values]

    groups: list[list[int]] = []
    if history_blocks:
        first_step = base_step - 4 * history_blocks
        groups.append(clipped([first_step]))
        for block_index in range(1, history_blocks):
            block_end = base_step - 4 * (history_blocks - block_index)
            groups.append(clipped(list(range(block_end - 3, block_end + 1))))
    groups.append(clipped(list(range(base_step - 3, base_step + 1))))
    return groups


def split_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape != (14,):
        raise ValueError(f"expected 14-element DROID state, got {state.shape}")
    return state[7:14], state[0:6], state[6:7]


def _read_video_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video {path}")
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {index} from {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


class DroidRequestReader:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self._states: dict[int, np.ndarray] = {}

    def _episode_states(self, episode_index: int) -> np.ndarray:
        if episode_index not in self._states:
            path = (
                self.dataset_path
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            table = pq.read_table(path, columns=["observation.state"])
            self._states[episode_index] = np.asarray(
                table.column("observation.state").to_pylist(), dtype=np.float64
            )
        return self._states[episode_index]

    def observations(
        self,
        request: dict[str, Any],
        *,
        history_blocks: int,
        session_id: str,
    ) -> list[dict[str, Any]]:
        episode_index = int(request["subset_episode_index"])
        trajectory_length = int(request["length"])
        base_step = round(
            (trajectory_length - 1) * float(request["stage"]["fraction"])
        )
        groups = history_frame_groups(
            base_step=base_step,
            trajectory_length=trajectory_length,
            history_blocks=history_blocks,
        )
        all_indices = [index for group in groups for index in group]
        camera_frames: dict[str, list[np.ndarray]] = {}
        chunk = episode_index // 1000
        for server_key, video_key in CAMERA_PATHS.items():
            path = (
                self.dataset_path
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            camera_frames[server_key] = _read_video_frames(path, all_indices)

        instruction_index = STAGE_TO_INSTRUCTION[request["stage"]["name"]]
        task = request["tasks"][instruction_index]
        states = self._episode_states(episode_index)
        observations = []
        cursor = 0
        for group in groups:
            joint, cartesian, gripper = split_state(states[group[-1]])
            observation: dict[str, Any] = {
                "observation/joint_position": joint,
                "observation/cartesian_position": cartesian,
                "observation/gripper_position": gripper,
                "prompt": task,
                "session_id": session_id,
            }
            for server_key in CAMERA_PATHS:
                frames = camera_frames[server_key][cursor : cursor + len(group)]
                observation[server_key] = (
                    frames[0] if len(frames) == 1 else np.stack(frames, axis=0)
                )
            observations.append(observation)
            cursor += len(group)
        return observations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DreamZero on the task-disjoint real DROID Oracle subset."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("early", "middle", "late"),
        default=("early", "middle", "late"),
    )
    parser.add_argument("--history-blocks", type=int, default=3)
    parser.add_argument("--warmup-target-requests", type=int, default=1)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    if args.warmup_target_requests < 0:
        parser.error("--warmup-target-requests must be non-negative")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")

    manifest = json.loads(args.manifest.read_text())
    plan = build_request_plan(
        manifest,
        splits=set(args.splits),
        stages=set(args.stages),
        max_requests=args.max_requests,
    )
    if len(plan) <= args.warmup_target_requests:
        raise ValueError("request plan contains no measured target requests")

    reader = DroidRequestReader(args.dataset_path)
    client = WebsocketClientPolicy(args.host, args.port)
    records = []
    measured_latencies = []
    session_id = f"dreamzero-droid-{args.label}"
    try:
        for request_index, request in enumerate(plan):
            observations = reader.observations(
                request,
                history_blocks=args.history_blocks,
                session_id=session_id,
            )
            history_latencies = []
            for observation in observations[:-1]:
                started = time.perf_counter()
                client.infer(observation)
                history_latencies.append(time.perf_counter() - started)

            started = time.perf_counter()
            response = client.infer(observations[-1])
            target_latency = time.perf_counter() - started
            action = np.asarray(response)
            phase = (
                "warmup"
                if request_index < args.warmup_target_requests
                else "measured"
            )
            record = {
                "request_index": request_index,
                "request_key": request["request_key"],
                "phase": phase,
                "split": request["split"],
                "trajectory_stage": request["stage"]["name"],
                "subset_episode_index": request["subset_episode_index"],
                "source_episode_index": request["source_episode_index"],
                "task": request["tasks"][
                    STAGE_TO_INSTRUCTION[request["stage"]["name"]]
                ],
                "history_request_count": len(history_latencies),
                "history_latency_seconds": history_latencies,
                "latency_seconds": target_latency,
                "action_shape": list(action.shape),
                "action_dtype": str(action.dtype),
                "action": action.tolist(),
            }
            records.append(record)
            if phase == "measured":
                measured_latencies.append(target_latency)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"action", "history_latency_seconds"}
                    }
                ),
                flush=True,
            )
            client.reset({"session_id": session_id})
    finally:
        client._ws.close()

    report = {
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "seed": manifest["seed"],
        "dataset_path": str(args.dataset_path),
        "manifest": str(args.manifest),
        "splits": args.splits,
        "stages": args.stages,
        "history_blocks": args.history_blocks,
        "warmup_requests": args.warmup_target_requests,
        "measured_requests": len(measured_latencies),
        "server_metadata": client.get_server_metadata(),
        "summary": summarize(measured_latencies),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
