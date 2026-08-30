"""Reconstruct missing DROID Oracle state/action condition summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATE_COMPONENTS = {
    "state.joint_position": slice(7, 14),
    "state.gripper_position": slice(6, 7),
}
ACTION_COMPONENTS = {
    "action.joint_position": slice(14, 21),
    "action.gripper_position": slice(12, 13),
}
ACTION_OFFSETS = np.arange(24, dtype=np.int64)


def _condition_summary(data_point: dict) -> dict[str, float]:
    """Mirror the collector formula without importing its model dependencies."""

    state_values = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in data_point.items()
        if key.startswith("state.")
    ]
    action_values = [
        np.asarray(value, dtype=np.float64)
        for key, value in data_point.items()
        if key.startswith("action.")
    ]
    state = np.concatenate(state_values) if state_values else np.zeros(1)
    action = (
        np.concatenate([value.reshape(-1) for value in action_values])
        if action_values
        else np.zeros(1)
    )
    action_delta = (
        np.concatenate([(value[-1] - value[0]).reshape(-1) for value in action_values])
        if action_values
        else np.zeros(1)
    )
    return {
        "state_l2": float(np.linalg.norm(state)),
        "state_abs_mean": float(np.mean(np.abs(state))),
        "action_l2": float(np.linalg.norm(action)),
        "action_std": float(np.std(action)),
        "action_temporal_delta_l2": float(np.linalg.norm(action_delta)),
    }


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def discover_requests(root: Path) -> list[dict]:
    requests = {}
    for path in sorted(root.rglob("request_results.jsonl")):
        for record in _read_jsonl(path):
            if not record.get("passed"):
                continue
            key = record["request_key"]
            if key in requests:
                raise ValueError(f"Duplicate completed request: {key}")
            requests[key] = record
    return [requests[key] for key in sorted(requests)]


def reconstruct_condition(dataset_path: Path, request: dict) -> dict[str, object]:
    episode = int(request["subset_episode_index"])
    parquet_path = (
        dataset_path
        / "data"
        / f"chunk-{episode // 1000:03d}"
        / f"episode_{episode:06d}.parquet"
    )
    table = pd.read_parquet(
        parquet_path, columns=["observation.state", "action", "frame_index"]
    )
    state = np.stack(table["observation.state"].to_numpy())
    action = np.stack(table["action"].to_numpy())
    base_step = int(request["trajectory_step"])
    if len(table) != int(request["trajectory_length"]):
        raise ValueError(
            f"{request['request_key']} trajectory length mismatch: "
            f"{len(table)} vs {request['trajectory_length']}"
        )
    if table["frame_index"].iloc[base_step] != base_step:
        raise ValueError(f"{request['request_key']} frame_index is not contiguous")
    action_indices = np.clip(ACTION_OFFSETS + base_step, 0, len(table) - 1)
    data_point = {
        key: state[[base_step], component]
        for key, component in STATE_COMPONENTS.items()
    }
    data_point.update(
        {
            key: action[action_indices, component]
            for key, component in ACTION_COMPONENTS.items()
        }
    )
    return {
        "request_key": request["request_key"],
        "subset_episode_index": episode,
        "source_episode_index": int(request["source_episode_index"]),
        "trajectory_step": base_step,
        "condition_source": "raw compact DROID parquet; released eval modality slices",
        **_condition_summary(data_point),
    }


def backfill(dataset_path: Path, oracle_root: Path, output: Path) -> dict[str, object]:
    requests = discover_requests(oracle_root)
    if not requests:
        raise ValueError("No passed Oracle requests found")
    records = [reconstruct_condition(dataset_path, request) for request in requests]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    temporary_output.replace(output)
    values = {
        key: np.asarray([record[key] for record in records], dtype=np.float64)
        for key in (
            "state_l2",
            "state_abs_mean",
            "action_l2",
            "action_std",
            "action_temporal_delta_l2",
        )
    }
    summary = {
        "request_count": len(records),
        "output": str(output),
        "all_finite": bool(all(np.isfinite(value).all() for value in values.values())),
        "condition_ranges": {
            key: {"min": float(value.min()), "max": float(value.max())}
            for key, value in values.items()
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(backfill(args.dataset_path, args.oracle_root, args.output), indent=2))


if __name__ == "__main__":
    main()
