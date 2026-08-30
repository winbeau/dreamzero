"""Deterministic, task-disjoint sampling for DreamZero DROID Oracle runs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Iterable


DEFAULT_REPO_ID = "GEAR-Dreams/DreamZero-DROID-Data"
DEFAULT_REVISION = "2abc197ca7f14f53a6bf464bf80018ce998f18cc"
METADATA_FILES = (
    "meta/episodes.jsonl",
    "meta/info.json",
    "meta/modality.json",
    "meta/relative_horizon_stats_dreamzero.json",
    "meta/relative_stats.json",
    "meta/relative_stats_dreamzero.json",
    "meta/stats.json",
    "meta/tasks.jsonl",
    "relative_stats_dreamzero.json",
)
TRAJECTORY_STAGES = (
    {"name": "early", "fraction": 0.10},
    {"name": "middle", "fraction": 0.50},
    {"name": "late", "fraction": 0.90},
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_task(task: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(task).lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _is_informative_task(task: str) -> bool:
    return normalize_task(task) not in {"", "none", "not provided", "unknown", "n a"}


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class _DisjointSet:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def _percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _length_bucket(length: int, low_edge: float, high_edge: float) -> str:
    if length <= low_edge:
        return "short"
    if length <= high_edge:
        return "middle"
    return "long"


def build_oracle_selection(
    episodes: list[dict],
    *,
    counts: dict[str, int],
    seed: int = 20260830,
    min_length: int = 96,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Select balanced episodes while preventing exact task leakage across splits.

    Episodes sharing any normalized, informative instruction are put in the same
    connected component. Components, not individual episodes, are assigned to a
    split. At most one episode is selected from each component to maximize task
    diversity.
    """

    split_names = ("train", "validation", "test")
    if set(counts) != set(split_names):
        raise ValueError(f"counts must have exactly these keys: {split_names}")
    for split, count in counts.items():
        if count <= 0 or count % 3:
            raise ValueError(f"{split} count must be a positive multiple of three")

    eligible = [
        episode
        for episode in episodes
        if episode.get("success", True)
        and int(episode["length"]) >= min_length
        and any(_is_informative_task(task) for task in episode.get("tasks", ()))
    ]
    if not eligible:
        raise ValueError("No eligible episodes")

    episode_by_id = {int(episode["episode_index"]): episode for episode in eligible}
    disjoint = _DisjointSet(episode_by_id)
    first_episode_for_task: dict[str, int] = {}
    normalized_tasks: dict[int, tuple[str, ...]] = {}
    for episode_id, episode in episode_by_id.items():
        tasks = tuple(
            sorted(
                {
                    normalize_task(task)
                    for task in episode.get("tasks", ())
                    if _is_informative_task(task)
                }
            )
        )
        normalized_tasks[episode_id] = tasks
        for task in tasks:
            previous = first_episode_for_task.setdefault(task, episode_id)
            disjoint.union(episode_id, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for episode_id in episode_by_id:
        components[disjoint.find(episode_id)].append(episode_id)

    lengths = sorted(int(episode["length"]) for episode in eligible)
    low_edge = _percentile(lengths, 1.0 / 3.0)
    high_edge = _percentile(lengths, 2.0 / 3.0)
    split_ranges = {
        "train": range(0, 4),
        "validation": range(4, 5),
        "test": range(5, 6),
    }
    candidate_components: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    for root, episode_ids in components.items():
        signature = sorted({task for episode_id in episode_ids for task in normalized_tasks[episode_id]})
        split_slot = _stable_int(seed, "component", *signature) % 6
        split = next(name for name, slots in split_ranges.items() if split_slot in slots)
        candidate_components[split].append((root, episode_ids))

    selected: list[dict] = []
    used_components: set[int] = set()
    for split in split_names:
        per_bucket = counts[split] // 3
        for bucket in ("short", "middle", "long"):
            candidates: list[tuple[int, int]] = []
            for root, episode_ids in candidate_components[split]:
                if root in used_components:
                    continue
                matching = [
                    episode_id
                    for episode_id in episode_ids
                    if _length_bucket(
                        int(episode_by_id[episode_id]["length"]), low_edge, high_edge
                    )
                    == bucket
                ]
                if matching:
                    episode_id = min(
                        matching,
                        key=lambda value: _stable_int(seed, split, bucket, value),
                    )
                    candidates.append((root, episode_id))
            candidates.sort(key=lambda item: _stable_int(seed, split, bucket, item[1]))
            if len(candidates) < per_bucket:
                raise ValueError(
                    f"Not enough {split}/{bucket} task components: "
                    f"need {per_bucket}, found {len(candidates)}"
                )
            for root, episode_id in candidates[:per_bucket]:
                used_components.add(root)
                episode = episode_by_id[episode_id]
                selected.append(
                    {
                        "source_episode_index": episode_id,
                        "split": split,
                        "length": int(episode["length"]),
                        "length_bucket": bucket,
                        "success": bool(episode.get("success", True)),
                        "tasks": list(episode.get("tasks", ())),
                        "normalized_tasks": list(normalized_tasks[episode_id]),
                        "trajectory_stages": [dict(stage) for stage in TRAJECTORY_STAGES],
                    }
                )

    split_order = {name: index for index, name in enumerate(split_names)}
    selected.sort(
        key=lambda item: (
            split_order[item["split"]],
            ("short", "middle", "long").index(item["length_bucket"]),
            item["source_episode_index"],
        )
    )
    for subset_episode_index, item in enumerate(selected):
        item["subset_episode_index"] = subset_episode_index

    tasks_by_split = {
        split: {
            task
            for item in selected
            if item["split"] == split
            for task in item["normalized_tasks"]
        }
        for split in split_names
    }
    leakage = {
        f"{left}_{right}": sorted(tasks_by_split[left] & tasks_by_split[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    if any(leakage.values()):
        raise AssertionError(f"Task leakage detected: {leakage}")

    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "seed": seed,
        "minimum_episode_length": min_length,
        "length_bucket_edges": {"short_max": low_edge, "middle_max": high_edge},
        "requested_counts": counts,
        "selected_episode_count": len(selected),
        "selected_request_count": len(selected) * len(TRAJECTORY_STAGES),
        "task_component_count": len(components),
        "task_leakage": leakage,
        "selections": selected,
    }


def required_episode_files(info: dict, source_episode_index: int) -> list[str]:
    chunk_size = int(info["chunks_size"])
    chunk = source_episode_index // chunk_size
    paths = [
        info["data_path"].format(
            episode_chunk=chunk, episode_index=source_episode_index
        )
    ]
    for video_key, feature in info["features"].items():
        if feature.get("dtype") == "video":
            paths.append(
                info["video_path"].format(
                    episode_chunk=chunk,
                    episode_index=source_episode_index,
                    video_key=video_key,
                )
            )
    return paths


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise ValueError(f"Existing materialized file has the wrong size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_lerobot_subset(
    source_root: Path, destination_root: Path, manifest: dict
) -> dict:
    """Create a compact, sequentially indexed LeRobot view of downloaded files."""

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    source_info = json.loads((source_root / "meta/info.json").read_text())
    selections = manifest["selections"]
    destination_root.mkdir(parents=True, exist_ok=True)

    for metadata_file in METADATA_FILES:
        source = source_root / metadata_file
        if not source.exists():
            continue
        destination = destination_root / metadata_file
        if metadata_file in {"meta/info.json", "meta/episodes.jsonl"}:
            continue
        _link_or_copy(source, destination)

    total_frames = sum(int(item["length"]) for item in selections)
    destination_info = dict(source_info)
    destination_info.update(
        {
            "total_episodes": len(selections),
            "total_frames": total_frames,
            "total_chunks": math.ceil(len(selections) / int(source_info["chunks_size"])),
            "splits": {"train": "0:100"},
        }
    )
    info_path = destination_root / "meta/info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(destination_info, indent=2) + "\n", encoding="utf-8")

    episodes_path = destination_root / "meta/episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for item in selections:
            record = {
                "episode_index": int(item["subset_episode_index"]),
                "tasks": item["tasks"],
                "length": int(item["length"]),
                "success": bool(item["success"]),
                "source_episode_index": int(item["source_episode_index"]),
                "oracle_split": item["split"],
                "length_bucket": item["length_bucket"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for item in selections:
        source_episode_index = int(item["source_episode_index"])
        subset_episode_index = int(item["subset_episode_index"])
        source_paths = required_episode_files(source_info, source_episode_index)
        destination_paths = required_episode_files(destination_info, subset_episode_index)
        for source_relative, destination_relative in zip(source_paths, destination_paths):
            source = source_root / source_relative
            if not source.exists():
                raise FileNotFoundError(source)
            _link_or_copy(source, destination_root / destination_relative)

    subset_manifest = destination_root / "oracle_subset_manifest.json"
    subset_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "episodes": len(selections),
        "requests": sum(len(item["trajectory_stages"]) for item in selections),
        "frames": total_frames,
        "destination": str(destination_root),
    }
