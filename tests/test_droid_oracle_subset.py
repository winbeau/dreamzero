import json
from pathlib import Path

from groot.vla.data.droid_oracle_subset import (
    build_oracle_selection,
    materialize_lerobot_subset,
    required_episode_files,
)


def _episodes(count: int = 180) -> list[dict]:
    episodes = []
    for episode_index in range(count):
        length = 100 + episode_index
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [
                    f"unique instruction {episode_index}",
                    f"shared component {episode_index // 2}",
                ],
                "length": length,
                "success": True,
            }
        )
    return episodes


def test_selection_is_balanced_reproducible_and_task_disjoint():
    counts = {"train": 12, "validation": 6, "test": 6}
    first = build_oracle_selection(_episodes(), counts=counts, seed=7)
    second = build_oracle_selection(_episodes(), counts=counts, seed=7)

    assert first == second
    assert first["selected_episode_count"] == 24
    assert first["selected_request_count"] == 72
    assert all(not leaked for leaked in first["task_leakage"].values())
    for split, count in counts.items():
        rows = [row for row in first["selections"] if row["split"] == split]
        assert len(rows) == count
        assert {
            bucket: sum(row["length_bucket"] == bucket for row in rows)
            for bucket in ("short", "middle", "long")
        } == {bucket: count // 3 for bucket in ("short", "middle", "long")}
        assert all(len(row["trajectory_stages"]) == 3 for row in rows)


def test_shared_task_components_never_cross_splits():
    manifest = build_oracle_selection(
        _episodes(), counts={"train": 12, "validation": 6, "test": 6}, seed=11
    )
    task_splits = {}
    for row in manifest["selections"]:
        for task in row["normalized_tasks"]:
            task_splits.setdefault(task, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in task_splits.values())


def test_materialize_reindexes_paths_and_metadata(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    info = {
        "total_episodes": 20,
        "total_frames": 2000,
        "total_chunks": 2,
        "total_tasks": 2,
        "chunks_size": 10,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.left": {"dtype": "video"},
            "observation.state": {"dtype": "float64"},
        },
    }
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text(json.dumps(info))
    for relative in (
        "meta/modality.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/relative_stats.json",
        "meta/relative_stats_dreamzero.json",
        "meta/relative_horizon_stats_dreamzero.json",
        "relative_stats_dreamzero.json",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    selections = []
    for subset_episode_index, source_episode_index in enumerate((12, 17)):
        for relative in required_episode_files(info, source_episode_index):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"source-{source_episode_index}-{relative}".encode())
        selections.append(
            {
                "source_episode_index": source_episode_index,
                "subset_episode_index": subset_episode_index,
                "split": "train",
                "length": 100 + subset_episode_index,
                "length_bucket": "short",
                "success": True,
                "tasks": [f"task {source_episode_index}"],
                "normalized_tasks": [f"task {source_episode_index}"],
                "trajectory_stages": [{"name": "middle", "fraction": 0.5}],
            }
        )
    manifest = {"selections": selections}

    summary = materialize_lerobot_subset(source, destination, manifest)

    compact_info = json.loads((destination / "meta/info.json").read_text())
    compact_episodes = [json.loads(line) for line in (destination / "meta/episodes.jsonl").open()]
    assert summary["episodes"] == 2
    assert summary["requests"] == 2
    assert compact_info["total_episodes"] == 2
    assert compact_info["total_frames"] == 201
    assert [row["episode_index"] for row in compact_episodes] == [0, 1]
    assert [row["source_episode_index"] for row in compact_episodes] == [12, 17]
    for subset_episode_index in (0, 1):
        for relative in required_episode_files(compact_info, subset_episode_index):
            assert (destination / relative).exists()
