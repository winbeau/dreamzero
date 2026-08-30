#!/usr/bin/env python3
"""Build a deterministic task-disjoint DROID episode manifest for Oracle runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groot.vla.data.droid_oracle_subset import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    build_oracle_selection,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--validation-episodes", type=int, default=6)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--minimum-episode-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    episodes = read_jsonl(args.metadata_root / "meta/episodes.jsonl")
    manifest = build_oracle_selection(
        episodes,
        counts={
            "train": args.train_episodes,
            "validation": args.validation_episodes,
            "test": args.test_episodes,
        },
        seed=args.seed,
        min_length=args.minimum_episode_length,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    summary = {
        key: manifest[key]
        for key in (
            "repo_id",
            "revision",
            "selected_episode_count",
            "selected_request_count",
            "length_bucket_edges",
            "requested_counts",
            "task_component_count",
            "task_leakage",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
