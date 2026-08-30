#!/usr/bin/env python3
"""
Download all or an exact Oracle subset of DreamZero DROID from Hugging Face.

Use this script when `huggingface-cli download` hits 429 (Too Many Requests).
It uses a single worker and retries with backoff to stay within the 3000 req/5min limit.

Usage:
  python scripts/data/download_droid_hf.py --metadata-only --local-dir /artifact/droid_metadata

  python scripts/data/download_droid_hf.py \
    --selection-manifest /artifact/oracle_selection.json \
    --local-dir /artifact/droid_oracle_subset

  # Or with env (same as CLI default):
  DROID_DATA_ROOT=./data/droid_lerobot python scripts/data/download_droid_hf.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

from groot.vla.data.droid_oracle_subset import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    METADATA_FILES,
    materialize_lerobot_subset,
    required_episode_files,
)


REPO_ID = DEFAULT_REPO_ID
REPO_TYPE = "dataset"


def _with_rate_limit_retry(operation, retry_wait: int):
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except Exception as error:
            error_text = str(error).lower()
            if any(marker in error_text for marker in ("429", "too many requests", "rate limit")):
                print(
                    f"Rate limited on attempt {attempt}. Waiting {retry_wait}s before retry...",
                    file=sys.stderr,
                )
                time.sleep(retry_wait)
                continue
            raise


def _download_file(filename: str, *, local_dir: Path, revision: str, retry_wait: int) -> Path:
    def operation():
        return hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
        )

    return Path(_with_rate_limit_retry(operation, retry_wait))


def _download_metadata(*, local_dir: Path, revision: str, retry_wait: int) -> None:
    for filename in METADATA_FILES:
        print(f"metadata: {filename}")
        _download_file(
            filename, local_dir=local_dir, revision=revision, retry_wait=retry_wait
        )


def _download_selection(
    *, local_dir: Path, manifest_path: Path, retry_wait: int
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != REPO_ID:
        raise ValueError(f"Unexpected manifest repo_id: {manifest.get('repo_id')}")
    revision = manifest["revision"]
    source_dir = local_dir / "_source"
    _download_metadata(local_dir=source_dir, revision=revision, retry_wait=retry_wait)
    info = json.loads((source_dir / "meta/info.json").read_text())
    for item in manifest["selections"]:
        source_episode_index = int(item["source_episode_index"])
        for filename in required_episode_files(info, source_episode_index):
            print(f"episode {source_episode_index}: {filename}")
            _download_file(
                filename,
                local_dir=source_dir,
                revision=revision,
                retry_wait=retry_wait,
            )
    return materialize_lerobot_subset(source_dir, local_dir, manifest)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download DreamZero DROID dataset with rate-limit handling."
    )
    p.add_argument(
        "--local-dir",
        default=os.environ.get("DROID_DATA_ROOT", "./data/droid_lerobot"),
        help="Local directory to download into (default: DROID_DATA_ROOT or ./data/droid_lerobot)",
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Immutable dataset revision used for full or metadata-only download.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only the small metadata files, using exact file requests.",
    )
    mode.add_argument(
        "--selection-manifest",
        type=Path,
        help="Download and materialize only episodes in an Oracle selection manifest.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent download threads (default 1 to reduce API requests and avoid 429)",
    )
    p.add_argument(
        "--retry-wait",
        type=int,
        default=320,
        help="Seconds to wait on 429 before retry (default 320 ≈ 5min)",
    )
    args = p.parse_args()

    local_dir = Path(os.path.abspath(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_only:
        _download_metadata(
            local_dir=local_dir,
            revision=args.revision,
            retry_wait=args.retry_wait,
        )
        print(f"Done. Metadata at: {local_dir}")
        return

    if args.selection_manifest:
        summary = _download_selection(
            local_dir=local_dir,
            manifest_path=args.selection_manifest,
            retry_wait=args.retry_wait,
        )
        print(json.dumps(summary, indent=2))
        return

    def operation():
        print(f"Full download (max_workers={args.max_workers})...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=local_dir,
            revision=args.revision,
            max_workers=args.max_workers,
        )

    _with_rate_limit_retry(operation, args.retry_wait)
    print(f"Done. Dataset at: {local_dir}")


if __name__ == "__main__":
    main()
