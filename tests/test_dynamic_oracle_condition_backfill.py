import json

import numpy as np
import pandas as pd

from benchmarks.backfill_dynamic_oracle_conditions import backfill


def test_backfill_reconstructs_released_droid_modalities(tmp_path):
    dataset = tmp_path / "dataset"
    parquet_dir = dataset / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)
    state = np.zeros((30, 14), dtype=np.float64)
    action = np.zeros((30, 28), dtype=np.float64)
    state[5, 7:14] = 3.0
    state[5, 6] = 4.0
    action[5, 14:21] = 1.0
    action[28, 14:21] = 2.0
    action[5, 12] = 0.25
    action[28, 12] = 0.75
    pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "frame_index": np.arange(30),
        }
    ).to_parquet(parquet_dir / "episode_000000.parquet")

    oracle = tmp_path / "oracle"
    oracle.mkdir()
    request = {
        "request_key": "train_subset000_source000123_early",
        "subset_episode_index": 0,
        "source_episode_index": 123,
        "trajectory_step": 5,
        "trajectory_length": 30,
        "passed": True,
    }
    (oracle / "request_results.jsonl").write_text(json.dumps(request) + "\n")
    output = tmp_path / "conditions.jsonl"

    summary = backfill(dataset, oracle, output)
    condition = json.loads(output.read_text())

    assert summary["request_count"] == 1
    assert summary["all_finite"]
    assert condition["state_l2"] == np.linalg.norm([*[3.0] * 7, 4.0])
    assert condition["action_temporal_delta_l2"] > 0.0
