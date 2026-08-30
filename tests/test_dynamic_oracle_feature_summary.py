import json

import numpy as np
import pandas as pd

from benchmarks.summarize_dynamic_oracle_features import summarize


def test_oracle_feature_summary_preserves_u_shaped_layer_evidence(tmp_path):
    rows = []
    for episode, split in ((0, "train"), (1, "validation"), (2, "test")):
        for dit_index in range(8):
            for layer_index, budget in ((0, 0.75), (1, 0.2), (2, 1.0)):
                for head_index in range(2):
                    rows.append(
                        {
                            "request_key": f"{split}-{episode}",
                            "split": split,
                            "trajectory_stage": ("early", "middle", "late")[episode],
                            "source_episode_index": episode,
                            "dit_index": dit_index,
                            "layer_index": layer_index,
                            "head_index": head_index,
                            "oracle_min_keep_ratio": budget,
                            "video_oracle_min_keep_ratio": budget,
                            "action_oracle_min_keep_ratio": budget,
                            "support_turnover_max": 0.1,
                            "vv_output_change_relative_l2_max": 0.02,
                            "qa_qv_key_importance_correlation_mean": 0.8,
                            "worst_mass_p05_r020": 0.95 if budget <= 0.2 else 0.5,
                        }
                    )
    compact = tmp_path / "compact.parquet"
    pd.DataFrame(rows).to_parquet(compact)
    cube = tmp_path / "cube.npz"
    shape = (2, 2, 8, 3, 2)
    np.savez_compressed(
        cube,
        mean_budget=np.full(shape, 0.5),
        task_std=np.full(shape, 0.1),
        dense_fallback_rate=np.full(shape, 0.2),
        mean_mass_retention_at_20pct=np.full(shape, 0.85),
    )

    summary = summarize(compact, cube, tmp_path / "output")

    layers = {row["layer_index"]: row for row in summary["layer"]}
    assert layers[1]["oracle_mean"] < layers[0]["oracle_mean"] < layers[2]["oracle_mean"]
    assert summary["overall"]["fixed_20pct_worst_mass_p05_pass_rate"] == 1 / 3
    assert {row["split"] for row in summary["split"]} == {"train", "val", "test"}
    saved = json.loads((tmp_path / "output" / "oracle_feature_summary.json").read_text())
    assert saved["row_count"] == len(rows)
