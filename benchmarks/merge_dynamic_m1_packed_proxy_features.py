"""Merge causal Packed-proxy observations into the per-Head Oracle table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from groot.vla.model.dreamzero.modules.dynamic_m1_observation import (
    load_packed_m1_observations,
)

PROXY_FEATURE_COLUMNS = (
    "previous_packed_route_support_turnover_max",
    "previous_packed_route_normalized_entropy_mean",
    "previous_packed_route_max_mass_mean",
    "previous_packed_action_output_change_relative_l2_max",
    "previous_two_packed_action_output_change_relative_l2_max",
    "previous_packed_action_output_change_cosine_min",
    "previous_packed_cfg_disagreement_relative_l2",
    "previous_packed_action_output_signature_norm",
)


def proxy_feature_frame(
    artifact: Path,
    *,
    expected_dit_steps: int = 8,
) -> pd.DataFrame:
    observations, metadata = load_packed_m1_observations(artifact)
    request_key = metadata.get("request_key")
    if not isinstance(request_key, str) or not request_key:
        raise ValueError(f"Packed proxy artifact has no request_key: {artifact}")
    if len(observations) != expected_dit_steps:
        raise ValueError(
            f"Packed proxy artifact must contain {expected_dit_steps} real DiTs"
        )
    first = next(
        (observation for observation in observations if observation is not None),
        None,
    )
    if first is None:
        raise ValueError(f"Packed proxy artifact contains no valid DiT: {artifact}")
    num_layers, num_heads = first.shape
    grid_size = num_layers * num_heads

    def observation_metric(observation, name: str) -> np.ndarray:
        if observation is None:
            return np.full(grid_size, np.nan)
        return observation.metric(name).reshape(-1)

    rows: list[pd.DataFrame] = []
    for dit_index in range(expected_dit_steps):
        previous = observations[dit_index - 1] if dit_index >= 1 else None
        previous_two = observations[dit_index - 2] if dit_index >= 2 else None
        data: dict[str, object] = {
            "request_key": np.repeat(request_key, grid_size),
            "dit_index": np.repeat(dit_index, grid_size),
            "layer_index": np.repeat(np.arange(num_layers), num_heads),
            "head_index": np.tile(np.arange(num_heads), num_layers),
        }

        data.update(
            {
                "previous_packed_route_support_turnover_max": observation_metric(
                    previous, "packed_route_support_turnover_max"
                ),
                "previous_packed_route_normalized_entropy_mean": observation_metric(
                    previous, "packed_route_normalized_entropy_mean"
                ),
                "previous_packed_route_max_mass_mean": observation_metric(
                    previous, "packed_route_max_mass_mean"
                ),
                "previous_packed_action_output_change_relative_l2_max": (
                    observation_metric(
                        previous, "packed_action_output_change_relative_l2_max"
                    )
                ),
                "previous_packed_action_output_change_cosine_min": observation_metric(
                    previous, "packed_action_output_change_cosine_min"
                ),
                "previous_packed_cfg_disagreement_relative_l2": observation_metric(
                    previous, "packed_cfg_disagreement_relative_l2"
                ),
                "previous_packed_action_output_signature_norm": observation_metric(
                    previous, "packed_action_output_signature_norm"
                ),
                "previous_two_packed_action_output_change_relative_l2_max": (
                    observation_metric(
                        previous_two, "packed_action_output_change_relative_l2_max"
                    )
                ),
            }
        )
        rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True)


def merge_proxy_features(
    oracle_table: pd.DataFrame,
    proxy_table: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["request_key", "dit_index", "layer_index", "head_index"]
    if oracle_table.duplicated(keys).any():
        raise ValueError("Oracle table contains duplicate request/DiT/layer/Head rows")
    if proxy_table.duplicated(keys).any():
        raise ValueError("Packed proxy table contains duplicate grid rows")
    overlap = set(PROXY_FEATURE_COLUMNS) & set(oracle_table.columns)
    if overlap:
        raise ValueError(
            f"Oracle table already contains proxy columns: {sorted(overlap)}"
        )
    merged = oracle_table.merge(
        proxy_table,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged["_merge"] != "both"
    if missing.any():
        missing_requests = merged.loc[missing, "request_key"].nunique()
        raise ValueError(
            f"Packed proxy coverage is missing for {missing_requests} Oracle requests"
        )
    return merged.drop(columns="_merge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-table", type=Path, required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    artifacts = sorted(args.proxy_dir.rglob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No Packed proxy artifacts under {args.proxy_dir}")
    proxy = pd.concat(
        [proxy_feature_frame(path) for path in artifacts],
        ignore_index=True,
    )
    oracle = pd.read_parquet(args.oracle_table)
    merged = merge_proxy_features(oracle, proxy)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_table, index=False)
    summary = {
        "oracle_table": str(args.oracle_table),
        "proxy_dir": str(args.proxy_dir),
        "output_table": str(args.output_table),
        "artifact_count": len(artifacts),
        "request_count": int(proxy["request_key"].nunique()),
        "proxy_rows": len(proxy),
        "merged_rows": len(merged),
        "feature_columns": list(PROXY_FEATURE_COLUMNS),
    }
    summary_path = args.summary or args.output_table.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
