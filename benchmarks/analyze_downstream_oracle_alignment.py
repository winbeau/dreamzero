from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


LOCAL_COLUMNS = (
    "request_key",
    "dit_index",
    "layer_index",
    "head_index",
    "oracle_min_keep_ratio",
    "video_oracle_min_keep_ratio",
    "action_oracle_min_keep_ratio",
)


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def finite_correlation(
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
    *,
    rank: bool = False,
) -> float | None:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[finite]
    second_array = second_array[finite]
    if first_array.size < 2:
        return None
    if rank:
        first_array = average_tie_ranks(first_array)
        second_array = average_tie_ranks(second_array)
    if np.ptp(first_array) == 0.0 or np.ptp(second_array) == 0.0:
        return None
    return float(np.corrcoef(first_array, second_array)[0, 1])


def build_local_lookup(table) -> dict[tuple[str, int, int, int], dict[str, float]]:
    lookup = {}
    columns = {name: table[name].to_pylist() for name in LOCAL_COLUMNS}
    for row_index in range(table.num_rows):
        key = (
            str(columns["request_key"][row_index]),
            int(columns["dit_index"][row_index]),
            int(columns["layer_index"][row_index]),
            int(columns["head_index"][row_index]),
        )
        if key in lookup:
            raise ValueError(f"duplicate local Oracle row {key}")
        lookup[key] = {
            name: float(columns[name][row_index])
            for name in LOCAL_COLUMNS[4:]
        }
    return lookup


def align_record(
    record: dict[str, Any],
    lookup: dict[tuple[str, int, int, int], dict[str, float]],
) -> dict[str, Any]:
    intervention = record["intervention"]
    metrics = []
    for head_index in intervention["head_indices"]:
        key = (
            record["request_key"],
            int(intervention["dit_index"]),
            int(intervention["layer_index"]),
            int(head_index),
        )
        if key not in lookup:
            raise KeyError(f"missing local Oracle row {key}")
        metrics.append(lookup[key])

    local_oracle = np.asarray(
        [metric["oracle_min_keep_ratio"] for metric in metrics], dtype=np.float64
    )
    video_oracle = np.asarray(
        [metric["video_oracle_min_keep_ratio"] for metric in metrics],
        dtype=np.float64,
    )
    action_oracle = np.asarray(
        [metric["action_oracle_min_keep_ratio"] for metric in metrics],
        dtype=np.float64,
    )
    return {
        **record,
        "local_oracle_keep_mean": float(local_oracle.mean()),
        "local_oracle_keep_max": float(local_oracle.max()),
        "local_oracle_dense_fraction": float(np.mean(local_oracle >= 1.0)),
        "local_video_oracle_keep_mean": float(video_oracle.mean()),
        "local_action_oracle_keep_mean": float(action_oracle.mean()),
        "downstream_cosine_failure": bool(record["action_cosine"] < 0.999),
        "downstream_l2_failure": bool(record["action_relative_l2"] > 0.05),
    }


def alignment_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    local = [record["local_oracle_keep_mean"] for record in records]
    downstream = [record["action_relative_l2"] for record in records]
    by_request = {}
    for request_key in sorted({record["request_key"] for record in records}):
        request_records = [
            record for record in records if record["request_key"] == request_key
        ]
        by_request[request_key] = {
            "candidates": len(request_records),
            "pearson": finite_correlation(
                [record["local_oracle_keep_mean"] for record in request_records],
                [record["action_relative_l2"] for record in request_records],
            ),
            "spearman": finite_correlation(
                [record["local_oracle_keep_mean"] for record in request_records],
                [record["action_relative_l2"] for record in request_records],
                rank=True,
            ),
        }
    request_spearman = [
        summary["spearman"]
        for summary in by_request.values()
        if summary["spearman"] is not None
    ]
    return {
        "rows": len(records),
        "requests": len(by_request),
        "pearson_local_keep_vs_downstream_l2": finite_correlation(
            local, downstream
        ),
        "spearman_local_keep_vs_downstream_l2": finite_correlation(
            local, downstream, rank=True
        ),
        "mean_within_request_spearman": (
            float(np.mean(request_spearman)) if request_spearman else None
        ),
        "cosine_failures": sum(
            record["downstream_cosine_failure"] for record in records
        ),
        "relative_l2_failures": sum(
            record["downstream_l2_failure"] for record in records
        ),
        "by_request": by_request,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Align same-noise downstream action interventions with the q32 "
            "local per-head Oracle budgets."
        )
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--m1-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enriched-jsonl", type=Path)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.records.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("downstream record file is empty")
    request_keys = sorted({record["request_key"] for record in records})
    table = pq.read_table(
        args.m1_table,
        columns=list(LOCAL_COLUMNS),
        filters=[("request_key", "in", request_keys)],
    )
    lookup = build_local_lookup(table)
    enriched = [align_record(record, lookup) for record in records]
    summary = alignment_summary(enriched)
    report = {
        "records": str(args.records),
        "m1_table": str(args.m1_table),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    enriched_path = args.enriched_jsonl or args.output.with_suffix(".jsonl")
    enriched_path.write_text(
        "".join(json.dumps(record) + "\n" for record in enriched)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
