"""Compare bounded video-query Oracle labels against a larger reference sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _records(path: Path) -> dict[tuple[int, int, str], dict]:
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (
            int(record["dit_index"]),
            int(record["layer_index"]),
            record["cfg_branch"],
        )
        if key in records:
            raise ValueError(f"Duplicate Oracle record key {key} in {path}")
        records[key] = record
    return records


def _flatten(values):
    for row in values:
        yield from row


def compare_oracle_captures(reference_path: Path, candidate_path: Path) -> dict:
    reference = _records(reference_path)
    candidate = _records(candidate_path)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(f"Capture keys differ: missing={missing[:5]}, extra={extra[:5]}")

    label_total = 0
    label_matches = 0
    false_sparse = 0
    overconservative = 0
    absolute_budget_error = []
    metric_errors: dict[str, list[float]] = {
        "mass_mean": [],
        "mass_p05": [],
        "output_cosine_mean": [],
        "output_cosine_p05": [],
        "output_relative_l2_mean": [],
        "output_relative_l2_p95": [],
        "top_p_token_count_mean": [],
        "top_p_token_count_p95": [],
        "normalized_entropy_mean": [],
        "max_attention_mass_mean": [],
    }

    for key in sorted(reference):
        ref_record = reference[key]
        candidate_record = candidate[key]
        if ref_record["sample_metadata"]["request_key"] != candidate_record[
            "sample_metadata"
        ]["request_key"]:
            raise ValueError(f"Request mismatch at {key}")
        ref_budget = ref_record["video_oracle_min_keep_ratio"]
        candidate_budget = candidate_record["video_oracle_min_keep_ratio"]
        if len(ref_budget) != len(candidate_budget):
            raise ValueError(f"Head count mismatch at {key}")
        for ref_value, candidate_value in zip(ref_budget, candidate_budget):
            label_total += 1
            label_matches += ref_value == candidate_value
            false_sparse += candidate_value < ref_value
            overconservative += candidate_value > ref_value
            absolute_budget_error.append(abs(candidate_value - ref_value))

        for metric in metric_errors:
            ref_values = ref_record["video"][metric]
            candidate_values = candidate_record["video"][metric]
            ref_flat = list(_flatten(ref_values)) if isinstance(ref_values[0], list) else ref_values
            candidate_flat = (
                list(_flatten(candidate_values))
                if isinstance(candidate_values[0], list)
                else candidate_values
            )
            if len(ref_flat) != len(candidate_flat):
                raise ValueError(f"Metric shape mismatch for {metric} at {key}")
            metric_errors[metric].extend(
                abs(candidate_value - ref_value)
                for ref_value, candidate_value in zip(ref_flat, candidate_flat)
            )

    action_exact = all(
        reference[key][field] == candidate[key][field]
        for key in reference
        for field in (
            "action_oracle_min_keep_ratio",
            "action",
        )
    )
    result = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "request_key": next(iter(reference.values()))["sample_metadata"]["request_key"],
        "reference_video_queries": next(iter(reference.values()))[
            "num_sampled_video_queries"
        ],
        "candidate_video_queries": next(iter(candidate.values()))[
            "num_sampled_video_queries"
        ],
        "record_count": len(reference),
        "head_label_count": label_total,
        "label_agreement": label_matches / label_total,
        "false_sparse_rate": false_sparse / label_total,
        "overconservative_rate": overconservative / label_total,
        "mean_absolute_budget_error": statistics.fmean(absolute_budget_error),
        "max_absolute_budget_error": max(absolute_budget_error),
        "action_statistics_exact": action_exact,
        "video_metric_mean_absolute_error": {
            metric: statistics.fmean(errors)
            for metric, errors in metric_errors.items()
        },
        "video_metric_max_absolute_error": {
            metric: max(errors) for metric, errors in metric_errors.items()
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_oracle_captures(args.reference, args.candidate)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
