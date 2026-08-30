import json
from pathlib import Path

from benchmarks.analyze_oracle_query_convergence import compare_oracle_captures


def _record(video_queries: int, budgets: list[float], mass_shift: float = 0.0) -> dict:
    metric_row = [[0.9 + mass_shift, 0.8 + mass_shift]]
    video = {
        "mass_mean": metric_row,
        "mass_p05": metric_row,
        "output_cosine_mean": metric_row,
        "output_cosine_p05": metric_row,
        "output_relative_l2_mean": metric_row,
        "output_relative_l2_p95": metric_row,
        "top_p_token_count_mean": metric_row,
        "top_p_token_count_p95": metric_row,
        "normalized_entropy_mean": [0.5 + mass_shift, 0.4 + mass_shift],
        "max_attention_mass_mean": [0.2 + mass_shift, 0.1 + mass_shift],
    }
    action = dict(video)
    return {
        "dit_index": 0,
        "layer_index": 0,
        "cfg_branch": "conditional",
        "sample_metadata": {"request_key": "sample"},
        "num_sampled_video_queries": video_queries,
        "video_oracle_min_keep_ratio": budgets,
        "action_oracle_min_keep_ratio": [0.5, 1.0],
        "video": video,
        "action": action,
    }


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n")


def test_convergence_reports_false_sparse_and_metric_error(tmp_path: Path):
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(reference, _record(32, [0.5, 1.0]))
    _write(candidate, _record(8, [0.25, 1.0], mass_shift=0.1))

    result = compare_oracle_captures(reference, candidate)

    assert result["head_label_count"] == 2
    assert result["label_agreement"] == 0.5
    assert result["false_sparse_rate"] == 0.5
    assert result["overconservative_rate"] == 0.0
    assert result["mean_absolute_budget_error"] == 0.125
    assert result["action_statistics_exact"] is False
    assert abs(result["video_metric_mean_absolute_error"]["mass_mean"] - 0.1) < 1e-9
