import json

import numpy as np

from benchmarks.analyze_sparse_flow_sentinel import (
    TRACE_PREFIX,
    align_target_traces,
    choose_thresholds,
    evaluate_thresholds,
    parse_flow_sentinel_traces,
)


def _trace(marker: float) -> list[dict[str, float | int | bool]]:
    return [
        {
            "dit_index": dit_index,
            "checked": dit_index >= 2,
            "cosine": marker,
            "relative_l2": marker,
        }
        for dit_index in range(8)
    ]


def test_parse_flow_sentinel_traces_accepts_prefixed_server_lines(tmp_path):
    path = tmp_path / "server.log"
    expected = _trace(0.5)
    path.write_text(
        "unrelated log line\n"
        + f"[rank0] {TRACE_PREFIX}{json.dumps(expected)}"
        + "concurrent worker log suffix\n"
    )

    assert parse_flow_sentinel_traces(path) == [expected]


def test_align_target_traces_uses_last_call_after_request_history():
    report = {
        "records": [
            {
                "phase": "measured",
                "history_request_count": 1,
                "request_key": "request-0",
            },
            {
                "phase": "measured",
                "history_request_count": 2,
                "request_key": "request-1",
            },
        ]
    }
    stale = [_trace(-1.0)]
    calls = [
        _trace(0.0),
        _trace(0.1),
        _trace(1.0),
        _trace(1.1),
        _trace(1.2),
    ]

    aligned = align_target_traces(report, stale + calls)

    assert aligned[0][1][0]["cosine"] == 0.1
    assert aligned[1][1][0]["cosine"] == 1.2


def test_choose_thresholds_detects_all_unsafe_validation_requests():
    cosine = np.asarray(
        [
            [0.9998, 0.9997],
            [0.9800, 0.9995],
            [0.9994, 0.9993],
        ]
    )
    relative_l2 = np.asarray(
        [
            [0.010, 0.020],
            [0.015, 0.020],
            [0.400, 0.010],
        ]
    )
    quality_pass = np.asarray([True, False, False])

    selected = choose_thresholds(cosine, relative_l2, quality_pass)

    assert selected["unsafe_request_count"] == 2
    assert selected["detected_unsafe_request_count"] == 2
    assert selected["false_positive_request_count"] == 0
    assert selected["triggered_request_count"] == 2
    assert selected["triggered_step_count"] == 2
    assert selected["request_trigger"] == [False, True, True]


def test_evaluate_thresholds_reports_test_false_negatives_without_retuning():
    evaluated = evaluate_thresholds(
        np.asarray([[0.99], [0.80], [0.95]]),
        np.asarray([[0.01], [0.10], [0.20]]),
        np.asarray([True, False, False]),
        minimum_cosine=0.90,
        maximum_relative_l2=0.15,
    )

    assert evaluated["triggered_request_count"] == 2
    assert evaluated["detected_unsafe_request_count"] == 2
    assert evaluated["false_negative_request_count"] == 0
    assert evaluated["false_positive_request_count"] == 0
