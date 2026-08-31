import numpy as np
import torch

from benchmarks.analyze_vv_linear_extrapolation import (
    extrapolation_metrics,
    fit_linear_alpha,
    select_sentinel_threshold,
)


def test_fit_linear_alpha_recovers_exact_rows():
    previous_two = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    previous = torch.tensor([[2.0, 4.0], [3.0, 6.0]])
    current = torch.tensor([[3.5, 7.0], [3.5, 7.0]])

    alpha = fit_linear_alpha(current, previous, previous_two)

    assert torch.allclose(alpha, torch.tensor([1.5, 0.5]))
    cosine, relative_l2 = extrapolation_metrics(
        current,
        previous,
        previous_two,
        alpha,
    )
    assert torch.allclose(cosine, torch.ones_like(cosine))
    assert torch.allclose(relative_l2, torch.zeros_like(relative_l2))


def test_fit_linear_alpha_clamps_unstable_direction():
    previous_two = torch.tensor([[0.0, 0.0]])
    previous = torch.tensor([[1.0, 1.0]])
    current = torch.tensor([[5.0, 5.0]])

    assert fit_linear_alpha(current, previous, previous_two).item() == 2.0


def test_sentinel_threshold_maximizes_safe_coverage():
    previous_error = np.asarray([0.01, 0.02, 0.03, 0.04])
    current_safe = np.asarray([True, True, False, True])
    eligible = np.ones(4, dtype=bool)

    result = select_sentinel_threshold(
        previous_error,
        current_safe,
        eligible,
        maximum_false_extrapolation_rate=0.0,
    )

    assert result["threshold"] == 0.02
    assert result["selected"] == 2
    assert result["unsafe"] == 0
