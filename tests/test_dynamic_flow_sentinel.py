import pytest
import torch

from groot.vla.model.dreamzero.modules.dynamic_flow_sentinel import (
    FlowSentinelConfig,
    flow_sentinel_metrics,
    linear_flow_prediction,
)


def test_linear_flow_prediction_respects_nonuniform_timestep_spacing():
    previous_two = torch.tensor([[1.0, 2.0]])
    previous = torch.tensor([[3.0, 6.0]])

    predicted, alpha = linear_flow_prediction(
        previous,
        previous_two,
        current_timestep=700,
        previous_timestep=800,
        previous_two_timestep=1000,
    )

    assert alpha == pytest.approx(0.5)
    assert torch.equal(predicted, torch.tensor([[4.0, 8.0]]))


def test_exact_linear_flow_does_not_trigger_sentinel():
    previous_two = torch.tensor([[1.0, 2.0]])
    previous = torch.tensor([[3.0, 6.0]])
    current = torch.tensor([[4.0, 8.0]])
    config = FlowSentinelConfig(minimum_cosine=0.999, maximum_relative_l2=0.01)

    metrics = flow_sentinel_metrics(
        current,
        previous,
        previous_two,
        current_timestep=700,
        previous_timestep=800,
        previous_two_timestep=1000,
    )

    assert metrics.cosine == pytest.approx(1.0)
    assert metrics.relative_l2 == pytest.approx(0.0)
    assert not metrics.triggered(config)


def test_large_flow_residual_triggers_sentinel():
    metrics = flow_sentinel_metrics(
        torch.tensor([[8.0, -4.0]]),
        torch.tensor([[3.0, 6.0]]),
        torch.tensor([[1.0, 2.0]]),
        current_timestep=700,
        previous_timestep=800,
        previous_two_timestep=1000,
    )

    assert metrics.triggered(
        FlowSentinelConfig(minimum_cosine=0.99, maximum_relative_l2=0.25)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_cosine": -1.1},
        {"minimum_cosine": 1.1},
        {"maximum_relative_l2": -0.1},
        {"start_dit_index": 1},
    ],
)
def test_flow_sentinel_config_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        FlowSentinelConfig(**kwargs)
