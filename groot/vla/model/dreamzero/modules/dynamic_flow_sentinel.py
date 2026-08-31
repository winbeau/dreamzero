"""Online action-flow sentinel for dynamic sparse DreamZero execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowSentinelConfig:
    """Thresholds for validating a sparse DiT output against linear history."""

    minimum_cosine: float = 0.99
    maximum_relative_l2: float = 0.25
    start_dit_index: int = 2
    rerun_dense: bool = True

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_cosine <= 1.0:
            raise ValueError("minimum_cosine must lie in [-1, 1]")
        if self.maximum_relative_l2 < 0.0:
            raise ValueError("maximum_relative_l2 must be non-negative")
        if self.start_dit_index < 2:
            raise ValueError("flow sentinel requires two real historical DiT outputs")


@dataclass(frozen=True)
class FlowSentinelMetrics:
    """Prediction error for one real DiT evaluation."""

    cosine: float
    relative_l2: float
    alpha: float

    def triggered(self, config: FlowSentinelConfig) -> bool:
        return bool(
            self.cosine < config.minimum_cosine
            or self.relative_l2 > config.maximum_relative_l2
        )


def linear_flow_prediction(
    previous: torch.Tensor,
    previous_two: torch.Tensor,
    *,
    current_timestep: int | float | torch.Tensor,
    previous_timestep: int | float | torch.Tensor,
    previous_two_timestep: int | float | torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Extrapolate a flow using the actual nonuniform scheduler spacing."""

    if previous.shape != previous_two.shape:
        raise ValueError("flow history tensors must share a shape")
    current_value = float(torch.as_tensor(current_timestep).flatten()[0].item())
    previous_value = float(torch.as_tensor(previous_timestep).flatten()[0].item())
    previous_two_value = float(
        torch.as_tensor(previous_two_timestep).flatten()[0].item()
    )
    denominator = previous_value - previous_two_value
    alpha = 1.0 if abs(denominator) <= 1e-12 else (
        (current_value - previous_value) / denominator
    )
    # Scheduler spacings are positive in ratio even though timesteps descend.
    # The clamp prevents a malformed schedule from producing an unbounded
    # sentinel reference.
    alpha = float(min(max(alpha, 0.0), 2.0))
    return previous + alpha * (previous - previous_two), alpha


def flow_sentinel_metrics(
    current: torch.Tensor,
    previous: torch.Tensor,
    previous_two: torch.Tensor,
    *,
    current_timestep: int | float | torch.Tensor,
    previous_timestep: int | float | torch.Tensor,
    previous_two_timestep: int | float | torch.Tensor,
) -> FlowSentinelMetrics:
    """Measure current action-flow agreement with a two-step extrapolation."""

    if current.shape != previous.shape or current.shape != previous_two.shape:
        raise ValueError("current and historical flow tensors must share a shape")
    predicted, alpha = linear_flow_prediction(
        previous.float(),
        previous_two.float(),
        current_timestep=current_timestep,
        previous_timestep=previous_timestep,
        previous_two_timestep=previous_two_timestep,
    )
    current_flat = current.float().flatten(1)
    predicted_flat = predicted.flatten(1)
    cosine = F.cosine_similarity(current_flat, predicted_flat, dim=1).amin()
    relative_l2 = (
        torch.linalg.vector_norm(current_flat - predicted_flat, dim=1)
        / torch.linalg.vector_norm(predicted_flat, dim=1).clamp_min(1e-12)
    ).amax()
    return FlowSentinelMetrics(
        cosine=float(cosine.item()),
        relative_l2=float(relative_l2.item()),
        alpha=alpha,
    )
