"""Low-overhead causal observations for a retrained Packed-path M1.

This observer intentionally records only signals already produced by the
Packed executor: action/state-register attention outputs and the layer-zero
anchor-route scores.  It never materializes Dense video attention.  Its
metrics use a distinct schema and must not be consumed by a bundle trained on
the offline Dense-Oracle feature semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from groot.vla.model.dreamzero.modules.dynamic_m1_observation import (
    PACKED_M1_OBSERVATION_SCHEMA,
    M1CausalObservation,
)


def _relative_l2(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    delta = torch.linalg.vector_norm(current - reference, dim=-1)
    denominator = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-12)
    return delta / denominator


def _signature(output: torch.Tensor) -> torch.Tensor:
    """Reduce Dense registers to stable mean/RMS Head signatures."""

    if output.ndim != 4:
        raise ValueError("Packed M1 action output must have shape [B, A, H, D]")
    if not all(output.shape):
        raise ValueError("Packed M1 action output dimensions must be non-empty")
    output = output.detach().float()
    mean = output.mean(dim=(0, 1))
    root_mean_square = output.square().mean(dim=(0, 1)).sqrt()
    return torch.cat((mean, root_mean_square), dim=-1)


def route_proxy_metrics(
    scores: torch.Tensor,
    previous_scores: torch.Tensor | None,
    *,
    support_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return route turnover, entropy, and max mass for each request sample."""

    if scores.ndim != 3 or not all(scores.shape):
        raise ValueError("Packed M1 route scores must have shape [B, F, P]")
    if not 0.0 < support_ratio <= 1.0:
        raise ValueError("Packed M1 route support ratio must lie in (0, 1]")
    current = scores.detach().float()
    centered = current - current.mean(dim=-1, keepdim=True)
    standardized = centered / current.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-6)
    probabilities = torch.softmax(standardized, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / torch.log(
        torch.tensor(
            probabilities.shape[-1],
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
    ).clamp_min(1e-12)
    entropy = entropy.mean(dim=-1)
    max_mass = probabilities.amax(dim=-1).mean(dim=-1)
    support_count = max(1, round(current.shape[-1] * support_ratio))
    current_support = torch.topk(
        current,
        support_count,
        dim=-1,
        sorted=False,
    ).indices
    if previous_scores is None:
        turnover = torch.full_like(entropy, torch.nan)
    else:
        previous = previous_scores.detach().float()
        if previous.shape != current.shape:
            raise ValueError("Packed M1 adjacent route-score shapes must align")
        previous_support = torch.topk(
            previous,
            support_count,
            dim=-1,
            sorted=False,
        ).indices
        overlap = (
            (current_support[..., :, None] == previous_support[..., None, :])
            .any(dim=-1)
            .float()
            .mean(dim=-1)
        )
        turnover = (1.0 - overlap).amax(dim=-1)
    return turnover, entropy, max_mass


@dataclass
class _StepAccumulator:
    dit_index: int
    signatures: dict[str, dict[int, torch.Tensor]]
    route_scores: dict[str, torch.Tensor]


class PackedM1CausalObserver:
    """Collect per-layer/Head proxy signals during one eight-DiT request."""

    def __init__(
        self,
        *,
        num_layers: int = 40,
        num_heads: int = 40,
        support_ratio: float = 0.20,
        cfg_branches: tuple[str, ...] = ("conditional", "unconditional"),
    ) -> None:
        if min(num_layers, num_heads) <= 0:
            raise ValueError("Packed M1 observer geometry must be positive")
        if not 0.0 < support_ratio <= 1.0:
            raise ValueError("Packed M1 support ratio must lie in (0, 1]")
        if not cfg_branches or len(set(cfg_branches)) != len(cfg_branches):
            raise ValueError("Packed M1 CFG branches must be non-empty and unique")
        if any(
            branch not in {"conditional", "unconditional"} for branch in cfg_branches
        ):
            raise ValueError("Packed M1 observer received an invalid CFG branch")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.support_ratio = support_ratio
        self.cfg_branches = cfg_branches
        self._active = False
        self._step: _StepAccumulator | None = None
        self._previous_action_signatures: dict[str, torch.Tensor] = {}
        self._previous_route_scores: dict[str, torch.Tensor] = {}

    def begin_request(self) -> None:
        self._active = True
        self._step = None
        self._previous_action_signatures.clear()
        self._previous_route_scores.clear()

    @property
    def step_active(self) -> bool:
        return self._step is not None

    def end_request(self) -> None:
        self._active = False
        self._step = None
        self._previous_action_signatures.clear()
        self._previous_route_scores.clear()

    def begin_step(self, dit_index: int) -> None:
        if not self._active:
            raise RuntimeError("Packed M1 begin_request must precede begin_step")
        if self._step is not None:
            raise RuntimeError("Packed M1 previous step has not been finalized")
        if dit_index < 0:
            raise ValueError("Packed M1 DiT index must be non-negative")
        self._step = _StepAccumulator(
            dit_index,
            {branch: {} for branch in self.cfg_branches},
            {},
        )

    def observe_action_output(
        self,
        *,
        layer_index: int,
        cfg_branch: str,
        action_output: torch.Tensor,
    ) -> None:
        step = self._step
        if step is None:
            raise RuntimeError("Packed M1 begin_step must precede observation")
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("Packed M1 layer index is outside observer geometry")
        if cfg_branch not in self.cfg_branches:
            raise ValueError("Packed M1 CFG branch is outside observer scope")
        signature = _signature(action_output)
        if signature.shape[0] != self.num_heads:
            raise ValueError("Packed M1 action output Head count is invalid")
        # A Flow sentinel may rerun this same DiT densely.  The last completed
        # invocation is the state consumed by the scheduler, so it replaces
        # the earlier sparse observation.
        step.signatures[cfg_branch][layer_index] = signature

    def observe_route_scores(self, scores: torch.Tensor, *, cfg_branch: str) -> None:
        step = self._step
        if step is None:
            raise RuntimeError("Packed M1 begin_step must precede route observation")
        if cfg_branch not in self.cfg_branches:
            raise ValueError("Packed M1 CFG branch is outside observer scope")
        if scores.ndim != 3 or not all(scores.shape):
            raise ValueError("Packed M1 route scores must have shape [B, F, P]")
        step.route_scores[cfg_branch] = scores.detach()

    def _complete_distributed_cfg_branches(
        self,
        step: _StepAccumulator,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
        """Join conditional/unconditional proxies split across two IP ranks."""

        required_layers = set(range(self.num_layers))
        complete_local = {
            branch: torch.stack(
                [step.signatures[branch][layer] for layer in range(self.num_layers)]
            )
            for branch in self.cfg_branches
            if set(step.signatures[branch]) == required_layers
        }
        if set(complete_local) == set(self.cfg_branches):
            return complete_local, dict(step.route_scores)
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
            or len(complete_local) != 1
        ):
            return None

        exemplar = next(iter(complete_local.values()))
        branch_signatures = exemplar.new_zeros(
            (len(self.cfg_branches), *exemplar.shape)
        )
        branch_presence = exemplar.new_zeros(len(self.cfg_branches))
        for branch, value in complete_local.items():
            branch_index = self.cfg_branches.index(branch)
            branch_signatures[branch_index].copy_(value)
            branch_presence[branch_index] = 1.0
        dist.all_reduce(branch_signatures)
        dist.all_reduce(branch_presence)
        if not torch.equal(
            branch_presence,
            torch.ones_like(branch_presence),
        ):
            return None
        complete_signatures = {
            branch: branch_signatures[index]
            for index, branch in enumerate(self.cfg_branches)
        }

        local_routes = {
            branch: value
            for branch, value in step.route_scores.items()
            if branch in self.cfg_branches
        }
        if len(local_routes) != 1:
            return complete_signatures, {}
        route_exemplar = next(iter(local_routes.values()))
        branch_routes = route_exemplar.new_zeros(
            (len(self.cfg_branches), *route_exemplar.shape)
        )
        route_presence = route_exemplar.new_zeros(len(self.cfg_branches))
        for branch, value in local_routes.items():
            branch_index = self.cfg_branches.index(branch)
            branch_routes[branch_index].copy_(value)
            route_presence[branch_index] = 1.0
        dist.all_reduce(branch_routes)
        dist.all_reduce(route_presence)
        if not torch.equal(route_presence, torch.ones_like(route_presence)):
            return complete_signatures, {}
        return complete_signatures, {
            branch: branch_routes[index]
            for index, branch in enumerate(self.cfg_branches)
        }

    def finish_step(self) -> M1CausalObservation | None:
        step = self._step
        if step is None:
            raise RuntimeError("Packed M1 begin_step must precede finish_step")
        self._step = None
        completed = self._complete_distributed_cfg_branches(step)
        if completed is None:
            return None

        shape = (self.num_layers, self.num_heads)
        current_by_branch, route_scores = completed
        signature_shapes = {tuple(value.shape) for value in current_by_branch.values()}
        if len(signature_shapes) != 1:
            return None
        if all(
            branch in self._previous_action_signatures for branch in self.cfg_branches
        ):
            relative_l2 = (
                torch.stack(
                    [
                        _relative_l2(
                            current_by_branch[branch],
                            self._previous_action_signatures[branch],
                        )
                        for branch in self.cfg_branches
                    ]
                )
                .amax(dim=0)
                .cpu()
            )
            cosine = (
                torch.stack(
                    [
                        F.cosine_similarity(
                            current_by_branch[branch],
                            self._previous_action_signatures[branch],
                            dim=-1,
                        )
                        for branch in self.cfg_branches
                    ]
                )
                .amin(dim=0)
                .cpu()
            )
        else:
            relative_l2 = torch.full(shape, torch.nan, dtype=torch.float32)
            cosine = torch.full(shape, torch.nan, dtype=torch.float32)
        if len(self.cfg_branches) == 2:
            cfg_disagreement = _relative_l2(
                current_by_branch[self.cfg_branches[0]],
                current_by_branch[self.cfg_branches[1]],
            ).cpu()
        else:
            cfg_disagreement = torch.full(shape, torch.nan, dtype=torch.float32)
        signature_norm = (
            torch.stack(
                [
                    torch.linalg.vector_norm(
                        current_by_branch[branch],
                        dim=-1,
                    )
                    for branch in self.cfg_branches
                ]
            )
            .amax(dim=0)
            .cpu()
        )

        if not route_scores:
            route_turnover = torch.tensor(torch.nan)
            route_entropy = torch.tensor(torch.nan)
            route_max_mass = torch.tensor(torch.nan)
        else:
            route_metrics = [
                route_proxy_metrics(
                    route_scores[branch],
                    self._previous_route_scores.get(branch),
                    support_ratio=self.support_ratio,
                )
                for branch in self.cfg_branches
                if branch in route_scores
            ]
            route_turnover = torch.stack(
                [value[0].amax() for value in route_metrics]
            ).amax()
            route_entropy = torch.stack(
                [value[1].mean() for value in route_metrics]
            ).mean()
            route_max_mass = torch.stack(
                [value[2].mean() for value in route_metrics]
            ).mean()
            self._previous_route_scores.update(route_scores)

        self._previous_action_signatures = {
            branch: current.detach() for branch, current in current_by_branch.items()
        }

        def broadcast(value: torch.Tensor) -> torch.Tensor:
            return value.detach().float().cpu().expand(shape).clone()

        metrics = {
            "packed_route_support_turnover_max": broadcast(route_turnover),
            "packed_route_normalized_entropy_mean": broadcast(route_entropy),
            "packed_route_max_mass_mean": broadcast(route_max_mass),
            "packed_action_output_change_relative_l2_max": relative_l2,
            "packed_action_output_change_cosine_min": cosine,
            "packed_cfg_disagreement_relative_l2": cfg_disagreement,
            "packed_action_output_signature_norm": signature_norm,
        }
        return M1CausalObservation(
            dit_index=step.dit_index,
            schema=PACKED_M1_OBSERVATION_SCHEMA,
            metrics={name: value.numpy() for name, value in metrics.items()},
        )
