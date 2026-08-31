import numpy as np
import pytest
import torch

from groot.vla.model.dreamzero.modules.dynamic_m1_observation import (
    PACKED_M1_OBSERVATION_SCHEMA,
    load_packed_m1_observations,
    save_packed_m1_observations,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_packed_observer import (
    PackedM1CausalObserver,
    route_proxy_metrics,
)


def _output(scale: float, *, layer_index: int) -> torch.Tensor:
    base = torch.arange(1, 1 + 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
    return base * scale + layer_index


def _record_complete_step(
    observer: PackedM1CausalObserver,
    dit_index: int,
    *,
    conditional_scale: float,
    unconditional_scale: float,
    route_scores: torch.Tensor,
):
    observer.begin_step(dit_index)
    for branch, scale in (
        ("conditional", conditional_scale),
        ("unconditional", unconditional_scale),
    ):
        for layer_index in range(2):
            observer.observe_action_output(
                layer_index=layer_index,
                cfg_branch=branch,
                action_output=_output(scale, layer_index=layer_index),
            )
        observer.observe_route_scores(route_scores, cfg_branch=branch)
    return observer.finish_step()


def test_route_proxy_metrics_are_normalized_and_causal() -> None:
    current = torch.zeros((1, 1, 4))
    turnover, entropy, max_mass = route_proxy_metrics(
        current,
        None,
        support_ratio=0.5,
    )

    assert torch.isnan(turnover).all()
    assert entropy.item() == pytest.approx(1.0)
    assert max_mass.item() == pytest.approx(0.25)

    _, peaked_entropy, peaked_max_mass = route_proxy_metrics(
        torch.tensor([[[3.0, 2.0, 0.0, 0.0]]]),
        None,
        support_ratio=0.5,
    )
    assert peaked_entropy.item() < 0.9
    assert peaked_max_mass.item() > 0.5

    previous = torch.tensor([[[3.0, 2.0, 0.0, 0.0]]])
    current = torch.tensor([[[0.0, 0.0, 3.0, 2.0]]])
    turnover, _, _ = route_proxy_metrics(current, previous, support_ratio=0.5)
    assert turnover.item() == pytest.approx(1.0)


def test_packed_observer_emits_schema_specific_per_head_history() -> None:
    observer = PackedM1CausalObserver(num_layers=2, num_heads=2, support_ratio=0.5)
    observer.begin_request()

    first = _record_complete_step(
        observer,
        0,
        conditional_scale=1.0,
        unconditional_scale=0.5,
        route_scores=torch.tensor([[[3.0, 2.0, 0.0, 0.0]]]),
    )
    assert first is not None
    assert first.schema == PACKED_M1_OBSERVATION_SCHEMA
    assert first.shape == (2, 2)
    assert np.isnan(first.metric("packed_action_output_change_relative_l2_max")).all()
    assert np.isnan(first.metric("packed_route_support_turnover_max")).all()
    assert np.all(first.metric("packed_cfg_disagreement_relative_l2") > 0.0)

    second = _record_complete_step(
        observer,
        1,
        conditional_scale=2.0,
        unconditional_scale=1.0,
        route_scores=torch.tensor([[[0.0, 0.0, 3.0, 2.0]]]),
    )
    assert second is not None
    assert np.all(second.metric("packed_action_output_change_relative_l2_max") > 0.0)
    assert np.allclose(
        second.metric("packed_action_output_change_cosine_min"),
        1.0,
        atol=3e-4,
    )
    assert np.allclose(
        second.metric("packed_route_support_turnover_max"),
        1.0,
    )


def test_packed_observer_incomplete_or_invalid_step_fails_closed() -> None:
    observer = PackedM1CausalObserver(num_layers=2, num_heads=2)
    observer.begin_request()
    observer.begin_step(0)
    observer.observe_action_output(
        layer_index=0,
        cfg_branch="conditional",
        action_output=_output(1.0, layer_index=0),
    )

    assert observer.finish_step() is None
    with pytest.raises(RuntimeError, match="begin_step"):
        observer.observe_action_output(
            layer_index=0,
            cfg_branch="conditional",
            action_output=_output(1.0, layer_index=0),
        )


def test_packed_observer_joins_cfg_branches_split_across_ip_ranks(monkeypatch):
    observer = PackedM1CausalObserver(num_layers=2, num_heads=2, support_ratio=0.5)
    observer.begin_request()
    observer.begin_step(0)
    for layer_index in range(2):
        observer.observe_action_output(
            layer_index=layer_index,
            cfg_branch="conditional",
            action_output=_output(1.0, layer_index=layer_index),
        )
    observer.observe_route_scores(
        torch.tensor([[[3.0, 2.0, 0.0, 0.0]]]),
        cfg_branch="conditional",
    )

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor):
        if tensor.ndim == 1:
            tensor[1] = 1.0
        elif tensor.shape[-1] == 6:
            tensor[1].copy_(tensor[0] * 0.5)
        else:
            tensor[1].copy_(torch.flip(tensor[0], dims=(-1,)))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    observation = observer.finish_step()

    assert observation is not None
    assert np.all(observation.metric("packed_cfg_disagreement_relative_l2") > 0.0)
    assert np.isfinite(
        observation.metric("packed_route_normalized_entropy_mean")
    ).all()


def test_packed_observer_dense_rerun_replaces_sparse_signature() -> None:
    observer = PackedM1CausalObserver(
        num_layers=1,
        num_heads=2,
        cfg_branches=("conditional",),
    )
    observer.begin_request()
    observer.begin_step(0)
    observer.observe_action_output(
        layer_index=0,
        cfg_branch="conditional",
        action_output=_output(1.0, layer_index=0),
    )
    observer.observe_action_output(
        layer_index=0,
        cfg_branch="conditional",
        action_output=_output(2.0, layer_index=0),
    )
    first = observer.finish_step()
    assert first is not None

    observer.begin_step(1)
    observer.observe_action_output(
        layer_index=0,
        cfg_branch="conditional",
        action_output=_output(2.0, layer_index=0),
    )
    second = observer.finish_step()
    assert second is not None
    assert np.allclose(
        second.metric("packed_action_output_change_relative_l2_max"),
        0.0,
    )


def test_packed_observation_artifact_round_trip(tmp_path) -> None:
    observer = PackedM1CausalObserver(
        num_layers=2,
        num_heads=2,
        cfg_branches=("conditional",),
    )
    observer.begin_request()
    observer.begin_step(0)
    for layer_index in range(2):
        observer.observe_action_output(
            layer_index=layer_index,
            cfg_branch="conditional",
            action_output=_output(1.0, layer_index=layer_index),
        )
    observation = observer.finish_step()
    assert observation is not None
    path = tmp_path / "proxy.npz"

    save_packed_m1_observations(
        path,
        (observation,),
        request_metadata={"request_key": "test-0", "state_l2": 1.25},
    )
    restored, metadata = load_packed_m1_observations(path)

    assert metadata == {"request_key": "test-0", "state_l2": 1.25}
    assert restored[0] is not None
    assert restored[0].schema == PACKED_M1_OBSERVATION_SCHEMA
    for name in observation.metrics:
        assert np.allclose(
            restored[0].metric(name),
            observation.metric(name),
            equal_nan=True,
        )
