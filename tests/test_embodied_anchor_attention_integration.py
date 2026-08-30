import os
import sys
import types

import torch
import torch.nn as nn

from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import AnchorSparseConfig


def _load_attention_module():
    """Load the attention file without requiring the full deployment environment."""

    os.environ["ATTENTION_BACKEND"] = "torch"
    try:
        import diffusers  # noqa: F401
    except ModuleNotFoundError:
        configuration_utils = types.ModuleType("diffusers.configuration_utils")
        modeling_utils = types.ModuleType("diffusers.models.modeling_utils")

        class ConfigMixin:
            pass

        class ModelMixin(nn.Module):
            pass

        def register_to_config(function):
            return function

        configuration_utils.ConfigMixin = ConfigMixin
        configuration_utils.register_to_config = register_to_config
        modeling_utils.ModelMixin = ModelMixin
        sys.modules["diffusers"] = types.ModuleType("diffusers")
        sys.modules["diffusers.configuration_utils"] = configuration_utils
        sys.modules["diffusers.models"] = types.ModuleType("diffusers.models")
        sys.modules["diffusers.models.modeling_utils"] = modeling_utils

    import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as module

    # This test exercises KV routing and attention wiring, not RoPE indexing.
    module.causal_rope_action_apply = lambda x, **kwargs: x
    return module


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(1, 7, 8)
    cache = torch.randn(2, 1, 8, 2, 4)
    freqs = torch.ones(32, 2, dtype=torch.complex64)
    return x, cache, freqs


def test_cached_route_is_reused_by_real_kv_attention_path() -> None:
    module = _load_attention_module()
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=config,
    )
    x, cache, freqs = _inputs()

    output, updated_cache, route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    cached_output, _, reused_route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        anchor_route_indices=route,
    )

    assert output.shape == (1, 7, 8)
    assert updated_cache.shape == (2, 1, 12, 2, 4)
    assert route.shape == (1, 8)
    assert reused_route is route
    assert torch.equal(cached_output, output)


def test_state_register_is_dense_but_not_used_as_router_query(monkeypatch) -> None:
    module = _load_attention_module()
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=config,
    )
    x, cache, freqs = _inputs()
    captured = {}
    original_router = module.route_action_conditioned_video_keys

    def capture_router(*, action_query, video_key, config):
        captured["query_tokens"] = action_query.shape[1]
        return original_router(
            action_query=action_query,
            video_key=video_key,
            config=config,
        )

    monkeypatch.setattr(module, "route_action_conditioned_video_keys", capture_router)
    output, _, _ = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )

    assert output.shape == (1, 7, 8)
    assert captured["query_tokens"] == 2


def test_full_budget_uses_exact_original_dense_path() -> None:
    module = _load_attention_module()
    dense = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    full_budget = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=AnchorSparseConfig(
            frame_seqlen=4,
            grid_height=2,
            grid_width=2,
            keep_ratio=1.0,
            recent_dense_frames=1,
            smooth_radius=0,
        ),
    )
    full_budget.load_state_dict(dense.state_dict())
    x, cache, freqs = _inputs()

    dense_output, dense_cache, dense_route = dense(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    full_output, full_cache, full_route = full_budget(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )

    assert dense_route is None
    assert full_route is None
    assert torch.equal(full_output, dense_output)
    assert torch.equal(full_cache, dense_cache)


def test_post_checkpoint_configuration_updates_every_block() -> None:
    module = _load_attention_module()
    model = module.CausalWanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=880,
        text_len=4,
        in_dim=2,
        dim=8,
        ffn_dim=16,
        freq_dim=8,
        text_dim=8,
        out_dim=2,
        num_heads=2,
        num_layers=2,
        action_dim=2,
        max_state_dim=4,
        hidden_size=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )

    model.configure_anchor_sparse_attention(
        enabled=True,
        keep_ratio=0.2,
        current_keep_ratio=0.25,
        dense_prefix_layers=1,
        dense_suffix_layers=0,
        propagate_radius=1,
        propagate_every=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
        record_diagnostics=True,
    )

    assert model.anchor_sparse_enabled
    assert model.anchor_sparse_config is not None
    assert model.anchor_sparse_config.anchor_tokens_per_frame == 176
    assert model.anchor_sparse_config.recent_dense_frames == 2
    assert all(
        block.self_attn.anchor_sparse_config is model.anchor_sparse_config
        for block in model.blocks
    )
    assert model.blocks[0].self_attn.record_anchor_diagnostics
    assert not model.blocks[1].self_attn.record_anchor_diagnostics
    assert not model.blocks[0].sparse_current_compute
    assert model.blocks[1].sparse_current_compute
    assert model.blocks[0].current_propagate_radius == 0
    assert model.blocks[1].current_propagate_radius == 1

    model.configure_anchor_sparse_attention(enabled=False)
    assert model.anchor_sparse_config is None
    assert all(block.self_attn.anchor_sparse_config is None for block in model.blocks)
    assert not any(block.sparse_current_compute for block in model.blocks)
    assert not any(block.current_propagate_radius for block in model.blocks)


def test_sparse_current_update_keeps_unselected_video_tokens_and_updates_registers() -> None:
    module = _load_attention_module()
    x = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    e = tuple(torch.zeros(1, 7, 1, 1) for _ in range(6))
    current_video_indices = torch.tensor([[1, 3]])

    def add_ten(selected_x, selected_e):
        assert selected_x.shape == (1, 5, 1)
        assert all(part.shape == (1, 5, 1, 1) for part in selected_e)
        return selected_x + 10

    updated = module.sparse_current_token_update(
        x=x,
        e=e,
        current_video_indices=current_video_indices,
        action_register_length=3,
        anchor_sparse_config=AnchorSparseConfig(
            frame_seqlen=4,
            grid_height=2,
            grid_width=2,
            keep_ratio=0.5,
            recent_dense_frames=0,
            smooth_radius=0,
        ),
        propagate_radius=0,
        update_fn=add_ten,
    )

    assert torch.equal(updated[0, :, 0], torch.tensor([0.0, 11.0, 2.0, 13.0, 14.0, 15.0, 16.0]))
