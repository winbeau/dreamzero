from dataclasses import replace
from typing import Any, TypeAlias

from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorRoute,
    AnchorSparseConfig,
    build_current_video_query_route,
    droid_composite_view_regions,
    gather_sequence_by_index,
    propagate_spatial_anchor_updates,
    route_action_conditioned_video_keys,
    scatter_sequence_by_index,
)
from groot.vla.model.dreamzero.modules.dynamic_packed_sparse import (
    NestedAnchorProfile,
    PackedMiddleState,
    apply_packed_rope,
    build_nested_current_profile,
    build_nested_history_profile,
    gather_packed_rope_frequencies,
    pack_middle_state,
)
from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicDenseActionHistoryTable,
    DynamicPackedHeadGroupBudgetTable,
    DynamicPackedBudgetTable,
    stabilize_current_budgets_for_segments,
)
from groot.vla.model.dreamzero.modules.dynamic_attention_oracle import (
    DownstreamHeadIntervention,
)
from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    rope_action_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention.flex_attention import BlockMask
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist
import os

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"


def action_conditioned_causal_routing_is_eligible(
    current_start_frame: int,
    action_register_length: int | None,
) -> bool:
    """Whether a pass can safely use action-conditioned current-token routing."""

    return current_start_frame > 0 and action_register_length is not None


def sparse_current_token_update(
    x: torch.Tensor,
    e: tuple[torch.Tensor, ...],
    current_video_indices: torch.Tensor,
    action_register_length: int,
    anchor_sparse_config: AnchorSparseConfig,
    propagate_radius: int,
    update_fn,
) -> torch.Tensor:
    """Apply a token-wise block update to anchors plus all action/state tokens.

    The unselected current-video tokens retain the self-attention residual that
    precedes this function.  Action and state registers always remain dense.
    """

    if action_register_length <= 0 or action_register_length >= x.shape[1]:
        raise ValueError("Sparse current-token compute requires a non-empty register suffix")
    video_seq_len = x.shape[1] - action_register_length
    if current_video_indices.shape[0] != x.shape[0]:
        raise ValueError("current_video_indices batch size differs from x")
    compute_indices, register_indices = build_current_compute_indices(
        current_video_indices,
        video_seq_len=video_seq_len,
        action_register_length=action_register_length,
    )
    selected_x = gather_sequence_by_index(x, compute_indices, validate_indices=False)
    selected_e = tuple(
        gather_sequence_by_index(part, compute_indices, validate_indices=False)
        for part in e
    )
    updated = update_fn(selected_x, selected_e)
    if propagate_radius > 0:
        selected_delta = updated - selected_x
        num_anchor_tokens = current_video_indices.shape[1]
        video_delta = propagate_spatial_anchor_updates(
            selected_delta[:, :num_anchor_tokens],
            current_video_indices,
            video_seq_len=video_seq_len,
            config=anchor_sparse_config,
            radius=propagate_radius,
        )
        full_delta = x.new_zeros(x.shape)
        full_delta[:, :video_seq_len] = video_delta
        full_delta = scatter_sequence_by_index(
            full_delta,
            register_indices,
            selected_delta[:, num_anchor_tokens:],
            validate_indices=False,
        )
        return x + full_delta
    return scatter_sequence_by_index(
        x,
        compute_indices,
        updated,
        validate_indices=False,
    )


def build_current_compute_indices(
    current_video_indices: torch.Tensor,
    *,
    video_seq_len: int,
    action_register_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append dense action/state register positions to current-video anchors."""

    if current_video_indices.ndim != 2:
        raise ValueError("current_video_indices must have shape [B, K]")
    if video_seq_len <= 0 or action_register_length <= 0:
        raise ValueError("video and register lengths must be positive")
    register_indices = torch.arange(
        video_seq_len,
        video_seq_len + action_register_length,
        device=current_video_indices.device,
        dtype=torch.long,
    ).expand(current_video_indices.shape[0], -1)
    return torch.cat([current_video_indices, register_indices], dim=1), register_indices


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


def causal_rope_action_apply(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index):
    if ENABLE_TENSORRT:
        return causal_rope_action_apply_no_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)
    else:
        return causal_rope_action_apply_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)


def causal_rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, d = x.shape
    
    # (B, seq_len, n, d) -> (B, seq_len, n, d/2, 2)
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real = x[..., 0] 
    x_imag = x[..., 1] 
    
    # Split freqs into cos and sin components
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos = freqs[..., 0] # Shape: (1, seq_len', 1, d/2)
    freqs_sin = freqs[..., 1] # Shape: (1, seq_len', 1, d/2)
    
    #  Handle the Action/State Register Frequencies
    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        
        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        
        # Combine the action/state tokens for this frame
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
            action_register_length, 1, -1, 2
        )
        
        # Split the new action/state frequencies
        freqs_cos_1d = freqs_1d[..., 0]
        freqs_sin_1d = freqs_1d[..., 1]

        # Append the action/state register sin/cos to the main sequence sin/cos
        freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_sin_1d], dim=0).unsqueeze(0)
    
    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos
    
    x_rotated = torch.stack((x_real_rotated, x_imag_rotated), dim=-1)
    
    return x_rotated.flatten(3)

def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 anchor_sparse_config: AnchorSparseConfig | None = None,
                 record_anchor_diagnostics: bool = False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.anchor_sparse_config = anchor_sparse_config
        self.record_anchor_diagnostics = record_anchor_diagnostics
        self.packed_dense_action_history = False
        self.packed_max_action_current = False
        self.dynamic_oracle_collector: Any | None = None
        self.dynamic_m1_packed_observer: Any | None = None
        self.dynamic_m1_cfg_branch: str | None = None
        self.downstream_head_intervention: DownstreamHeadIntervention | None = None
        self.downstream_intervention_dit_index: int | None = None
        self.downstream_intervention_cfg_branch: str | None = None
        self.downstream_head_intervention_count = 0
        self.layer_index = -1
        self.last_anchor_route: AnchorRoute | None = None
        self._anchor_sparse_history_cache_key: tuple[Any, ...] | None = None
        self._anchor_sparse_history_k: torch.Tensor | None = None
        self._anchor_sparse_history_v: torch.Tensor | None = None
        self._anchor_sparse_history_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._packed_head_index_cache: dict[
            tuple[tuple[int, ...], str, int | None], torch.Tensor
        ] = {}
        self._packed_head_channel_index_cache: dict[
            tuple[tuple[int, ...], str, int | None], torch.Tensor
        ] = {}
        self._packed_head_projection_weight_cache: dict[
            tuple[tuple[int, ...], str, int | None, torch.dtype],
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor,
            ],
        ] = {}
        self._packed_query_position_cache: dict[
            tuple[int, int, int, str, int | None], torch.Tensor
        ] = {}
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)
        self.causal_attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim, causal=True)

    def _observe_dynamic_m1_action_output(
        self,
        output: torch.Tensor,
        *,
        action_register_length: int,
        registers_first: bool,
    ) -> None:
        observer = self.dynamic_m1_packed_observer
        if observer is None:
            return
        branch = self.dynamic_m1_cfg_branch
        if branch is None:
            return
        if output.ndim != 4 or output.shape[2] != self.num_heads:
            raise ValueError("M1 attention output must have shape [B, L, H, D]")
        if not 0 < action_register_length <= output.shape[1]:
            raise ValueError("M1 action/state register length is invalid")
        register_output = (
            output[:, :action_register_length]
            if registers_first
            else output[:, -action_register_length:]
        )
        observer.observe_action_output(
            layer_index=self.layer_index,
            cfg_branch=branch,
            action_output=register_output,
        )

    def _apply_downstream_head_intervention(
        self,
        output: torch.Tensor,
        *,
        action_register_length: int | None,
    ) -> torch.Tensor:
        """Scale selected Dense attention heads before the O projection."""

        intervention = self.downstream_head_intervention
        if intervention is None or not intervention.applies(
            dit_index=self.downstream_intervention_dit_index,
            layer_index=self.layer_index,
            cfg_branch=self.downstream_intervention_cfg_branch,
        ):
            return output
        if output.ndim != 4 or output.shape[2] != self.num_heads:
            raise ValueError("attention output must have shape [B, L, H, D]")
        if intervention.query_scope == "all":
            query_start, query_stop = 0, output.shape[1]
        else:
            if action_register_length is None or not (
                0 < action_register_length < output.shape[1]
            ):
                raise RuntimeError(
                    "video/register head intervention requires action/state registers"
                )
            register_start = output.shape[1] - action_register_length
            if intervention.query_scope == "video":
                query_start, query_stop = 0, register_start
            else:
                query_start, query_stop = register_start, output.shape[1]

        self.downstream_head_intervention_count += 1
        if intervention.scale == 1.0:
            return output
        head_indices = self._packed_head_indices(
            intervention.head_indices,
            output.device,
        )
        intervened = output.clone()
        selected = intervened[
            :,
            query_start:query_stop,
        ].index_select(2, head_indices)
        updated_region = intervened[:, query_start:query_stop].index_copy(
            2,
            head_indices,
            selected * intervention.scale,
        )
        intervened[:, query_start:query_stop] = updated_region
        return intervened

    def clear_anchor_sparse_history_cache(self) -> None:
        """Release gathered historical KV retained across action denoise steps."""

        self._anchor_sparse_history_cache_key = None
        self._anchor_sparse_history_k = None
        self._anchor_sparse_history_v = None
        self._anchor_sparse_history_cache.clear()

    def clear_packed_head_projection_weight_cache(self) -> None:
        """Keep dynamic Head membership from retaining old full-weight slices."""

        self._packed_head_projection_weight_cache.clear()

    def _get_sparse_history_kv(
        self,
        history_k: torch.Tensor,
        history_v: torch.Tensor,
        history_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather historical anchors once for an immutable rollout KV cache."""

        cache_key = (
            history_k.data_ptr(),
            history_k.storage_offset(),
            tuple(history_k.shape),
            tuple(history_k.stride()),
            history_v.data_ptr(),
            history_v.storage_offset(),
            tuple(history_v.shape),
            tuple(history_v.stride()),
            history_indices.data_ptr(),
            history_indices.storage_offset(),
            tuple(history_indices.shape),
            tuple(history_indices.stride()),
        )
        cached = self._anchor_sparse_history_cache.get(cache_key)
        if cached is not None:
            self._anchor_sparse_history_cache_key = cache_key
            self._anchor_sparse_history_k, self._anchor_sparse_history_v = cached
            return cached

        selected_k = gather_sequence_by_index(
            history_k,
            history_indices,
            validate_indices=False,
        )
        selected_v = gather_sequence_by_index(
            history_v,
            history_indices,
            validate_indices=False,
        )
        if not history_k.requires_grad and not history_v.requires_grad:
            self._anchor_sparse_history_cache_key = cache_key
            self._anchor_sparse_history_k = selected_k.detach()
            self._anchor_sparse_history_v = selected_v.detach()
            self._anchor_sparse_history_cache[cache_key] = (
                self._anchor_sparse_history_k,
                self._anchor_sparse_history_v,
            )
            return self._anchor_sparse_history_k, self._anchor_sparse_history_v
        return selected_k, selected_v

    def _packed_head_indices(
        self,
        heads: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (heads, device.type, device.index)
        cached = self._packed_head_index_cache.get(cache_key)
        if cached is None:
            cached = torch.tensor(heads, device=device, dtype=torch.long)
            self._packed_head_index_cache[cache_key] = cached
        return cached

    def _packed_head_channel_indices(
        self,
        heads: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Flatten head indices into their contiguous projection channels."""

        cache_key = (heads, device.type, device.index)
        cached = self._packed_head_channel_index_cache.get(cache_key)
        if cached is None:
            head_indices = self._packed_head_indices(heads, device)
            channel_offsets = torch.arange(
                self.head_dim,
                device=device,
                dtype=torch.long,
            )
            cached = (
                head_indices[:, None] * self.head_dim + channel_offsets[None, :]
            ).flatten()
            self._packed_head_channel_index_cache[cache_key] = cached
        return cached

    def _packed_head_projection_weights(
        self,
        heads: tuple[int, ...],
        channel_indices: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Prepack fused QKV rows and O columns for a stable M1 head class."""

        cache_key = (
            heads,
            channel_indices.device.type,
            channel_indices.device.index,
            dtype,
        )
        use_cache = not torch.is_grad_enabled()
        if use_cache:
            cached = self._packed_head_projection_weight_cache.get(cache_key)
            if cached is not None:
                return cached

        q_weight = self.q.weight.index_select(0, channel_indices)
        k_weight = self.k.weight.index_select(0, channel_indices)
        v_weight = self.v.weight.index_select(0, channel_indices)
        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()
        qkv_bias = None
        if self.q.bias is not None:
            assert self.k.bias is not None and self.v.bias is not None
            qkv_bias = torch.cat(
                (
                    self.q.bias.index_select(0, channel_indices),
                    self.k.bias.index_select(0, channel_indices),
                    self.v.bias.index_select(0, channel_indices),
                ),
                dim=0,
            ).contiguous()
        q_norm_weight = (
            self.norm_q.weight.index_select(0, channel_indices).contiguous()
            if isinstance(self.norm_q, WanRMSNorm)
            else None
        )
        k_norm_weight = (
            self.norm_k.weight.index_select(0, channel_indices).contiguous()
            if isinstance(self.norm_k, WanRMSNorm)
            else None
        )
        output_weight = self.o.weight.index_select(
            1,
            channel_indices,
        ).contiguous()
        packed = (
            qkv_weight,
            qkv_bias,
            q_norm_weight,
            k_norm_weight,
            output_weight,
        )
        if use_cache:
            packed = tuple(
                value.detach() if value is not None else None
                for value in packed
            )
            self._packed_head_projection_weight_cache[cache_key] = packed
        return packed

    def _project_packed_qkv_head_group(
        self,
        x: torch.Tensor,
        heads: tuple[int, ...],
        channel_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one fused QKV GEMM for the video tokens of a head group.

        Video Q/K normalization uses the group's channels as an unbiased
        estimate of the full-width RMS. Dense action/state registers take the
        exact full-width projection path separately.
        """

        (
            qkv_weight,
            qkv_bias,
            q_norm_weight,
            k_norm_weight,
            output_weight,
        ) = self._packed_head_projection_weights(
            heads,
            channel_indices,
            x.dtype,
        )
        qkv = F.linear(
            x,
            qkv_weight,
            qkv_bias,
        )
        group_dim = channel_indices.numel()
        query, key, value = qkv.split(group_dim, dim=-1)
        if q_norm_weight is not None:
            query = self.norm_q._norm(query.float()).type_as(query) * q_norm_weight
        if k_norm_weight is not None:
            key = self.norm_k._norm(key.float()).type_as(key) * k_norm_weight
        return query, key, value, output_weight

    def _forward_packed_current_head_groups(
        self,
        x: torch.Tensor,
        packed_freqs: torch.Tensor,
        *,
        action_register_length: int,
        kv_cache: torch.Tensor,
        history_token_count: int,
        head_groups: tuple[tuple[tuple[int, ...], float, float], ...],
        history_indices_by_ratio: dict[float, torch.Tensor],
        current_video_tokens_by_ratio: dict[float, int],
    ) -> torch.Tensor:
        """Execute grouped current Q/K/V/O without full-width video GEMMs."""

        batch, sequence, _ = x.shape
        register_x = x[:, :action_register_length]
        register_freqs = packed_freqs[:, :action_register_length]

        # Action/state registers are few and quality critical. Keep their Q/K/V
        # projections and full-width RMS normalization byte-identical to Dense.
        register_query = self.norm_q(self.q(register_x)).view(
            batch, action_register_length, self.num_heads, self.head_dim
        )
        register_key = self.norm_k(self.k(register_x)).view(
            batch, action_register_length, self.num_heads, self.head_dim
        )
        register_value = self.v(register_x).view(
            batch, action_register_length, self.num_heads, self.head_dim
        )
        register_query = apply_packed_rope(
            register_query,
            register_freqs,
        ).type_as(register_value)
        register_key = apply_packed_rope(
            register_key,
            register_freqs,
        ).type_as(register_value)

        history_key = (
            kv_cache[0, :, -history_token_count:]
            if history_token_count > 0
            else kv_cache[0, :, :0]
        )
        history_value = (
            kv_cache[1, :, -history_token_count:]
            if history_token_count > 0
            else kv_cache[1, :, :0]
        )

        group_attention_inputs: list[
            tuple[
                tuple[int, ...],
                int,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for group_heads, history_keep_ratio, current_keep_ratio in head_groups:
            head_indices = self._packed_head_indices(group_heads, x.device)
            channel_indices = self._packed_head_channel_indices(
                group_heads,
                x.device,
            )
            group_video_tokens = current_video_tokens_by_ratio.get(
                current_keep_ratio
            )
            if group_video_tokens is None:
                raise ValueError(
                    "Missing packed current-token count for head-group ratio "
                    f"{current_keep_ratio}"
                )
            group_length = action_register_length + group_video_tokens
            group_video_x = x[
                :,
                action_register_length:group_length,
            ]
            group_video_freqs = packed_freqs[
                :,
                action_register_length:group_length,
            ]
            group_head_count = len(group_heads)

            (
                video_query,
                video_key,
                video_value,
                group_output_weight,
            ) = self._project_packed_qkv_head_group(
                group_video_x,
                group_heads,
                channel_indices,
            )
            video_query = video_query.view(
                batch,
                group_video_tokens,
                group_head_count,
                self.head_dim,
            )
            video_key = video_key.view(
                batch,
                group_video_tokens,
                group_head_count,
                self.head_dim,
            )
            video_value = video_value.view(
                batch,
                group_video_tokens,
                group_head_count,
                self.head_dim,
            )
            video_query = apply_packed_rope(
                video_query,
                group_video_freqs,
            ).type_as(video_value)
            video_key = apply_packed_rope(
                video_key,
                group_video_freqs,
            ).type_as(video_value)

            group_query = torch.cat(
                (
                    register_query.index_select(2, head_indices),
                    video_query,
                ),
                dim=1,
            )
            group_current_key = torch.cat(
                (
                    register_key.index_select(2, head_indices),
                    video_key,
                ),
                dim=1,
            )
            group_current_value = torch.cat(
                (
                    register_value.index_select(2, head_indices),
                    video_value,
                ),
                dim=1,
            )

            if history_keep_ratio == 1.0:
                group_history_key = history_key
                group_history_value = history_value
            else:
                group_history_indices = history_indices_by_ratio.get(
                    history_keep_ratio
                )
                if group_history_indices is None:
                    raise ValueError(
                        "Missing packed history indices for head-group ratio "
                        f"{history_keep_ratio}"
                    )
                group_history_key, group_history_value = self._get_sparse_history_kv(
                    history_key,
                    history_value,
                    group_history_indices,
                )

            group_key = torch.cat(
                (
                    group_history_key.index_select(2, head_indices),
                    group_current_key,
                ),
                dim=1,
            )
            group_value = torch.cat(
                (
                    group_history_value.index_select(2, head_indices),
                    group_current_value,
                ),
                dim=1,
            )
            group_attention_inputs.append(
                (
                    group_heads,
                    group_length,
                    group_head_count,
                    group_query,
                    group_key,
                    group_value,
                    group_output_weight,
                )
            )

        # Treat each original attention head as an independent varlen batch
        # sequence. Heterogeneous M1 budgets then share one FA2 launch instead
        # of paying one launch plus gather/scatter chain per head group.
        packed_queries = []
        packed_keys = []
        packed_values = []
        query_lengths = []
        key_lengths = []
        max_query_length = 0
        max_key_length = 0
        for (
            _group_heads,
            group_length,
            group_head_count,
            group_query,
            group_key,
            group_value,
            _group_output_weight,
        ) in group_attention_inputs:
            group_key_length = group_key.shape[1]
            group_sequence_count = batch * group_head_count
            packed_queries.append(
                group_query.permute(0, 2, 1, 3).reshape(
                    group_sequence_count * group_length,
                    1,
                    self.head_dim,
                )
            )
            packed_keys.append(
                group_key.permute(0, 2, 1, 3).reshape(
                    group_sequence_count * group_key_length,
                    1,
                    self.head_dim,
                )
            )
            packed_values.append(
                group_value.permute(0, 2, 1, 3).reshape(
                    group_sequence_count * group_key_length,
                    1,
                    self.head_dim,
                )
            )
            query_lengths.append(
                torch.full(
                    (group_sequence_count,),
                    group_length,
                    device=x.device,
                    dtype=torch.int32,
                )
            )
            key_lengths.append(
                torch.full(
                    (group_sequence_count,),
                    group_key_length,
                    device=x.device,
                    dtype=torch.int32,
                )
            )
            max_query_length = max(max_query_length, group_length)
            max_key_length = max(max_key_length, group_key_length)

        packed_output = self.attn.forward_varlen_head_sequences(
            torch.cat(packed_queries, dim=0),
            torch.cat(packed_keys, dim=0),
            torch.cat(packed_values, dim=0),
            q_lens=torch.cat(query_lengths),
            k_lens=torch.cat(key_lengths),
            max_seqlen_q=max_query_length,
            max_seqlen_k=max_key_length,
        )

        projected_output = x.new_zeros(x.shape)
        observed_register_output = (
            x.new_empty(
                batch,
                action_register_length,
                self.num_heads,
                self.head_dim,
            )
            if self.dynamic_m1_packed_observer is not None
            and self.dynamic_m1_cfg_branch is not None
            else None
        )
        output_offset = 0
        for (
            group_heads,
            group_length,
            group_head_count,
            _group_query,
            _group_key,
            _group_value,
            group_output_weight,
        ) in group_attention_inputs:
            group_output_tokens = batch * group_head_count * group_length
            group_output = packed_output[
                output_offset:output_offset + group_output_tokens
            ].reshape(
                batch,
                group_head_count,
                group_length,
                self.head_dim,
            ).permute(0, 2, 1, 3)
            group_output_projection = F.linear(
                group_output.flatten(2),
                group_output_weight,
                None,
            )
            projected_output[:, :group_length].add_(group_output_projection)
            if observed_register_output is not None:
                head_indices = self._packed_head_indices(group_heads, x.device)
                observed_register_output[:, :action_register_length].index_copy_(
                    2,
                    head_indices,
                    group_output[:, :action_register_length],
                )
            output_offset += group_output_tokens

        if observed_register_output is not None:
            self._observe_dynamic_m1_action_output(
                observed_register_output,
                action_register_length=action_register_length,
                registers_first=True,
            )
        if self.o.bias is not None:
            projected_output.add_(self.o.bias)
        return projected_output

    def _packed_query_positions(
        self,
        *,
        video_tokens: int,
        sequence: int,
        action_register_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        maximum_video_tokens = sequence - action_register_length
        if not 0 <= video_tokens <= maximum_video_tokens:
            raise ValueError(
                "Grouped current-query tokens exceed the active packed video prefix"
            )
        cache_key = (
            video_tokens,
            sequence,
            action_register_length,
            device.type,
            device.index,
        )
        cached = self._packed_query_position_cache.get(cache_key)
        if cached is None:
            # PackedMiddleState places the Dense action/state registers first,
            # followed by the nested current-video anchor prefix.
            cached = torch.arange(
                action_register_length + video_tokens,
                device=device,
                dtype=torch.long,
            )
            self._packed_query_position_cache[cache_key] = cached
        return cached

    def forward_packed(
        self,
        x: torch.Tensor,
        packed_freqs: torch.Tensor,
        *,
        action_register_length: int,
        kv_cache: torch.Tensor,
        history_indices: torch.Tensor,
        history_token_count: int,
        head_groups: tuple[tuple[Any, ...], ...] | None = None,
        history_indices_by_ratio: dict[float, torch.Tensor] | None = None,
        current_video_tokens_by_ratio: dict[float, int] | None = None,
        maximum_current_x: torch.Tensor | None = None,
        maximum_current_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Q/K/V/O only on packed current tokens plus Dense registers.

        ``x`` may use any token order because causal action denoising uses
        unmasked attention for the current block.  ``packed_freqs`` preserves
        every token's original frame/row/column or action/state position.
        Historical K/V are already RoPE-applied in the immutable causal cache.
        This path never mutates that cache.
        """

        if x.ndim != 3:
            raise ValueError("packed x must have shape [B, L, C]")
        if not 0 < action_register_length < x.shape[1]:
            raise ValueError("packed attention requires Dense action/state registers")
        if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
            raise ValueError("kv_cache must have shape [2, B, L, H, D]")
        if history_indices.ndim != 2 or history_indices.shape[0] != x.shape[0]:
            raise ValueError("history_indices must have shape [B, K]")
        if not 0 <= history_token_count <= kv_cache.shape[2]:
            raise ValueError("history_token_count exceeds the immutable KV cache")
        batch, sequence, _ = x.shape
        if self.packed_max_action_current and head_groups is not None:
            raise ValueError(
                "Maximum action-current K/V currently requires one shared head group"
            )
        if (
            head_groups is not None
            and history_indices_by_ratio is not None
            and current_video_tokens_by_ratio is not None
            and all(len(group) == 3 and group[2] is not None for group in head_groups)
        ):
            normalized_current_head_groups = tuple(
                (group[0], float(group[1]), float(group[2]))
                for group in head_groups
            )
            flattened_heads = sorted(
                head
                for group_heads, _, _ in normalized_current_head_groups
                for head in group_heads
            )
            if flattened_heads != list(range(self.num_heads)):
                raise ValueError(
                    "Packed head groups must partition every attention head"
                )
            return self._forward_packed_current_head_groups(
                x,
                packed_freqs,
                action_register_length=action_register_length,
                kv_cache=kv_cache,
                history_token_count=history_token_count,
                head_groups=normalized_current_head_groups,
                history_indices_by_ratio=history_indices_by_ratio,
                current_video_tokens_by_ratio=current_video_tokens_by_ratio,
            )
        query = self.norm_q(self.q(x)).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        key = self.norm_k(self.k(x)).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        value = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)
        query = apply_packed_rope(query, packed_freqs).type_as(value)
        key = apply_packed_rope(key, packed_freqs).type_as(value)

        maximum_video_key: torch.Tensor | None = None
        maximum_video_value: torch.Tensor | None = None
        if self.packed_max_action_current and maximum_current_x is not None:
            if maximum_current_freqs is None:
                raise ValueError(
                    "Maximum action-current K/V requires maximum-prefix RoPE"
                )
            if maximum_current_x.ndim != 3:
                raise ValueError("maximum_current_x must have shape [B, L, C]")
            if (
                maximum_current_x.shape[0] != batch
                or maximum_current_x.shape[2] != self.dim
                or maximum_current_x.shape[1] < sequence
            ):
                raise ValueError(
                    "maximum_current_x must contain the active packed sequence"
                )
            if maximum_current_freqs.shape[:2] != maximum_current_x.shape[:2]:
                raise ValueError(
                    "maximum_current_freqs must align with maximum_current_x"
                )
            if maximum_current_x.shape[1] > sequence:
                maximum_sequence = maximum_current_x.shape[1]
                maximum_key = self.norm_k(self.k(maximum_current_x)).view(
                    batch,
                    maximum_sequence,
                    self.num_heads,
                    self.head_dim,
                )
                maximum_value = self.v(maximum_current_x).view(
                    batch,
                    maximum_sequence,
                    self.num_heads,
                    self.head_dim,
                )
                maximum_key = apply_packed_rope(
                    maximum_key,
                    maximum_current_freqs,
                ).type_as(maximum_value)
                maximum_video_key = maximum_key[:, action_register_length:]
                maximum_video_value = maximum_value[:, action_register_length:]

        history_key = (
            kv_cache[0, :, -history_token_count:]
            if history_token_count > 0
            else kv_cache[0, :, :0]
        )
        history_value = (
            kv_cache[1, :, -history_token_count:]
            if history_token_count > 0
            else kv_cache[1, :, :0]
        )
        if head_groups is None:
            if history_indices.shape[1] == history_key.shape[1]:
                # A 100% nested profile is only a permutation of the immutable
                # Dense history.  Attention is invariant to a joint K/V
                # permutation, while gathering it would retain another full
                # history copy per layer and per denoise route.
                sparse_history_key = history_key
                sparse_history_value = history_value
            elif history_indices.numel():
                sparse_history_key, sparse_history_value = self._get_sparse_history_kv(
                    history_key,
                    history_value,
                    history_indices,
                )
            else:
                sparse_history_key = history_key[:, :0]
                sparse_history_value = history_value[:, :0]
            dense_action_history_active = (
                self.packed_dense_action_history
                and sparse_history_key.shape[1] < history_key.shape[1]
            )
            max_action_current_active = maximum_video_key is not None
            if dense_action_history_active or max_action_current_active:
                action_query = query[:, :action_register_length]
                video_query = query[:, action_register_length:]
                action_history_key = (
                    history_key if dense_action_history_active else sparse_history_key
                )
                action_history_value = (
                    history_value if dense_action_history_active else sparse_history_value
                )
                if max_action_current_active:
                    assert maximum_video_key is not None
                    assert maximum_video_value is not None
                    action_current_key = torch.cat(
                        (maximum_video_key, key[:, :action_register_length]),
                        dim=1,
                    )
                    action_current_value = torch.cat(
                        (maximum_video_value, value[:, :action_register_length]),
                        dim=1,
                    )
                else:
                    action_current_key = key
                    action_current_value = value
                action_output = self.attn(
                    action_query,
                    torch.cat((action_history_key, action_current_key), dim=1),
                    torch.cat((action_history_value, action_current_value), dim=1),
                )
                video_output = self.attn(
                    video_query,
                    torch.cat((sparse_history_key, key), dim=1),
                    torch.cat((sparse_history_value, value), dim=1),
                )
                output = torch.cat((action_output, video_output), dim=1)
            else:
                output = self.attn(
                    query,
                    torch.cat((sparse_history_key, key), dim=1),
                    torch.cat((sparse_history_value, value), dim=1),
                )
        else:
            if history_indices_by_ratio is None:
                raise ValueError(
                    "Grouped packed attention requires history indices by ratio"
                )
            if any(len(group) not in (2, 3) for group in head_groups):
                raise ValueError(
                    "Packed head groups require heads, history ratio, and optional current ratio"
                )
            normalized_head_groups = tuple(
                (
                    group[0],
                    group[1],
                    group[2] if len(group) == 3 else None,
                )
                for group in head_groups
            )
            flattened_heads = sorted(
                head
                for group_heads, _, _ in normalized_head_groups
                for head in group_heads
            )
            if flattened_heads != list(range(self.num_heads)):
                raise ValueError("Packed head groups must partition every attention head")
            output = value.new_zeros(value.shape)
            for (
                group_heads,
                history_keep_ratio,
                current_keep_ratio,
            ) in normalized_head_groups:
                head_indices = self._packed_head_indices(group_heads, query.device)
                if history_keep_ratio == 1.0:
                    group_history_key = history_key
                    group_history_value = history_value
                else:
                    group_history_indices = history_indices_by_ratio.get(
                        history_keep_ratio
                    )
                    if group_history_indices is None:
                        raise ValueError(
                            "Missing packed history indices for head-group ratio "
                            f"{history_keep_ratio}"
                        )
                    group_history_key, group_history_value = (
                        self._get_sparse_history_kv(
                            history_key,
                            history_value,
                            group_history_indices,
                        )
                    )
                if current_keep_ratio is None:
                    query_positions = None
                    group_query = query.index_select(2, head_indices)
                    group_current_key = key.index_select(2, head_indices)
                    group_current_value = value.index_select(2, head_indices)
                else:
                    if current_video_tokens_by_ratio is None:
                        raise ValueError(
                            "Grouped current Q/K/V requires video-token counts by ratio"
                        )
                    group_video_tokens = current_video_tokens_by_ratio.get(
                        current_keep_ratio
                    )
                    if group_video_tokens is None:
                        raise ValueError(
                            "Missing packed current-token count for head-group ratio "
                            f"{current_keep_ratio}"
                        )
                    query_positions = self._packed_query_positions(
                        video_tokens=group_video_tokens,
                        sequence=sequence,
                        action_register_length=action_register_length,
                        device=query.device,
                    )
                    group_query = query.index_select(1, query_positions).index_select(
                        2, head_indices
                    )
                    group_current_key = key.index_select(
                        1, query_positions
                    ).index_select(2, head_indices)
                    group_current_value = value.index_select(
                        1, query_positions
                    ).index_select(2, head_indices)
                group_output = self.attn(
                    group_query,
                    torch.cat(
                        (
                            group_history_key.index_select(2, head_indices),
                            group_current_key,
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            group_history_value.index_select(2, head_indices),
                            group_current_value,
                        ),
                        dim=1,
                    ),
                )
                if query_positions is None:
                    output = output.index_copy(2, head_indices, group_output)
                else:
                    selected_output = output.index_select(1, query_positions)
                    selected_output = selected_output.index_copy(
                        2, head_indices, group_output
                    )
                    output = output.index_copy(
                        1, query_positions, selected_output
                    )
        self._observe_dynamic_m1_action_output(
            output,
            action_register_length=action_register_length,
            registers_first=True,
        )
        return self.o(output.flatten(2))

    def _visualize_attention_mask(self, total_len, first_image_len, image_blocks_len, 
                                   action_len, state_len, num_image_blocks, 
                                   num_action_blocks, num_state_blocks,
                                   num_frame_per_block, frame_seqlen,
                                   num_action_per_block, num_state_per_block):
        """
        Create and print a visualization of the attention mask pattern.
        Returns a binary mask [total_len, total_len] where 1 = can attend, 0 = cannot attend.
        """
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Create mask tensor
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # First image: self-attention only
        mask[first_image_start:first_image_end, first_image_start:first_image_end] = True
        
        # Image blocks
        for block_idx in range(num_image_blocks):
            block_start = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            
            # Attend to first image
            mask[block_start:block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[block_start:block_end, image_kv_start:block_end] = True
            
            # Attend to current action block
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            mask[block_start:block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[block_start:block_end, state_block_start:state_block_end] = True
        
        # Action blocks
        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            
            # Attend to first image
            mask[action_block_start:action_block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            image_block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[action_block_start:action_block_end, image_kv_start:image_block_end] = True
            
            # Self-attention
            mask[action_block_start:action_block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[action_block_start:action_block_end, state_block_start:state_block_end] = True
        
        # State blocks: self-attention only
        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[state_block_start:state_block_end, state_block_start:state_block_end] = True
        
        return mask

    def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, num_frame_per_block=1, 
                                       action_horizon=None, state_horizon=None, 
                                       num_action_per_block=None, num_state_per_block=None,
                                       visualize_mask=False):
        """
        Implement blockwise causal attention using flash_attention.
        Matches the pattern from _prepare_blockwise_causal_attn_mask:
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image blocks + current action block + current state block
        - Action blocks: can attend to previous image blocks + current image block + current state block + first image
        - State blocks: conditioning only, cannot attend to anything
        
        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            frame_seqlen: Number of tokens per frame
            num_frame_per_block: Number of frames per attention block
            action_horizon: Total number of action tokens (if None, no action/state tokens)
            state_horizon: Total number of state tokens (if None, no action/state tokens)
            num_action_per_block: Number of action tokens per block
            num_state_per_block: Number of state tokens per block
            visualize_mask: If True, print the attention mask pattern
        
        Returns:
            Attention output [B, L, num_heads, head_dim]
        """
        b, total_len, n, d = q.shape
        
        # Check if we have action/state tokens
        has_action_state = (action_horizon is not None and state_horizon is not None)
        
        if not has_action_state:
            # OPTIMIZED: Simple blockwise causal attention (without action/state tokens)
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            num_blocks = (num_frames - 1) // num_frame_per_block
            
            # Handle edge case when sequence is too short (no blocks to process)
            if num_blocks <= 0:
                # Process entire sequence as a single block
                return self.attn(q, k, v)
            
            # OPTIMIZATION: For global attention, process all blocks in one call with causal masking
            if self.local_attn_size == -1:
                # Single flash_attention call with causal=True for all blocks at once
                # This is much faster than looping!
                return self.causal_attn(q, k, v)
            
            # With local attention, still need loop but optimize it
            # Pre-allocate output tensor
            output = torch.empty_like(q)
            
            # Pre-compute block boundaries
            block_starts = [frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            kv_starts = [max(0, end - self.local_attn_size * frame_seqlen) for end in block_ends]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]
                
                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end]
                )
            
            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        # Multi-modal structure: [first image] [image blocks] [action blocks] [state blocks]
        # Calculate block structure
        first_image_len = frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        image_blocks_len = total_len - first_image_len - action_len - state_len
        
        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block

        assert num_image_blocks == num_action_blocks == num_state_blocks
        
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Visualize attention mask if requested
        if visualize_mask:
            mask = self._visualize_attention_mask(
                total_len, first_image_len, image_blocks_len, 
                action_len, state_len, num_image_blocks,
                num_action_blocks, num_state_blocks,
                num_frame_per_block, frame_seqlen,
                num_action_per_block, num_state_per_block
            )
            
            print("\n" + "="*80)
            print("ATTENTION MASK VISUALIZATION")
            print("="*80)
            print(f"Total length: {total_len}")
            print(f"First image: [{first_image_start}:{first_image_end}] (len={first_image_len})")
            print(f"Image blocks: [{image_blocks_start}:{image_blocks_end}] (len={image_blocks_len}, num_blocks={num_image_blocks})")
            print(f"Action tokens: [{action_start}:{action_end}] (len={action_len}, num_blocks={num_action_blocks})")
            print(f"State tokens: [{state_start}:{state_end}] (len={state_len}, num_blocks={num_state_blocks})")
            print(f"Local attention size: {self.local_attn_size}")
            print("-"*80)
            
            # Print a downsampled version of the mask if it's too large
            if total_len <= 100:
                # Print full mask for small sequences
                print("Attention mask (1=can attend, 0=cannot attend):")
                print("Rows=Query tokens, Cols=Key tokens")
                for i in range(total_len):
                    row = "".join(["1" if mask[i, j] else "." for j in range(total_len)])
                    print(f"{i:4d}: {row}")
            else:
                # Print downsampled version for large sequences
                downsample = max(1, total_len // 100)
                print(f"Attention mask (downsampled by {downsample}x):")
                print("Rows=Query tokens, Cols=Key tokens (1=can attend, .=cannot attend)")
                for i in range(0, total_len, downsample):
                    row = "".join(["1" if mask[i, j] else "." for j in range(0, total_len, downsample)])
                    print(f"{i:4d}: {row}")
            
            # Save mask as image
            try:
                import cv2
                import numpy as np
                mask_np = mask.cpu().float().numpy()
                # Resize for visualization if needed
                if total_len > 1000:
                    mask_np = cv2.resize(mask_np, (1000, 1000), interpolation=cv2.INTER_NEAREST)
                mask_img = (mask_np * 255).astype(np.uint8)
                cv2.imwrite("attention_mask_blockwise_flash.png", mask_img)
                print(f"\nMask saved to: attention_mask_blockwise_flash.png")
            except Exception as e:
                print(f"Could not save mask image: {e}")
            
            print("="*80 + "\n")
        
        # OPTIMIZED: Pre-allocate output tensor and pre-compute all indices
        output = torch.empty_like(q)
        
        # Process first image (conditioning, can only self-attend)
        output[:, first_image_start:first_image_end] = self.attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end]
        )
        
        # Pre-compute all block indices for image blocks
        image_block_starts = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        image_block_ends = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        if self.local_attn_size != -1:
            image_kv_starts = [max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks
        
        # Pre-compute action and state block indices
        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]
        
        # Process each image block
        for block_idx in range(num_image_blocks):
            block_start = image_block_starts[block_idx]
            block_end = image_block_ends[block_idx]
            image_kv_start = image_kv_starts[block_idx]
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Build context: first image + relevant image blocks + current action + current state
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )
        
        # Process each action block
        for block_idx in range(num_action_blocks):
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            image_block_end = image_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Determine image context range
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            
            # Build context
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:image_block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:image_block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )
        
        # Process state blocks (conditioning, can only self-attend)
        for block_idx in range(num_state_blocks):
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end]
            )
        
        return output

    def _process_clean_image_only(self, clean_image_q, clean_image_k, clean_image_v, clean_frames):
        """Process clean image blocks with causal attention pattern - OPTIMIZED
        
        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to first frame + previous blocks (0 to i-1) + current block
        
        OPTIMIZATION: Instead of looping through blocks, we batch process them together
        by using a single flash_attention call with properly structured KV cache.
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            # Only first frame - single attention call
            return self.attn(
                clean_image_q[:, :self.frame_seqlen],
                clean_image_k[:, :self.frame_seqlen],
                clean_image_v[:, :self.frame_seqlen]
            )
        
        # Pre-allocate output tensor (avoids list append + cat overhead)
        b, total_len, n, d = clean_image_q.shape
        output = torch.empty_like(clean_image_q)
        
        # First frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            clean_image_q[:, :self.frame_seqlen],
            clean_image_k[:, :self.frame_seqlen],
            clean_image_v[:, :self.frame_seqlen]
        )
        
        # OPTIMIZATION: Process all blocks together with causal masking
        # For global attention (no local_attn_size), we can process all blocks in one call
        if self.local_attn_size == -1:
            # Single attention call for all blocks!
            # Each position can attend to first_frame + everything up to itself
            blocks_q = clean_image_q[:, self.frame_seqlen:]
            blocks_k = clean_image_k  # Can attend to everything including first frame
            blocks_v = clean_image_v
            
            # Use causal masking: each block token can see first frame + all previous tokens
            output[:, self.frame_seqlen:] = self.causal_attn(
                blocks_q, blocks_k, blocks_v
            )
        else:
            # With local attention, we still need to loop but with optimizations
            # Pre-compute all block boundaries to reduce overhead
            block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                
                q_block = clean_image_q[:, block_start:block_end]
                
                # Context: first frame + recent blocks within local_attn_size
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                k_context = torch.cat([
                    clean_image_k[:, :self.frame_seqlen],  # First frame
                    clean_image_k[:, image_kv_start:block_end]  # Recent blocks + current
                ], dim=1)
                v_context = torch.cat([
                    clean_image_v[:, :self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end]
                ], dim=1)
                
                output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
        """Process state blocks: self-attention only - OPTIMIZED
        
        OPTIMIZATION: State blocks only do self-attention within each block.
        Instead of looping, we can process all blocks in a single call with block-diagonal masking,
        or even simpler: just one attention call since they're independent.
        """
        num_blocks = state_horizon // self.num_state_per_block
        
        if num_blocks == 1:
            # Single block - one attention call
            return self.attn(state_q, state_k, state_v)
        
        # OPTIMIZATION: Since each state block only attends to itself (no cross-block attention),
        # we can process all blocks in a single batched call. Flash attention will handle this
        # efficiently. The blocks are independent, so this is safe.
        # Alternative: reshape and process as separate batch items
        
        # Pre-allocate output
        output = torch.empty_like(state_q)
        
        # Process all blocks (keeping loop for now due to block-diagonal pattern)
        # This could be further optimized with custom masking
        for block_idx in range(num_blocks):
            state_block_start = block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            
            output[:, state_block_start:state_block_end] = self.attn(
                state_q[:, state_block_start:state_block_end],
                state_k[:, state_block_start:state_block_end],
                state_v[:, state_block_start:state_block_end]
            )
        
        return output
    
    def _process_noisy_image_blocks(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                     clean_image_k, clean_image_v,
                                     noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                                     half_frames, action_horizon, state_horizon):
        """Process noisy image blocks with teacher forcing pattern - OPTIMIZED
        
        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to action[i] + state[i] + first_clean_frame + clean_blocks[0:i] + current_noisy_block
        
        OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        # Pre-allocate output tensor
        output = torch.empty_like(noisy_image_q)
        
        # First noisy frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            noisy_image_k[:, :self.frame_seqlen],
            noisy_image_v[:, :self.frame_seqlen]
        )
        
        if num_blocks == 0:
            return output
        
        # Pre-compute all block indices to reduce loop overhead
        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        # Process noisy image blocks
        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            q_block = noisy_image_q[:, noisy_start:noisy_end]
            
            # Build context: first_clean_frame + clean_blocks[0:i] + current_noisy_block + action[i] + state[i]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_noisy_action_blocks(self, noisy_action_q, noisy_action_k, noisy_action_v,
                                      clean_image_k, clean_image_v,
                                      noisy_image_k, noisy_image_v,
                                      noisy_state_k, noisy_state_v,
                                      half_frames, action_horizon, state_horizon):
        """Process noisy action blocks with teacher forcing pattern - OPTIMIZED
        
        First action (for first frame): cannot attend to anything (self-attention only)
        Action block i: attends to first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
        
        OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
        """
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            return torch.empty_like(noisy_action_q)
        
        # Pre-allocate output tensor
        output = torch.empty_like(noisy_action_q)
        
        # Pre-compute all block indices
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        clean_context_ends = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_starts = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_ends = [start + self.frame_seqlen * self.num_frame_per_block for start in noisy_image_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        # Process noisy action blocks
        for block_idx in range(num_blocks):
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            noisy_img_start = noisy_image_block_starts[block_idx]
            noisy_img_end = noisy_image_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            q_block = noisy_action_q[:, action_start:action_end]
            
            # Build context: first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_img_start:noisy_img_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_img_start:noisy_img_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)
        
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        anchor_route_indices: torch.Tensor | None = None,
        current_video_indices: torch.Tensor | None = None,
        update_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        updated_kv_cache: torch.Tensor | None = None
        updated_anchor_route_indices = anchor_route_indices
        sparse_query_output_indices: torch.Tensor | None = None

        if kv_cache is None:
            if is_tf:
                # Teacher forcing training.
                if action_register_length is not None:
                    q_context = q[:, :(s-action_register_length)//2]
                    k_context = k[:, :(s-action_register_length)//2]
                    q_noisy = q[:, (s-action_register_length)//2:]  
                    k_noisy = k[:, (s-action_register_length)//2:]
                else:
                    q_context = q[:, :s//2]
                    k_context = k[:, :s//2]
                    q_noisy = q[:, s//2:]
                    k_noisy = k[:, s//2:]
                roped_query = []
                roped_key = []

                # rope should be same for clean and noisy parts
                rq_context = rope_action_apply(
                    x=q_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)
                rk_context = rope_action_apply(
                    x=k_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)

                rq_noisy = rope_action_apply(
                    x=q_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                rk_noisy = rope_action_apply(
                    x=k_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                roped_query.append(rq_context)
                roped_key.append(rk_context)
                roped_query.append(rq_noisy)
                roped_key.append(rk_noisy)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
                # Calculate sequence dimensions
                half_seq_len = (s - (action_register_length if action_register_length is not None else 0)) // 2
                
                if action_register_length is not None:
                    # Teacher forcing structure:
                    # Clean half: [image tokens only]
                    # Noisy half: [image tokens][action tokens][state tokens]
                    # Causality only applies to image blocks!
                    
                    # Clean half contains ONLY image tokens
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen
                    
                    # Noisy half contains image + action + state tokens
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block
                    
                    # Block layout must match actual register length. For 5B use 320x176 so latent frame_seqlen=55.
                    if roped_query.shape[1] != half_seq_len + noisy_image_seq_len + action_horizon + state_horizon:
                        raise ValueError(
                            "Sequence length does not match block layout. "
                            "For 5B use 320x176 (e.g. data=dreamzero/droid_relative_wan22 or image_resolution_width=320, image_resolution_height=176). "
                            f"Got noisy_frames={noisy_frames}, num_image_blocks={num_image_blocks}, "
                            f"action_register_length={action_register_length}. "
                            "Ensure (noisy_frames - 1) // num_frame_per_block >= 1 and register length equals "
                            "num_blocks * (num_action_per_block + num_state_per_block)."
                        )
                    
                    # Split clean and noisy parts
                    # Clean: [image tokens only]
                    clean_image_q = roped_query[:, :clean_image_seq_len]
                    clean_image_k = roped_key[:, :clean_image_seq_len]
                    clean_image_v = v[:, :clean_image_seq_len]

                    # Noisy: [image tokens][action tokens][state tokens]
                    noisy_image_q = roped_query[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_q = roped_query[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_q = roped_query[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    noisy_image_k = roped_key[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_k = roped_key[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_k = roped_key[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    noisy_image_v = v[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_v = v[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_v = v[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    # ========== Process CLEAN (context) image tokens ==========
                    # Clean images: simple blockwise causal attention (no action/state)
                    clean_image_outputs = self._process_clean_image_only(
                        clean_image_q, clean_image_k, clean_image_v, clean_frames)
                    
                    # ========== Process NOISY tokens ==========
                    # Noisy image blocks: attend to previous clean image blocks + current noisy image + current noisy action + current noisy state
                    noisy_image_outputs = self._process_noisy_image_blocks(
                        noisy_image_q, noisy_image_k, noisy_image_v,
                        clean_image_k, clean_image_v,
                        noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # Noisy action blocks: attend to previous clean image blocks (including first) + current noisy image + current noisy action + same state
                    noisy_action_outputs = self._process_noisy_action_blocks(
                        noisy_action_q, noisy_action_k, noisy_action_v,
                        clean_image_k, clean_image_v, 
                        noisy_image_k, noisy_image_v,
                        noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # Noisy state blocks: self-attention only
                    noisy_state_outputs = self._process_state_blocks(
                        noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)
                    
                    # Concatenate all outputs in order: clean_img, noisy_img, noisy_act, noisy_state
                    x = torch.cat([
                        clean_image_outputs,
                        noisy_image_outputs, noisy_action_outputs, noisy_state_outputs
                    ], dim=1)
                else:
                    # No action/state tokens, fall back to simple image-only teacher forcing
                    half_frames = half_seq_len // self.frame_seqlen
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]
                    
                    # Process clean frames with blockwise causal attention
                    x_clean = self._blockwise_causal_flash_attn(
                        clean_q, clean_k, clean_v, self.frame_seqlen, self.num_frame_per_block,
                        action_horizon=None, state_horizon=None,
                        num_action_per_block=None, num_state_per_block=None,
                        visualize_mask=False)
                    
                    # Process noisy frames: attend to all clean frames + themselves
                    full_k = torch.cat([clean_k, noisy_k], dim=1)
                    full_v = torch.cat([clean_v, noisy_v], dim=1)
                    x_noisy = self.attn(noisy_q, full_k, full_v)
                    
                    x = torch.cat([x_clean, x_noisy], dim=1)

            else:
                roped_query = rope_action_apply(
                    x=q,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    x=k,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                # Calculate dynamic action and state horizons
                if action_register_length is not None:
                    chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None

                # Use blockwise causal flash attention without massive padding
                visualize = False
                x = self._blockwise_causal_flash_attn(
                    roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=self.num_action_per_block if action_register_length else None,
                    num_state_per_block=self.num_state_per_block if action_register_length else None,
                    visualize_mask=visualize)

        else:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block

            roped_query = causal_rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            # split roped_query and roped_action_query (the last action_register_length tokens)
            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None

            num_new_tokens = roped_query.shape[1]
            assert roped_key.shape[1] == num_new_tokens
            assert v.shape[1] == num_new_tokens

            # If we are using local attention and the current KV cache size is larger
            # than the local attention size, we need to truncate the KV cache

            updated_k = kv_cache[0]
            updated_v = kv_cache[1]
            new_k: torch.Tensor | None = None
            new_v: torch.Tensor | None = None
            if update_kv_cache:
                # Preserve the original cache-producing path byte-for-byte when
                # the caller will consume the returned cache.
                new_k = torch.cat([updated_k, roped_key], dim=1)
                new_v = torch.cat([updated_v, v], dim=1)
                new_k = new_k[:, -self.max_attention_size:]
                new_v = new_v[:, -self.max_attention_size:]
                history_k = new_k[:, :-num_new_tokens]
                history_v = new_v[:, :-num_new_tokens]
            else:
                # Action denoising does not mutate the causal history.  Trim the
                # history before concatenation so sparse attention can avoid
                # materialising a full history+current KV tensor.
                history_capacity = max(0, self.max_attention_size - num_new_tokens)
                history_k = (
                    updated_k[:, -history_capacity:]
                    if history_capacity > 0
                    else updated_k[:, :0]
                )
                history_v = (
                    updated_v[:, -history_capacity:]
                    if history_capacity > 0
                    else updated_v[:, :0]
                )

            total_video_tokens = history_k.shape[1] + num_new_tokens

            if action_register_length is not None:
                if self.dynamic_oracle_collector is not None:
                    oracle_video_key = (
                        new_k
                        if new_k is not None
                        else torch.cat([history_k, roped_key], dim=1)
                    )
                    oracle_video_value = (
                        new_v
                        if new_v is not None
                        else torch.cat([history_v, v], dim=1)
                    )
                    tokens_per_block = (
                        self.num_action_per_block + self.num_state_per_block
                    )
                    num_register_blocks = (
                        roped_action_query.shape[1] // tokens_per_block
                    )
                    oracle_action_query = roped_action_query[
                        :, :num_register_blocks * self.num_action_per_block
                    ]
                    self.dynamic_oracle_collector.observe(
                        layer_index=self.layer_index,
                        video_query=roped_query,
                        action_query=oracle_action_query,
                        video_key=oracle_video_key,
                        video_value=oracle_video_value,
                    )
                attention_k: torch.Tensor | None = new_k
                attention_v: torch.Tensor | None = new_v
                sparse_config = self.anchor_sparse_config
                routing_active = (
                    sparse_config is not None
                    and (
                        sparse_config.keep_ratio < 1.0
                        or self.record_anchor_diagnostics
                    )
                )
                if routing_active:
                    assert sparse_config is not None
                    if total_video_tokens % sparse_config.frame_seqlen != 0:
                        raise ValueError(
                            "Embodied anchor routing requires complete video frames, got "
                            f"{total_video_tokens} keys for frame_seqlen={sparse_config.frame_seqlen}"
                        )
                    num_video_frames = total_video_tokens // sparse_config.frame_seqlen
                    expected_route_length = sparse_config.selected_video_tokens(num_video_frames)
                    route_is_valid = (
                        updated_anchor_route_indices is not None
                        and updated_anchor_route_indices.shape
                        == (history_k.shape[0], expected_route_length)
                        and updated_anchor_route_indices.device == history_k.device
                    )
                    if not route_is_valid:
                        tokens_per_block = self.num_action_per_block + self.num_state_per_block
                        if roped_action_query.shape[1] % tokens_per_block != 0:
                            raise ValueError(
                                "Action/state register length is incompatible with the anchor router: "
                                f"{roped_action_query.shape[1]} vs block size {tokens_per_block}"
                            )
                        num_register_blocks = roped_action_query.shape[1] // tokens_per_block
                        action_query = roped_action_query[
                            :, :num_register_blocks * self.num_action_per_block
                        ]
                        if new_k is None:
                            new_k = torch.cat([history_k, roped_key], dim=1)
                        route = route_action_conditioned_video_keys(
                            action_query=action_query,
                            video_key=new_k,
                            config=sparse_config,
                        )
                        updated_anchor_route_indices = route.video_indices.detach()
                        if self.record_anchor_diagnostics:
                            self.last_anchor_route = route.detached()
                        if (
                            self.dynamic_m1_packed_observer is not None
                            and self.dynamic_m1_cfg_branch is not None
                        ):
                            self.dynamic_m1_packed_observer.observe_route_scores(
                                route.scores,
                                cfg_branch=self.dynamic_m1_cfg_branch,
                            )
                    if sparse_config.keep_ratio < 1.0:
                        num_new_frames = num_new_tokens // sparse_config.frame_seqlen
                        direct_sparse_history = (
                            not update_kv_cache
                            and num_new_tokens % sparse_config.frame_seqlen == 0
                            and num_new_frames <= sparse_config.recent_dense_frames
                            and expected_route_length >= num_new_tokens
                        )
                        if direct_sparse_history:
                            historical_route_length = expected_route_length - num_new_tokens
                            history_indices = updated_anchor_route_indices[
                                :, :historical_route_length
                            ]
                            selected_history_k, selected_history_v = self._get_sparse_history_kv(
                                history_k,
                                history_v,
                                history_indices,
                            )
                            attention_k = torch.cat([selected_history_k, roped_key], dim=1)
                            attention_v = torch.cat([selected_history_v, v], dim=1)
                        else:
                            if new_k is None:
                                new_k = torch.cat([history_k, roped_key], dim=1)
                            if new_v is None:
                                new_v = torch.cat([history_v, v], dim=1)
                            attention_k = gather_sequence_by_index(
                                new_k,
                                updated_anchor_route_indices,
                                validate_indices=False,
                            )
                            attention_v = gather_sequence_by_index(
                                new_v,
                                updated_anchor_route_indices,
                                validate_indices=False,
                            )
                if attention_k is None:
                    attention_k = torch.cat([history_k, roped_key], dim=1)
                if attention_v is None:
                    attention_v = torch.cat([history_v, v], dim=1)
                if current_video_indices is not None:
                    selected_video_query = gather_sequence_by_index(
                        roped_query,
                        current_video_indices,
                        validate_indices=False,
                    )
                    attention_q = torch.cat(
                        [selected_video_query, roped_action_query],
                        dim=1,
                    )
                    sparse_query_output_indices, _ = build_current_compute_indices(
                        current_video_indices,
                        video_seq_len=num_new_tokens,
                        action_register_length=action_register_length,
                    )
                else:
                    attention_q = torch.cat(
                        [roped_query, roped_action_query],
                        dim=1,
                    )
                x = self.attn(
                    attention_q,
                    torch.cat([attention_k, roped_action_key], dim=1),
                    torch.cat([attention_v, action_v], dim=1),
                )
            else:
                if new_k is None:
                    new_k = torch.cat([history_k, roped_key], dim=1)
                if new_v is None:
                    new_v = torch.cat([history_v, v], dim=1)
                x = self.attn(
                    roped_query,
                    new_k,
                    new_v,
                )
            if update_kv_cache:
                assert new_k is not None and new_v is not None
                updated_kv_cache = torch.stack([new_k, new_v], dim=0)


        if action_register_length is not None and kv_cache is not None:
            self._observe_dynamic_m1_action_output(
                x,
                action_register_length=action_register_length,
                registers_first=False,
            )

        # Controlled downstream Oracle interventions act on the exact per-head
        # Dense attention output and then retain the released O projection and
        # every remaining layer/denoising step.
        if self.downstream_head_intervention is not None:
            x = self._apply_downstream_head_intervention(
                x,
                action_register_length=action_register_length,
            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        if sparse_query_output_indices is not None:
            full_x = x.new_zeros((b, s, self.dim))
            x = scatter_sequence_by_index(
                full_x,
                sparse_query_output_indices,
                x,
                validate_indices=False,
            )
        return x, updated_kv_cache, updated_anchor_route_indices


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 anchor_sparse_config: AnchorSparseConfig | None = None,
                 record_anchor_diagnostics: bool = False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.sparse_current_compute = False
        self.sparse_current_attention = False
        self.current_propagate_radius = 0

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            anchor_sparse_config=anchor_sparse_config,
            record_anchor_diagnostics=record_anchor_diagnostics,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        anchor_route_indices: torch.Tensor | None = None,
        current_video_indices: torch.Tensor | None = None,
        current_attention_query_indices: torch.Tensor | None = None,
        update_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Align modulation sequence length to x so mul/add broadcast (e.g. when F != L under compile)
        L = x.shape[1]
        aligned = []
        for part in e:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e = tuple(aligned)

        # self-attention
        sparse_self_attention = (
            self.sparse_current_attention
            and current_attention_query_indices is not None
            and action_conditioned_causal_routing_is_eligible(
                current_start_frame,
                action_register_length,
            )
        )
        y, updated_kv_cache, updated_anchor_route_indices = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
            anchor_route_indices=anchor_route_indices,
            current_video_indices=(
                current_attention_query_indices if sparse_self_attention else None
            ),
            update_kv_cache=update_kv_cache,
        )
        x = x + (y * e[2].squeeze(2))

        # cross-attention & ffn function
        def cross_attn_ffn(x, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn(
                (self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            )
            x = x + (y * e[5].squeeze(2))
            return x

        if (
            self.sparse_current_compute
            and current_video_indices is not None
            and action_conditioned_causal_routing_is_eligible(
                current_start_frame,
                action_register_length,
            )
        ):
            if self.self_attn.anchor_sparse_config is None:
                raise RuntimeError("Sparse current-token compute requires an anchor config")
            x = sparse_current_token_update(
                x=x,
                e=e,
                current_video_indices=current_video_indices,
                action_register_length=action_register_length,
                anchor_sparse_config=self.self_attn.anchor_sparse_config,
                propagate_radius=self.current_propagate_radius,
                update_fn=cross_attn_ffn,
            )
        else:
            x = cross_attn_ffn(x, e)
        return x, updated_kv_cache, updated_anchor_route_indices

    def forward_packed(
        self,
        x: torch.Tensor,
        e0: torch.Tensor,
        packed_freqs: torch.Tensor,
        *,
        action_register_length: int,
        context: torch.Tensor,
        kv_cache: torch.Tensor,
        history_indices: torch.Tensor,
        history_token_count: int,
        head_groups: (
            tuple[tuple[tuple[int, ...], float, float | None], ...] | None
        ) = None,
        history_indices_by_ratio: dict[float, torch.Tensor] | None = None,
        current_video_tokens_by_ratio: dict[float, int] | None = None,
        maximum_current_x: torch.Tensor | None = None,
        maximum_current_e0: torch.Tensor | None = None,
        maximum_current_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Keep a packed current-token state through one complete DiT block."""

        if e0.shape != (x.shape[0], x.shape[1], 6, self.dim):
            raise ValueError("packed e0 must have shape [B, L, 6, C]")
        modulation = (self.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
        maximum_attention_x: torch.Tensor | None = None
        if maximum_current_x is not None:
            if maximum_current_e0 is None or maximum_current_freqs is None:
                raise ValueError(
                    "Maximum action-current inputs require state, modulation, and RoPE"
                )
            if maximum_current_e0.shape != (
                maximum_current_x.shape[0],
                maximum_current_x.shape[1],
                6,
                self.dim,
            ):
                raise ValueError(
                    "maximum_current_e0 must align with maximum_current_x"
                )
            maximum_modulation = (
                self.modulation.unsqueeze(1) + maximum_current_e0
            ).chunk(6, dim=2)
            maximum_attention_x = self.norm1(maximum_current_x) * (
                1 + maximum_modulation[1].squeeze(2)
            ) + maximum_modulation[0].squeeze(2)
        y = self.self_attn.forward_packed(
            self.norm1(x) * (1 + modulation[1].squeeze(2))
            + modulation[0].squeeze(2),
            packed_freqs,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            history_indices=history_indices,
            history_token_count=history_token_count,
            head_groups=head_groups,
            history_indices_by_ratio=history_indices_by_ratio,
            current_video_tokens_by_ratio=current_video_tokens_by_ratio,
            maximum_current_x=maximum_attention_x,
            maximum_current_freqs=maximum_current_freqs,
        )
        x = x + y * modulation[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(
            self.norm2(x) * (1 + modulation[4].squeeze(2))
            + modulation[3].squeeze(2)
        )
        return x + y * modulation[5].squeeze(2)


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        # Align modulation sequence length to x (e.g. when F != L1 under compile)
        L = x.shape[1]
        aligned = []
        for part in e:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e = tuple(aligned)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 frame_seqlen=220,
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 max_chunk_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1, 
                 action_dim=32,
                 num_registers=8,
                 max_state_dim=64,
                 max_num_embodiments=32,
                 hidden_size=1024,
                 diffusion_model_pretrained_path=None,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 concat_first_frame_latent=True,
                 anchor_sparse_enabled=False,
                 anchor_sparse_keep_ratio=0.25,
                 anchor_sparse_recent_dense_frames=2,
                 anchor_sparse_probe_dim=16,
                 anchor_sparse_num_router_heads=4,
                 anchor_sparse_smooth_radius=1,
                 anchor_sparse_current_keep_ratio=1.0,
                 anchor_sparse_attention_query_keep_ratio=None,
                 anchor_sparse_dense_prefix_layers=1,
                 anchor_sparse_dense_suffix_layers=1,
                 anchor_sparse_propagate_radius=0,
                 anchor_sparse_propagate_every=1,
                 anchor_sparse_reuse_denoise=True,
                 anchor_sparse_current_attention=False,
                 anchor_sparse_packed_middle=False,
                 anchor_sparse_record_diagnostics=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            concat_first_frame_latent (`bool`, *optional*, defaults to True):
                If True, concat [x; y] before patch_embedding (14B I2V style). If False, latent only (5B pretrained style; first-frame via CLIP).
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent
        self.anchor_sparse_enabled = anchor_sparse_enabled
        self.anchor_sparse_current_keep_ratio = anchor_sparse_current_keep_ratio
        if anchor_sparse_attention_query_keep_ratio is None:
            anchor_sparse_attention_query_keep_ratio = anchor_sparse_current_keep_ratio
        packed_middle_active = (
            anchor_sparse_enabled
            and anchor_sparse_packed_middle
            and (
                anchor_sparse_keep_ratio < 1.0
                or anchor_sparse_current_keep_ratio < 1.0
            )
        )
        self.anchor_sparse_attention_query_keep_ratio = (
            anchor_sparse_attention_query_keep_ratio
        )
        self.anchor_sparse_dense_prefix_layers = anchor_sparse_dense_prefix_layers
        self.anchor_sparse_dense_suffix_layers = anchor_sparse_dense_suffix_layers
        self.anchor_sparse_propagate_radius = anchor_sparse_propagate_radius
        self.anchor_sparse_propagate_every = anchor_sparse_propagate_every
        self.anchor_sparse_reuse_denoise = anchor_sparse_reuse_denoise
        self.anchor_sparse_current_attention = anchor_sparse_current_attention
        self.anchor_sparse_packed_middle = packed_middle_active
        self.anchor_sparse_record_diagnostics = anchor_sparse_record_diagnostics
        self.anchor_sparse_dense_action_history = False
        self.anchor_sparse_max_action_current = False

        if not 0.0 < anchor_sparse_current_keep_ratio <= 1.0:
            raise ValueError("anchor_sparse_current_keep_ratio must lie in (0, 1]")
        if not 0.0 < anchor_sparse_attention_query_keep_ratio <= 1.0:
            raise ValueError(
                "anchor_sparse_attention_query_keep_ratio must lie in (0, 1]"
            )
        if (
            packed_middle_active
            and anchor_sparse_attention_query_keep_ratio
            != anchor_sparse_current_keep_ratio
        ):
            raise ValueError(
                "Packed Middle Stack uses one shared current-token shape; "
                "attention_query_keep_ratio must equal current_keep_ratio"
            )
        if anchor_sparse_dense_prefix_layers < 0 or anchor_sparse_dense_suffix_layers < 0:
            raise ValueError("Dense prefix/suffix layer counts must be non-negative")
        needs_current_route = anchor_sparse_current_keep_ratio < 1.0 or (
            anchor_sparse_current_attention
            and anchor_sparse_attention_query_keep_ratio < 1.0
        ) or (
            packed_middle_active
            and anchor_sparse_keep_ratio < 1.0
        )
        if needs_current_route and anchor_sparse_dense_prefix_layers < 1:
            raise ValueError("Current-token routing requires at least one dense prefix layer")
        if anchor_sparse_dense_prefix_layers + anchor_sparse_dense_suffix_layers > num_layers:
            raise ValueError("Dense prefix/suffix layers exceed the transformer depth")
        if anchor_sparse_propagate_radius < 0:
            raise ValueError("anchor_sparse_propagate_radius must be non-negative")
        if anchor_sparse_propagate_radius > 0 and anchor_sparse_propagate_every <= 0:
            raise ValueError("anchor_sparse_propagate_every must be positive when propagation is enabled")
        if anchor_sparse_enabled and frame_seqlen != 880:
            raise ValueError(
                "The initial embodied anchor router supports the released DreamZero-DROID "
                f"22x40 token layout (frame_seqlen=880), got {frame_seqlen}."
            )
        self.anchor_sparse_config = None
        if anchor_sparse_enabled:
            self.anchor_sparse_config = AnchorSparseConfig(
                frame_seqlen=frame_seqlen,
                grid_height=22,
                grid_width=40,
                keep_ratio=anchor_sparse_keep_ratio,
                recent_dense_frames=anchor_sparse_recent_dense_frames,
                probe_dim=anchor_sparse_probe_dim,
                num_router_heads=anchor_sparse_num_router_heads,
                smooth_radius=anchor_sparse_smooth_radius,
                views=droid_composite_view_regions(),
            )

        # A route contains only stable token positions, not activations.  It is
        # shared across heads/layers and optionally across denoising calls for
        # the same causal control block.  These are intentionally plain Python
        # attributes rather than state-dict buffers.
        self._anchor_sparse_route_cache: torch.Tensor | None = None
        self._anchor_sparse_current_route_cache: torch.Tensor | None = None
        self._anchor_sparse_current_attention_route_cache: torch.Tensor | None = None
        self._anchor_sparse_route_cache_key: tuple[Any, ...] | None = None
        self._anchor_sparse_packed_profile_cache_key: tuple[Any, ...] | None = None
        self._anchor_sparse_packed_current_profile: NestedAnchorProfile | None = None
        self._anchor_sparse_packed_history_profile: NestedAnchorProfile | None = None
        self._anchor_sparse_packed_history_indices: dict[float, torch.Tensor] = {}
        self._anchor_sparse_last_start_frame: int | None = None
        self._dynamic_packed_budget_table: DynamicPackedBudgetTable | None = None
        self._dynamic_packed_head_group_budget_table: (
            DynamicPackedHeadGroupBudgetTable | None
        ) = None
        self._dynamic_dense_action_history_table: (
            DynamicDenseActionHistoryTable | None
        ) = None
        self._dynamic_max_action_current_table: (
            DynamicDenseActionHistoryTable | None
        ) = None
        self._dynamic_sparse_dit_index: int | None = None
        self._dynamic_sparse_scheduler_index: int | None = None
        self._dynamic_sparse_scheduler_steps: int | None = None
        self._dynamic_sparse_force_dense = False
        self._anchor_sparse_last_packed_propagation_count = 0
        self._dynamic_attention_oracle_collector: Any | None = None
        self._dynamic_downstream_head_intervention: (
            DownstreamHeadIntervention | None
        ) = None
        self._dynamic_attention_oracle_cfg_branch: str | None = None
        self._dynamic_m1_packed_observer: Any | None = None
        self._dynamic_m1_packed_observations: list[Any | None] = []
        self._dynamic_m1_runtime: Any | None = None
        self._dynamic_m1_layer_shared_current_keep_ratio: float | None = None
        self._dynamic_m1_shared_budget_promotion = False
        self._dynamic_m1_shared_budget_trace: dict[
            tuple[int, int], dict[str, object]
        ] = {}
        self._dynamic_m1_request_condition: tuple[float, float] | None = None

        max_num_embodiments = 1

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        block_anchor_sparse_config = self.anchor_sparse_config
        if packed_middle_active and block_anchor_sparse_config is not None:
            # Prefix/suffix blocks stay genuinely Dense.  Layer zero still
            # records action-conditioned scores, but its all-token route avoids
            # historical KV gathering before the packed boundary.
            block_anchor_sparse_config = replace(
                block_anchor_sparse_config,
                keep_ratio=1.0,
            )
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                frame_seqlen,
                self.local_attn_size,
                sink_size,
                num_frame_per_block,
                qk_norm,
                cross_attn_norm,
                eps,
                num_action_per_block,
                num_state_per_block,
                block_anchor_sparse_config,
                (
                    anchor_sparse_record_diagnostics
                    or packed_middle_active
                    or anchor_sparse_current_keep_ratio < 1.0
                    or (
                        anchor_sparse_current_attention
                        and anchor_sparse_attention_query_keep_ratio < 1.0
                    )
                ) and block_index == 0,
            )
            for block_index in range(num_layers)
        ])
        self._packed_head_projection_signatures: list[
            tuple[tuple[int, ...], ...] | None
        ] = [None] * len(self.blocks)
        sparse_end = num_layers - anchor_sparse_dense_suffix_layers
        for block_index, block in enumerate(self.blocks):
            block.self_attn.layer_index = block_index
            block.sparse_current_compute = (
                anchor_sparse_enabled
                and not packed_middle_active
                and anchor_sparse_current_keep_ratio < 1.0
                and anchor_sparse_dense_prefix_layers <= block_index < sparse_end
            )
            block.sparse_current_attention = (
                anchor_sparse_enabled
                and not packed_middle_active
                and anchor_sparse_current_attention
                and anchor_sparse_attention_query_keep_ratio < 1.0
                and anchor_sparse_dense_prefix_layers <= block_index < sparse_end
            )
            sparse_offset = block_index - anchor_sparse_dense_prefix_layers + 1
            should_propagate = (
                block.sparse_current_compute
                and anchor_sparse_propagate_radius > 0
                and (
                    sparse_offset % anchor_sparse_propagate_every == 0
                    or block_index == sparse_end - 1
                )
            )
            block.current_propagate_radius = (
                anchor_sparse_propagate_radius if should_propagate else 0
            )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        
        self.freqs_action = rope_params(1024*10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        if model_type in ('i2v', 'ti2v'):
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = True
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True


    def clear_anchor_sparse_route_cache(self) -> None:
        """Clear per-control-block anchor indices, e.g. at an episode reset."""

        self._anchor_sparse_route_cache = None
        self._anchor_sparse_current_route_cache = None
        self._anchor_sparse_current_attention_route_cache = None
        self._anchor_sparse_route_cache_key = None
        self._anchor_sparse_packed_profile_cache_key = None
        self._anchor_sparse_packed_current_profile = None
        self._anchor_sparse_packed_history_profile = None
        self._anchor_sparse_packed_history_indices = {}
        self._anchor_sparse_last_start_frame = None
        for block in self.blocks:
            block.self_attn.last_anchor_route = None
            block.self_attn.clear_anchor_sparse_history_cache()

    def configure_dynamic_attention_oracle(
        self,
        *,
        output_dir: str | None,
        rank: int = 0,
        keep_ratios: tuple[float, ...] = (1.0, 0.75, 0.50, 0.35, 0.25, 0.20, 0.10),
        max_video_queries: int | None = 32,
        max_action_queries: int | None = None,
        query_chunk_size: int = 4,
        support_ratio: float = 0.75,
        layer_indices: tuple[int, ...] = (),
        task_id: str | None = None,
        trajectory_stage: str | None = None,
    ) -> None:
        """Enable or disable offline dense per-head Oracle collection."""

        if output_dir is None:
            collector = None
        else:
            from pathlib import Path

            from groot.vla.model.dreamzero.modules.dynamic_attention_oracle import (
                DenseAttentionOracleCollector,
                DenseAttentionOracleConfig,
            )

            collector = DenseAttentionOracleCollector(
                DenseAttentionOracleConfig(
                    output_dir=Path(output_dir),
                    rank=rank,
                    keep_ratios=keep_ratios,
                    max_video_queries=max_video_queries,
                    max_action_queries=max_action_queries,
                    query_chunk_size=query_chunk_size,
                    support_ratio=support_ratio,
                    layer_indices=layer_indices,
                    task_id=task_id,
                    trajectory_stage=trajectory_stage,
                )
            )
        self._dynamic_attention_oracle_collector = collector
        for block_index, block in enumerate(self.blocks):
            block.self_attn.layer_index = block_index
            block.self_attn.dynamic_oracle_collector = collector

    def set_dynamic_attention_oracle_request_metadata(
        self,
        *,
        task_id: str | None = None,
        trajectory_stage: str | None = None,
        sample_metadata: dict[str, object] | None = None,
    ) -> None:
        collector = self._dynamic_attention_oracle_collector
        if collector is not None:
            collector.set_next_request_metadata(
                task_id=task_id,
                trajectory_stage=trajectory_stage,
                sample_metadata=sample_metadata,
            )

    def begin_dynamic_attention_oracle_request(
        self,
        *,
        current_start_frame: int,
        instruction: object | None = None,
        task_id: str | None = None,
        trajectory_stage: str | None = None,
    ) -> None:
        self._dynamic_sparse_force_dense = False
        self._dynamic_sparse_dit_index = None
        self._dynamic_sparse_scheduler_index = None
        self._dynamic_sparse_scheduler_steps = None
        self._dynamic_attention_oracle_cfg_branch = None
        self._dynamic_m1_packed_observations = []
        self._dynamic_m1_shared_budget_trace = {}
        runtime = self._dynamic_m1_runtime
        if runtime is not None:
            condition = self._dynamic_m1_request_condition
            runtime.begin_request(
                state_l2=None if condition is None else condition[0],
                state_abs_mean=None if condition is None else condition[1],
            )
        elif self._dynamic_m1_packed_observer is not None:
            self._dynamic_m1_packed_observer.begin_request()
        self._dynamic_m1_request_condition = None
        intervention = self._dynamic_downstream_head_intervention
        if intervention is not None:
            block = self.blocks[intervention.layer_index]
            block.self_attn.downstream_intervention_dit_index = None
            block.self_attn.downstream_intervention_cfg_branch = None
            block.self_attn.downstream_head_intervention_count = 0
        collector = self._dynamic_attention_oracle_collector
        if collector is not None:
            collector.begin_request(
                current_start_frame=current_start_frame,
                instruction=instruction,
                task_id=task_id,
                trajectory_stage=trajectory_stage,
            )

    def set_dynamic_attention_oracle_step(
        self,
        *,
        scheduler_index: int,
        dit_index: int,
        scheduler_steps: int,
        timestep: int | torch.Tensor,
    ) -> None:
        # The action head invokes this setter immediately before every real DiT
        # evaluation even when Oracle collection is disabled.  Reuse that exact
        # execution context for dynamic sparse budgets; skipped scheduler steps
        # never advance ``dit_index``.
        runtime = self._dynamic_m1_runtime
        observer = self._dynamic_m1_packed_observer
        if runtime is not None:
            if torch.is_tensor(timestep):
                diffusion_timestep = int(timestep.reshape(-1)[0].item())
            else:
                diffusion_timestep = int(timestep)
            previous_observation, _decision = runtime.begin_step(
                scheduler_index=int(scheduler_index),
                dit_index=int(dit_index),
                scheduler_steps=int(scheduler_steps),
                diffusion_timestep=diffusion_timestep,
            )
            if dit_index > 0:
                self._dynamic_m1_packed_observations.append(previous_observation)
            if len(self._dynamic_m1_packed_observations) != dit_index:
                raise RuntimeError(
                    "Dynamic M1 observations do not align with real DiT order"
                )
        elif observer is not None:
            if observer.step_active:
                self._dynamic_m1_packed_observations.append(
                    observer.finish_step()
                )
            if len(self._dynamic_m1_packed_observations) != dit_index:
                raise RuntimeError(
                    "Packed M1 observations do not align with real DiT order"
                )
            observer.begin_step(int(dit_index))
        self._dynamic_sparse_scheduler_index = int(scheduler_index)
        self._dynamic_sparse_dit_index = int(dit_index)
        self._dynamic_sparse_scheduler_steps = int(scheduler_steps)
        self._evict_stale_packed_head_projection_weights()
        intervention = self._dynamic_downstream_head_intervention
        if intervention is not None:
            self.blocks[
                intervention.layer_index
            ].self_attn.downstream_intervention_dit_index = int(dit_index)
        collector = self._dynamic_attention_oracle_collector
        if collector is not None:
            collector.set_step(
                scheduler_index=scheduler_index,
                dit_index=dit_index,
                scheduler_steps=scheduler_steps,
                timestep=timestep,
            )

    def set_dynamic_attention_oracle_cfg_branch(self, branch: str | None) -> None:
        self._dynamic_attention_oracle_cfg_branch = branch
        for block in self.blocks:
            block.self_attn.dynamic_m1_cfg_branch = branch
        intervention = self._dynamic_downstream_head_intervention
        if intervention is not None:
            self.blocks[
                intervention.layer_index
            ].self_attn.downstream_intervention_cfg_branch = branch
        collector = self._dynamic_attention_oracle_collector
        if collector is not None:
            collector.set_cfg_branch(branch)

    def configure_dynamic_downstream_head_intervention(
        self,
        intervention: DownstreamHeadIntervention | None,
    ) -> None:
        """Configure a Dense per-head intervention for downstream sensitivity."""

        if intervention is not None:
            if self.anchor_sparse_enabled:
                raise ValueError(
                    "Downstream head interventions require the exact Dense path"
                )
            if intervention.dit_index >= 8:
                raise ValueError(
                    "DreamZero downstream interventions require a real DiT index in [0, 7]"
                )
            if intervention.layer_index >= len(self.blocks):
                raise ValueError("intervention layer exceeds model depth")
            if any(index >= self.num_heads for index in intervention.head_indices):
                raise ValueError("intervention head exceeds model head count")

        previous = self._dynamic_downstream_head_intervention
        if previous is not None:
            previous_attention = self.blocks[previous.layer_index].self_attn
            previous_attention.downstream_head_intervention = None
            previous_attention.downstream_intervention_dit_index = None
            previous_attention.downstream_intervention_cfg_branch = None
            previous_attention.downstream_head_intervention_count = 0
        if intervention is not None:
            attention = self.blocks[intervention.layer_index].self_attn
            attention.downstream_head_intervention = intervention
            attention.downstream_intervention_dit_index = (
                self._dynamic_sparse_dit_index
            )
            attention.downstream_intervention_cfg_branch = (
                self._dynamic_attention_oracle_cfg_branch
            )
            attention.downstream_head_intervention_count = 0
        self._dynamic_downstream_head_intervention = intervention

    def get_dynamic_downstream_head_intervention_trace(self) -> dict[str, object]:
        intervention = self._dynamic_downstream_head_intervention
        if intervention is None:
            return {"configured": False, "applied_count": 0}
        attention = self.blocks[intervention.layer_index].self_attn
        return {
            "configured": True,
            "dit_index": intervention.dit_index,
            "layer_index": intervention.layer_index,
            "head_indices": list(intervention.head_indices),
            "scale": intervention.scale,
            "cfg_branches": list(intervention.cfg_branches),
            "query_scope": intervention.query_scope,
            "applied_count": attention.downstream_head_intervention_count,
        }

    def flush_dynamic_attention_oracle_request(self):
        runtime = self._dynamic_m1_runtime
        observer = self._dynamic_m1_packed_observer
        if runtime is not None:
            runtime.finish_request()
            self._dynamic_m1_packed_observations = list(runtime.observations)
        elif observer is not None:
            if observer.step_active:
                self._dynamic_m1_packed_observations.append(
                    observer.finish_step()
                )
            observer.end_request()
        collector = self._dynamic_attention_oracle_collector
        if collector is None:
            return None
        return collector.flush_request()

    def configure_dynamic_m1_packed_observer(self, observer: Any | None) -> None:
        """Attach a low-overhead causal observer to Dense and Packed paths."""

        runtime = self._dynamic_m1_runtime
        if runtime is not None and observer is not runtime.observer:
            raise ValueError(
                "A Dynamic M1 runtime owns its Packed observer; detach it first"
            )
        if observer is not None:
            if observer.num_layers != len(self.blocks):
                raise ValueError("Packed M1 observer layer count differs from model")
            if observer.num_heads != self.num_heads:
                raise ValueError("Packed M1 observer Head count differs from model")
        self._dynamic_m1_packed_observer = observer
        self._dynamic_m1_packed_observations = []
        for block in self.blocks:
            block.self_attn.dynamic_m1_packed_observer = observer
            block.self_attn.dynamic_m1_cfg_branch = None

    def set_dynamic_m1_request_condition(
        self,
        *,
        state_l2: float | None,
        state_abs_mean: float | None,
    ) -> None:
        """Stage raw, pre-normalization state statistics for the next request."""

        if state_l2 is None or state_abs_mean is None:
            self._dynamic_m1_request_condition = None
            return
        values = (float(state_l2), float(state_abs_mean))
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            self._dynamic_m1_request_condition = None
            return
        self._dynamic_m1_request_condition = values

    def configure_dynamic_m1_runtime(
        self,
        runtime: Any | None,
        *,
        layer_shared_current_keep_ratio: float | None = None,
        shared_budget_promotion: bool = False,
    ) -> None:
        """Attach causal M1 decisions directly to the Packed-M2 executor."""

        previous = self._dynamic_m1_runtime
        shared_budget_promotion = bool(shared_budget_promotion)
        if layer_shared_current_keep_ratio is not None and (
            float(layer_shared_current_keep_ratio) not in (0.25, 0.50, 0.75, 1.00)
        ):
            raise ValueError(
                "Dynamic M1 layer-shared current ratio must use an executor bucket"
            )
        if shared_budget_promotion and layer_shared_current_keep_ratio is not None:
            raise ValueError(
                "Dynamic M1 shared budget promotion already controls the current budget"
            )
        if runtime is not None:
            if not self.anchor_sparse_packed_middle:
                raise ValueError("Dynamic M1 runtime requires Packed Middle Stack")
            if self.anchor_sparse_reuse_denoise:
                raise ValueError(
                    "Packed-proxy M1 requires reuse_denoise=False so each real DiT "
                    "observes a fresh action-conditioned route"
                )
            if self.anchor_sparse_dense_action_history:
                raise ValueError("Dynamic M1 Head groups do not support dense action history")
            if self.anchor_sparse_max_action_current:
                raise ValueError(
                    "Dynamic M1 Head groups do not support maximum action-current K/V"
                )
            if (
                self._dynamic_packed_budget_table is not None
                and not shared_budget_promotion
            ):
                raise ValueError("Dynamic M1 runtime conflicts with a static budget table")
            if self._dynamic_packed_head_group_budget_table is not None:
                raise ValueError("Dynamic M1 runtime conflicts with a static Head-group table")
            if runtime.num_dit_steps != 8:
                raise ValueError("DreamZero Dynamic M1 must cover exactly 8 real DiTs")
            if runtime.num_layers != len(self.blocks) or runtime.num_heads != self.num_heads:
                raise ValueError("Dynamic M1 runtime geometry differs from the model")
        self._dynamic_m1_runtime = runtime
        self._dynamic_m1_layer_shared_current_keep_ratio = (
            None
            if runtime is None or layer_shared_current_keep_ratio is None
            else float(layer_shared_current_keep_ratio)
        )
        self._dynamic_m1_shared_budget_promotion = bool(
            runtime is not None and shared_budget_promotion
        )
        self._dynamic_m1_shared_budget_trace = {}
        if runtime is not None:
            self.configure_dynamic_m1_packed_observer(runtime.observer)
        elif previous is not None and self._dynamic_m1_packed_observer is previous.observer:
            self.configure_dynamic_m1_packed_observer(None)
        self._packed_head_projection_signatures = [None] * len(self.blocks)
        self.clear_anchor_sparse_route_cache()

    def _evict_stale_packed_head_projection_weights(self) -> None:
        """Retain at most one prepacked QKV/O membership partition per layer."""

        if (
            self._dynamic_m1_runtime is None
            and self._dynamic_packed_head_group_budget_table is None
        ):
            return
        for layer_index, block in enumerate(self.blocks):
            groups = self._packed_head_groups_for_layer(layer_index)
            signature = (
                None
                if groups is None
                else tuple(tuple(heads) for heads, _, _ in groups)
            )
            if signature == self._packed_head_projection_signatures[layer_index]:
                continue
            block.self_attn.clear_packed_head_projection_weight_cache()
            self._packed_head_projection_signatures[layer_index] = signature

    def get_dynamic_m1_runtime_trace(self) -> dict[str, object]:
        runtime = self._dynamic_m1_runtime
        if runtime is None:
            return {"configured": False}
        shared_cells = tuple(self._dynamic_m1_shared_budget_trace.values())
        shared_summary: dict[str, object] = {
            "enabled": self._dynamic_m1_shared_budget_promotion,
            "aggregation": "maximum_head_budget",
            "observed_cell_count": len(shared_cells),
        }
        if shared_cells:
            shared_summary.update(
                {
                    "history_promoted_cell_count": sum(
                        float(cell["effective_history_keep_ratio"])
                        > float(cell["base_history_keep_ratio"])
                        for cell in shared_cells
                    ),
                    "current_promoted_cell_count": sum(
                        float(cell["effective_current_keep_ratio"])
                        > float(cell["base_current_keep_ratio"])
                        for cell in shared_cells
                    ),
                    "effective_dense_cell_count": sum(
                        float(cell["effective_history_keep_ratio"]) == 1.0
                        and float(cell["effective_current_keep_ratio"]) == 1.0
                        for cell in shared_cells
                    ),
                    "fallback_head_count": sum(
                        int(cell["fallback_head_count"])
                        for cell in shared_cells
                    ),
                    "minimum_route_confidence": min(
                        float(cell["minimum_route_confidence"])
                        for cell in shared_cells
                    ),
                }
            )
        return {
            "configured": True,
            "layer_shared_current_keep_ratio": (
                self._dynamic_m1_layer_shared_current_keep_ratio
            ),
            "shared_budget_promotion": shared_summary,
            **runtime.trace(),
        }

    def get_dynamic_m1_packed_observations(self) -> tuple[Any | None, ...]:
        return tuple(self._dynamic_m1_packed_observations)

    def get_dynamic_attention_oracle_last_flush_paths(self):
        collector = self._dynamic_attention_oracle_collector
        if collector is None:
            return None
        return collector.last_flush_paths

    def configure_anchor_sparse_attention(
        self,
        *,
        enabled: bool,
        keep_ratio: float = 0.25,
        recent_dense_frames: int = 2,
        probe_dim: int = 16,
        num_router_heads: int = 4,
        smooth_radius: int = 1,
        current_keep_ratio: float = 1.0,
        attention_query_keep_ratio: float | None = None,
        dense_prefix_layers: int = 1,
        dense_suffix_layers: int = 1,
        propagate_radius: int = 0,
        propagate_every: int = 1,
        reuse_denoise: bool = True,
        current_attention: bool = False,
        packed_middle: bool = False,
        dense_action_history: bool = False,
        max_action_current: bool = False,
        record_diagnostics: bool = False,
    ) -> None:
        """Configure sparse routing after loading an upstream checkpoint.

        DreamZero nests the diffusion model inside the action-head config, so a
        post-load API is substantially less brittle than rewriting checkpoint
        JSON.  It also guarantees that every existing transformer block receives
        the same immutable route configuration.
        """

        if enabled and self._dynamic_downstream_head_intervention is not None:
            raise ValueError(
                "Disable downstream head intervention before enabling sparse attention"
            )
        if enabled and self.frame_seqlen != 880:
            raise ValueError(
                "Embodied anchor routing currently supports only the released "
                f"DreamZero-DROID 22x40 layout, got frame_seqlen={self.frame_seqlen}."
            )
        if not 0.0 < current_keep_ratio <= 1.0:
            raise ValueError("current_keep_ratio must lie in (0, 1]")
        if attention_query_keep_ratio is None:
            attention_query_keep_ratio = current_keep_ratio
        packed_middle_active = (
            enabled
            and packed_middle
            and (keep_ratio < 1.0 or current_keep_ratio < 1.0)
        )
        if not 0.0 < attention_query_keep_ratio <= 1.0:
            raise ValueError("attention_query_keep_ratio must lie in (0, 1]")
        if packed_middle_active and attention_query_keep_ratio != current_keep_ratio:
            raise ValueError(
                "Packed Middle Stack uses one shared current-token shape; "
                "attention_query_keep_ratio must equal current_keep_ratio"
            )
        if dense_prefix_layers < 0 or dense_suffix_layers < 0:
            raise ValueError("Dense prefix/suffix layer counts must be non-negative")
        needs_current_route = current_keep_ratio < 1.0 or (
            current_attention and attention_query_keep_ratio < 1.0
        ) or (
            packed_middle_active and keep_ratio < 1.0
        )
        if needs_current_route and dense_prefix_layers < 1:
            raise ValueError("Current-token routing requires at least one dense prefix layer")
        if dense_prefix_layers + dense_suffix_layers > len(self.blocks):
            raise ValueError("Dense prefix/suffix layers exceed the transformer depth")
        if propagate_radius < 0:
            raise ValueError("propagate_radius must be non-negative")
        if propagate_radius > 0 and propagate_every <= 0:
            raise ValueError("propagate_every must be positive when propagation is enabled")
        config = None
        if enabled:
            config = AnchorSparseConfig(
                frame_seqlen=self.frame_seqlen,
                grid_height=22,
                grid_width=40,
                keep_ratio=keep_ratio,
                recent_dense_frames=recent_dense_frames,
                probe_dim=probe_dim,
                num_router_heads=num_router_heads,
                smooth_radius=smooth_radius,
                views=droid_composite_view_regions(),
            )

        self.anchor_sparse_enabled = enabled
        self.anchor_sparse_config = config
        self.anchor_sparse_current_keep_ratio = current_keep_ratio
        self.anchor_sparse_attention_query_keep_ratio = attention_query_keep_ratio
        self.anchor_sparse_dense_prefix_layers = dense_prefix_layers
        self.anchor_sparse_dense_suffix_layers = dense_suffix_layers
        self.anchor_sparse_propagate_radius = propagate_radius
        self.anchor_sparse_propagate_every = propagate_every
        self.anchor_sparse_reuse_denoise = reuse_denoise
        self.anchor_sparse_current_attention = current_attention
        self.anchor_sparse_packed_middle = packed_middle_active
        self.anchor_sparse_dense_action_history = (
            enabled and packed_middle_active and dense_action_history
        )
        self.anchor_sparse_max_action_current = (
            enabled and packed_middle_active and max_action_current
        )
        self.anchor_sparse_record_diagnostics = record_diagnostics
        if not packed_middle_active:
            self._dynamic_packed_budget_table = None
            self._dynamic_packed_head_group_budget_table = None
            self._dynamic_dense_action_history_table = None
            self._dynamic_max_action_current_table = None
        elif not self.anchor_sparse_dense_action_history:
            self._dynamic_dense_action_history_table = None
        if packed_middle_active and not self.anchor_sparse_max_action_current:
            self._dynamic_max_action_current_table = None
        sparse_end = len(self.blocks) - dense_suffix_layers
        block_config = config
        if packed_middle_active and block_config is not None:
            block_config = replace(block_config, keep_ratio=1.0)
        for block_index, block in enumerate(self.blocks):
            block.self_attn.anchor_sparse_config = block_config
            block.self_attn.packed_dense_action_history = (
                self.anchor_sparse_dense_action_history
                and dense_prefix_layers <= block_index < sparse_end
            )
            block.self_attn.packed_max_action_current = (
                self.anchor_sparse_max_action_current
                and dense_prefix_layers <= block_index < sparse_end
            )
            block.self_attn.record_anchor_diagnostics = (
                enabled
                and (
                    record_diagnostics
                    or packed_middle_active
                    or current_keep_ratio < 1.0
                    or (current_attention and attention_query_keep_ratio < 1.0)
                )
                and block_index == 0
            )
            block.sparse_current_compute = (
                enabled
                and not packed_middle_active
                and current_keep_ratio < 1.0
                and dense_prefix_layers <= block_index < sparse_end
            )
            block.sparse_current_attention = (
                enabled
                and not packed_middle_active
                and current_attention
                and attention_query_keep_ratio < 1.0
                and dense_prefix_layers <= block_index < sparse_end
            )
            sparse_offset = block_index - dense_prefix_layers + 1
            should_propagate = (
                block.sparse_current_compute
                and propagate_radius > 0
                and (
                    sparse_offset % propagate_every == 0
                    or block_index == sparse_end - 1
                )
            )
            block.current_propagate_radius = (
                propagate_radius if should_propagate else 0
            )
        self.clear_anchor_sparse_route_cache()

    def configure_dynamic_packed_budget_table(
        self,
        table: DynamicPackedBudgetTable | None,
    ) -> None:
        """Attach an eight-DiT by layer fixed-bucket budget table."""

        if table is not None:
            if (
                self._dynamic_m1_runtime is not None
                and not self._dynamic_m1_shared_budget_promotion
            ):
                raise ValueError("Static dynamic budgets conflict with Dynamic M1 runtime")
            if not self.anchor_sparse_packed_middle:
                raise ValueError(
                    "Dynamic packed budgets require an active Packed Middle Stack"
                )
            if table.num_dit_steps != 8:
                raise ValueError(
                    "DreamZero main-result budgets must cover exactly 8 real DiT evaluations"
                )
            if table.num_layers != len(self.blocks):
                raise ValueError(
                    "Dynamic packed budget layer count differs from the model depth"
                )
        self._dynamic_packed_budget_table = table
        self.clear_anchor_sparse_route_cache()

    def configure_dynamic_packed_head_group_budget_table(
        self,
        table: DynamicPackedHeadGroupBudgetTable | None,
    ) -> None:
        """Attach per-head history budgets collapsed to at most four groups."""

        if table is not None:
            if self._dynamic_m1_runtime is not None:
                raise ValueError("Static Head groups conflict with Dynamic M1 runtime")
            if not self.anchor_sparse_packed_middle:
                raise ValueError(
                    "Dynamic packed head groups require an active Packed Middle Stack"
                )
            if self.anchor_sparse_dense_action_history:
                raise ValueError(
                    "Dense action history currently requires one shared head group"
                )
            if self.anchor_sparse_max_action_current:
                raise ValueError(
                    "Maximum action-current K/V currently requires one shared head group"
                )
            if table.num_dit_steps != 8:
                raise ValueError(
                    "DreamZero head-group budgets must cover exactly 8 real DiT evaluations"
                )
            if table.num_layers != len(self.blocks):
                raise ValueError(
                    "Dynamic head-group layer count differs from the model depth"
                )
            if table.num_groups > 4:
                raise ValueError("Packed M2 supports at most four shared head groups")
            if table.num_heads != self.num_heads:
                raise ValueError(
                    "Dynamic head-group table width differs from the model head count"
                )
        self._dynamic_packed_head_group_budget_table = table
        self._packed_head_projection_signatures = [None] * len(self.blocks)
        self.clear_anchor_sparse_route_cache()

    def configure_dynamic_dense_action_history_table(
        self,
        table: DynamicDenseActionHistoryTable | None,
    ) -> None:
        """Attach an eight-DiT by layer action-history protection schedule."""

        if table is not None:
            if not self.anchor_sparse_packed_middle:
                raise ValueError(
                    "Dynamic action history requires an active Packed Middle Stack"
                )
            if not self.anchor_sparse_dense_action_history:
                raise ValueError(
                    "Enable dense_action_history before attaching its dynamic table"
                )
            if table.num_dit_steps != 8:
                raise ValueError(
                    "DreamZero action-history schedules must cover exactly 8 real DiT evaluations"
                )
            if table.num_layers != len(self.blocks):
                raise ValueError(
                    "Dynamic action-history layer count differs from the model depth"
                )
            if self._dynamic_packed_head_group_budget_table is not None:
                raise ValueError(
                    "Dynamic action history currently requires one shared head group"
                )
        self._dynamic_dense_action_history_table = table

    def _packed_dense_action_history_for_layer(self, layer_index: int) -> bool:
        if not self.anchor_sparse_dense_action_history:
            return False
        table = self._dynamic_dense_action_history_table
        if table is None:
            return True
        if self._dynamic_sparse_dit_index is None:
            raise RuntimeError(
                "Dynamic action-history table is active before the real DiT index was set"
            )
        return table.enabled(self._dynamic_sparse_dit_index, layer_index)

    def configure_dynamic_max_action_current_table(
        self,
        table: DynamicDenseActionHistoryTable | None,
    ) -> None:
        """Attach an eight-DiT by layer maximum-current action K/V schedule."""

        if table is not None:
            if not self.anchor_sparse_packed_middle:
                raise ValueError(
                    "Dynamic maximum action-current K/V requires Packed Middle Stack"
                )
            if not self.anchor_sparse_max_action_current:
                raise ValueError(
                    "Enable max_action_current before attaching its dynamic table"
                )
            if table.num_dit_steps != 8:
                raise ValueError(
                    "DreamZero max-action-current schedules must cover exactly 8 DiT evaluations"
                )
            if table.num_layers != len(self.blocks):
                raise ValueError(
                    "Dynamic max-action-current layer count differs from model depth"
                )
            if self._dynamic_packed_head_group_budget_table is not None:
                raise ValueError(
                    "Dynamic max-action-current K/V requires one shared head group"
                )
        self._dynamic_max_action_current_table = table

    def _packed_max_action_current_for_layer(self, layer_index: int) -> bool:
        if not self.anchor_sparse_max_action_current:
            return False
        table = self._dynamic_max_action_current_table
        if table is None:
            return True
        if self._dynamic_sparse_dit_index is None:
            raise RuntimeError(
                "Dynamic max-action-current table is active before the real DiT index was set"
            )
        return table.enabled(self._dynamic_sparse_dit_index, layer_index)

    def set_dynamic_sparse_force_dense(self, enabled: bool) -> None:
        """Temporarily bypass Packed M2 for a sentinel Dense recomputation."""

        self._dynamic_sparse_force_dense = bool(enabled)

    def _packed_budget_ratios_for_layer(
        self,
        layer_index: int,
    ) -> tuple[float, float]:
        if self._dynamic_sparse_force_dense:
            return 1.0, 1.0
        runtime = self._dynamic_m1_runtime
        shared_current = self._dynamic_m1_layer_shared_current_keep_ratio
        table = self._dynamic_packed_budget_table
        if self._dynamic_sparse_dit_index is None and table is not None:
            raise RuntimeError(
                "Dynamic packed budget table is active before the real DiT index was set"
            )
        config = self.anchor_sparse_config
        if config is None:
            raise RuntimeError("Packed budgets require an anchor sparse config")
        base_ratios = (
            (config.keep_ratio, self.anchor_sparse_current_keep_ratio)
            if table is None
            else table.ratios(self._dynamic_sparse_dit_index, layer_index)
        )
        if runtime is not None and self._dynamic_m1_shared_budget_promotion:
            decision = runtime.current_decision
            if decision is None:
                raise RuntimeError(
                    "Dynamic M1 runtime is active before the real DiT was routed"
                )
            layer_keep_ratios = decision.keep_ratios[layer_index]
            promotion_ratio = max(float(value) for value in layer_keep_ratios)
            effective_ratios = (
                max(base_ratios[0], promotion_ratio),
                max(base_ratios[1], promotion_ratio),
            )
            if self._dynamic_sparse_dit_index is None:
                raise RuntimeError(
                    "Dynamic M1 shared promotion is active before the real DiT index was set"
                )
            self._dynamic_m1_shared_budget_trace[
                (self._dynamic_sparse_dit_index, layer_index)
            ] = {
                "dit_index": self._dynamic_sparse_dit_index,
                "layer_index": layer_index,
                "base_history_keep_ratio": base_ratios[0],
                "base_current_keep_ratio": base_ratios[1],
                "m1_promotion_keep_ratio": promotion_ratio,
                "effective_history_keep_ratio": effective_ratios[0],
                "effective_current_keep_ratio": effective_ratios[1],
                "fallback_head_count": int(decision.fallback[layer_index].sum()),
                "minimum_route_confidence": float(
                    decision.route_confidence[layer_index].min()
                ),
            }
            return effective_ratios
        if runtime is not None and shared_current is not None:
            decision = runtime.current_decision
            if decision is None:
                raise RuntimeError(
                    "Dynamic M1 runtime is active before the real DiT was routed"
                )
            layer_keep_ratios = decision.keep_ratios[layer_index]
            current_keep_ratio = (
                1.0
                if all(float(value) == 1.0 for value in layer_keep_ratios)
                else shared_current
            )
            return config.keep_ratio, current_keep_ratio
        return base_ratios

    def _packed_head_groups_for_layer(
        self,
        layer_index: int,
    ) -> tuple[tuple[tuple[int, ...], float, float | None], ...] | None:
        if self._dynamic_sparse_force_dense:
            return None
        runtime = self._dynamic_m1_runtime
        if runtime is not None:
            if self._dynamic_m1_shared_budget_promotion:
                return None
            decision = runtime.current_decision
            if decision is None:
                raise RuntimeError(
                    "Dynamic M1 runtime is active before the real DiT was routed"
                )
            return tuple(
                (
                    group["head_indices"],
                    group["history_keep_ratio"],
                    (
                        None
                        if self._dynamic_m1_layer_shared_current_keep_ratio is not None
                        else group["current_keep_ratio"]
                    ),
                )
                for group in decision.execution_groups_for_layer(layer_index)
            )
        table = self._dynamic_packed_head_group_budget_table
        if table is None:
            return None
        if self._dynamic_sparse_dit_index is None:
            raise RuntimeError(
                "Dynamic packed head-group table is active before the real DiT index was set"
            )
        return table.execution_groups_for_layer(
            self._dynamic_sparse_dit_index,
            layer_index,
        )

    def get_last_anchor_route(self) -> AnchorRoute | None:
        """Return the last recorded layer-0 route for heatmap diagnostics."""

        if not self.blocks:
            return None
        return self.blocks[0].self_attn.last_anchor_route

    def _prepare_packed_anchor_profiles(
        self,
        *,
        route: AnchorRoute,
        current_frames: int,
        cache_key: tuple[Any, ...] | None,
        history_keep_ratios: tuple[float, ...],
    ) -> tuple[
        NestedAnchorProfile,
        NestedAnchorProfile,
        dict[float, torch.Tensor],
        int,
    ]:
        """Create or reuse the nested current/history routes for one rollout."""

        config = self.anchor_sparse_config
        if config is None:
            raise RuntimeError("Packed routing requires an anchor sparse config")
        if not 0 < current_frames <= route.num_video_frames:
            raise ValueError("Current frames do not fit the recorded anchor route")
        history_frames = route.num_video_frames - current_frames
        history_token_count = history_frames * config.frame_seqlen
        if route.scores.shape[1:] != (route.num_video_frames, config.frame_seqlen):
            raise ValueError("Recorded route scores do not match the video layout")

        cache_is_valid = (
            self.anchor_sparse_reuse_denoise
            and cache_key is not None
            and self._anchor_sparse_packed_profile_cache_key == cache_key
            and self._anchor_sparse_packed_current_profile is not None
            and self._anchor_sparse_packed_history_profile is not None
        )
        if cache_is_valid:
            assert self._anchor_sparse_packed_current_profile is not None
            assert self._anchor_sparse_packed_history_profile is not None
            current_profile = self._anchor_sparse_packed_current_profile
            history_profile = self._anchor_sparse_packed_history_profile
            history_indices = self._anchor_sparse_packed_history_indices
        else:
            current_profile = build_nested_current_profile(
                route.scores[:, -current_frames:],
                config,
            )
            # ``recent_dense_frames`` is defined over history+current.  Current
            # frames are represented by the packed current state, so only the
            # remaining recent-frame allowance is mandatory in historical KV.
            history_config = replace(
                config,
                recent_dense_frames=max(
                    0,
                    config.recent_dense_frames - current_frames,
                ),
            )
            history_profile = build_nested_history_profile(
                route.scores[:, :history_frames],
                history_config,
            )
            history_indices = {}

        for keep_ratio in history_keep_ratios:
            if keep_ratio not in history_indices:
                history_indices[keep_ratio] = history_profile.indices_for_ratio(
                    keep_ratio
                ).detach()

        if (
            not cache_is_valid
            and self.anchor_sparse_reuse_denoise
            and cache_key is not None
        ):
            self._anchor_sparse_packed_profile_cache_key = cache_key
            self._anchor_sparse_packed_current_profile = current_profile
            self._anchor_sparse_packed_history_profile = history_profile
            self._anchor_sparse_packed_history_indices = history_indices
        return (
            current_profile,
            history_profile,
            history_indices,
            history_token_count,
        )


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1, action_horizon=1, state_horizon=1, num_action_per_block=30, num_state_per_block=1
    ) -> BlockMask:
        """
        We will divide the token sequence into the following format:
        [first image (conditioning)] [image blocks] [action blocks] [state blocks]
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image block + current action block + current state block
        - Action blocks: can attend to previous image block + current image block + current state block
        - State blocks: conditioning only, cannot attend to anything
        
        Block alignment:
        - num_image_blocks = (num_frames - 1) // num_frame_per_block
        - num_action_blocks = action_horizon // num_action_per_block  
        - num_state_blocks = state_horizon // num_state_per_block
        - num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        """
        # Calculate block structure
        num_image_blocks = (num_frames - 1) // num_frame_per_block
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        
        # Verify the relationship: num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        assert num_image_blocks == num_action_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_action_blocks}"
        assert num_image_blocks == num_state_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_state_blocks}"
        
        # Token ranges
        first_image_len = frame_seqlen  # First image (conditioning)
        image_blocks_len = num_image_blocks * num_frame_per_block * frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        total_length = first_image_len + image_blocks_len + action_len + state_len
        
        # print("total_length", total_length, first_image_len, image_blocks_len, action_len, state_len)
        # Padding to multiple of 128
        # padded_length = math.ceil(total_length / 128) * 128 - total_length
        padded_length = math.ceil((local_attn_size * frame_seqlen + (local_attn_size - 1) + 32 * (local_attn_size - 1))/128) * 128 - total_length
        total_padded_length = total_length + padded_length
        # print("total_padded_length", total_padded_length, total_length, padded_length)
        
        # Define token ranges for each modality
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Precompute block indices for each token
        block_indices = torch.zeros(total_padded_length, device=device, dtype=torch.long)
        
        # First image gets special block index -1 (conditioning, cannot attend to anything)
        block_indices[first_image_start:first_image_end] = -1
        
        # Assign block indices for image blocks (0 to num_image_blocks-1)
        for block_idx in range(num_image_blocks):
            start_idx = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            end_idx = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            block_indices[start_idx:end_idx] = block_idx
        
        # Assign block indices for action tokens (0 to num_action_blocks-1)
        for block_idx in range(num_action_blocks):
            start_idx = action_start + block_idx * num_action_per_block
            end_idx = action_start + (block_idx + 1) * num_action_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # Assign block indices for state tokens (0 to num_state_blocks-1)
        for block_idx in range(num_state_blocks):
            start_idx = state_start + block_idx * num_state_per_block
            end_idx = state_start + (block_idx + 1) * num_state_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # Padding tokens get block index of last block + 1 (won't attend to anything)
        block_indices[total_length:] = num_image_blocks
        
        def attention_mask(b, h, q_idx, kv_idx):
            # Self-attention
            self_attn = (q_idx == kv_idx)
            
            # Determine which modality q and kv belong to
            q_is_first_image = (q_idx >= first_image_start) & (q_idx < first_image_end)
            q_is_image_block = (q_idx >= image_blocks_start) & (q_idx < image_blocks_end)
            q_is_action = (q_idx >= action_start) & (q_idx < action_end)
            q_is_state = (q_idx >= state_start) & (q_idx < state_end)
            
            kv_is_first_image = (kv_idx >= first_image_start) & (kv_idx < first_image_end)
            kv_is_image_block = (kv_idx >= image_blocks_start) & (kv_idx < image_blocks_end)
            kv_is_action = (kv_idx >= action_start) & (kv_idx < action_end)
            kv_is_state = (kv_idx >= state_start) & (kv_idx < state_end)
            
            q_block = block_indices[q_idx]
            kv_block = block_indices[kv_idx]
            
            # First image query (conditioning) - cannot attend to anything
            first_image_mask = q_is_first_image & False
            
            # Image block query
            image_to_first = q_is_image_block & kv_is_first_image  # Image block to first image: always allowed
            image_to_image = q_is_image_block & kv_is_image_block & (kv_block <= q_block)  # Image block to image block: can attend to current and previous image blocks
            image_to_action = q_is_image_block & kv_is_action & (kv_block == q_block)  # Image block to action: can attend to current action block
            image_to_state = q_is_image_block & kv_is_state & (kv_block == q_block)  # Image block to state: can attend to current state block
            
            image_block_mask = image_to_first | image_to_image | image_to_action | image_to_state
            
            # Action query
            action_to_image = q_is_action & kv_is_image_block & (kv_block <= q_block)  # Action to image block: can attend to current and all previous image blocks
            action_to_action = q_is_action & kv_is_action & (kv_block == q_block)  # Action to action: only same block
            action_to_state = q_is_action & kv_is_state & (kv_block == q_block)  # Action to state: only same block
            action_to_first = q_is_action & kv_is_first_image  # Action to first image: always allowed
            
            action_mask = action_to_image | action_to_action | action_to_state | action_to_first
            
            # State query (conditioning) - cannot attend to anything
            state_mask = q_is_state & False
            
            # Combine all masks
            return self_attn | first_image_mask | image_block_mask | action_mask | state_mask
        
        block_mask = create_block_mask(
            attention_mask, B=None, H=None, 
            Q_LEN=total_padded_length,
            KV_LEN=total_padded_length, 
            _compile=False, device=device
        )
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Created blockwise causal attention mask:")
            print(f"  first_image_tokens={first_image_len} (conditioning)")
            print(f"  num_image_blocks={num_image_blocks} (blocks of {num_frame_per_block * frame_seqlen})")
            print(f"  num_action_blocks={num_action_blocks} (blocks of {num_action_per_block})")
            print(f"  num_state_blocks={num_state_blocks} (blocks of {num_state_per_block})")
            print(f"  total_length={total_length}, padded_length={padded_length}")
            print(block_mask)

            # Debug: materialize a small slice of the mask into 0/1 strings
            try:
                dense_mask = create_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_padded_length,
                    KV_LEN=total_padded_length,
                    device=device,
                )[0, 0]  # [Q, K]
                preview_q = min(979, dense_mask.shape[0])
                preview_k = min(979, dense_mask.shape[1])
                print("Block mask (preview):")
                for qi in range(preview_q):
                    row = dense_mask[qi, :preview_k].to(torch.int8).tolist()
                    print(" ".join(str(int(v)) for v in row))
            except Exception as err:
                print("[warn] Failed to materialize block mask preview:", err)
        
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(self.local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
        update_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Forward pass through the diffusion model blocks.
        """
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]
        video_seq_len = x.shape[1]

        anchor_route_indices: torch.Tensor | None = None
        current_video_indices: torch.Tensor | None = None
        current_attention_query_indices: torch.Tensor | None = None
        anchor_route_cache_key: tuple[Any, ...] | None = None
        if self.anchor_sparse_config is not None and action is not None:
            if (
                self._anchor_sparse_last_start_frame is not None
                and current_start_frame < self._anchor_sparse_last_start_frame
            ):
                # A rewind denotes a new causal rollout/episode.  Do not carry
                # semantic anchors from the previous scene into the new one.
                self.clear_anchor_sparse_route_cache()
            self._anchor_sparse_last_start_frame = current_start_frame

            first_cache = kv_cache[0] if kv_cache else None
            if first_cache is not None:
                cached_video_tokens = first_cache[0].shape[1]
                routed_video_tokens = min(
                    cached_video_tokens + video_seq_len,
                    self.blocks[0].self_attn.max_attention_size,
                )
                if routed_video_tokens % self.frame_seqlen != 0:
                    raise ValueError(
                        "Anchor route cache requires frame-aligned video KV length, got "
                        f"{routed_video_tokens} tokens."
                    )
                anchor_route_cache_key = (
                    current_start_frame,
                    cached_video_tokens,
                    video_seq_len,
                    B,
                    x.device.type,
                    x.device.index,
                    x.dtype,
                )
                if (
                    self.anchor_sparse_reuse_denoise
                    and self._anchor_sparse_route_cache_key == anchor_route_cache_key
                ):
                    anchor_route_indices = self._anchor_sparse_route_cache
                    current_video_indices = self._anchor_sparse_current_route_cache
                    current_attention_query_indices = (
                        self._anchor_sparse_current_attention_route_cache
                    )

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # time embeddings: expand to exactly seq_len so e matches x (5B: frame_seqlen=50, 1 frame -> 50 tokens)
        if F <= seq_len:
            repeat = (seq_len + F - 1) // F
            timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(0, F - 1, seq_len, device=timestep.device, dtype=torch.long)
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)
        
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        updated_kv_caches: list[torch.Tensor] = []
        sparse_end = len(self.blocks) - self.anchor_sparse_dense_suffix_layers
        middle_layer_indices = tuple(
            range(self.anchor_sparse_dense_prefix_layers, sparse_end)
        )
        packed_context_is_eligible = (
            self.anchor_sparse_packed_middle
            and self.anchor_sparse_config is not None
            and action_conditioned_causal_routing_is_eligible(
                current_start_frame,
                action_register_length,
            )
            and not update_kv_cache
            and bool(middle_layer_indices)
        )
        packed_layer_ratios: dict[int, tuple[float, float]] = {}
        packed_layer_head_groups: dict[
            int,
            tuple[tuple[tuple[int, ...], float, float | None], ...] | None,
        ] = {}
        if packed_context_is_eligible:
            packed_layer_ratios = {
                layer_index: self._packed_budget_ratios_for_layer(layer_index)
                for layer_index in middle_layer_indices
            }
            packed_layer_head_groups = {
                layer_index: self._packed_head_groups_for_layer(layer_index)
                for layer_index in middle_layer_indices
            }
            for layer_index, groups in packed_layer_head_groups.items():
                if groups is None:
                    continue
                grouped_current_ratios = tuple(
                    current_ratio
                    for _, _, current_ratio in groups
                    if current_ratio is not None
                )
                if grouped_current_ratios:
                    history_ratio, current_ratio = packed_layer_ratios[layer_index]
                    packed_layer_ratios[layer_index] = (
                        history_ratio,
                        max(current_ratio, *grouped_current_ratios),
                    )
            if self.anchor_sparse_propagate_radius > 0:
                stable_ratios = stabilize_current_budgets_for_segments(
                    tuple(
                        packed_layer_ratios[layer_index]
                        for layer_index in middle_layer_indices
                    ),
                    segment_length=self.anchor_sparse_propagate_every,
                )
                packed_layer_ratios = dict(
                    zip(middle_layer_indices, stable_ratios, strict=True)
                )
            if self._dynamic_m1_shared_budget_promotion:
                if self._dynamic_sparse_dit_index is None:
                    raise RuntimeError(
                        "Dynamic M1 trace is active before the real DiT index was set"
                    )
                for layer_index, (history_ratio, current_ratio) in (
                    packed_layer_ratios.items()
                ):
                    trace_cell = self._dynamic_m1_shared_budget_trace.get(
                        (self._dynamic_sparse_dit_index, layer_index)
                    )
                    if trace_cell is not None:
                        trace_cell["effective_history_keep_ratio"] = history_ratio
                        trace_cell["effective_current_keep_ratio"] = current_ratio
        packed_has_sparse_layer = any(
            history_ratio < 1.0 or current_ratio < 1.0
            for history_ratio, current_ratio in packed_layer_ratios.values()
        ) or any(
            groups is not None
            and any(
                history_ratio < 1.0
                or (current_ratio is not None and current_ratio < 1.0)
                for _, history_ratio, current_ratio in groups
            )
            for groups in packed_layer_head_groups.values()
        )
        packed_middle_active = (
            packed_context_is_eligible
            and packed_has_sparse_layer
        )
        packed_state: PackedMiddleState | None = None
        packed_current_profile: NestedAnchorProfile | None = None
        packed_freqs: torch.Tensor | None = None
        packed_history_indices: dict[float, torch.Tensor] = {}
        packed_history_token_count = 0
        maximum_current_keep_ratio = (
            max(current for _, current in packed_layer_ratios.values())
            if packed_layer_ratios
            else self.anchor_sparse_current_keep_ratio
        )
        history_keep_ratios = tuple(
            sorted(
                {history for history, _ in packed_layer_ratios.values()}
                | {
                    history_ratio
                    for groups in packed_layer_head_groups.values()
                    if groups is not None
                    for _, history_ratio, _ in groups
                }
            )
        )
        self._anchor_sparse_last_packed_propagation_count = 0

        for block_index, block in enumerate(self.blocks):
            in_packed_middle = (
                packed_middle_active
                and self.anchor_sparse_dense_prefix_layers
                <= block_index
                < sparse_end
            )
            if in_packed_middle:
                if action_register_length is None:
                    raise RuntimeError("Packed middle execution requires action/state registers")
                if packed_state is None:
                    route = self.blocks[0].self_attn.last_anchor_route
                    if route is None:
                        raise RuntimeError(
                            "The Dense prefix did not expose action-conditioned route scores"
                        )
                    if video_seq_len % self.frame_seqlen != 0:
                        raise ValueError(
                            "Packed current routing requires complete video frames, got "
                            f"{video_seq_len} tokens for frame_seqlen={self.frame_seqlen}"
                        )
                    current_frames = video_seq_len // self.frame_seqlen
                    (
                        packed_current_profile,
                        _history_profile,
                        packed_history_indices,
                        packed_history_token_count,
                    ) = self._prepare_packed_anchor_profiles(
                        route=route,
                        current_frames=current_frames,
                        cache_key=anchor_route_cache_key,
                        history_keep_ratios=history_keep_ratios,
                    )
                    packed_state = pack_middle_state(
                        x,
                        e0,
                        packed_current_profile,
                        maximum_keep_ratio=maximum_current_keep_ratio,
                        action_register_length=action_register_length,
                    )
                    num_state_tokens = action_register_length - action_length
                    if (
                        action_length != self.num_action_per_block
                        or num_state_tokens != self.num_state_per_block
                    ):
                        raise ValueError(
                            "Packed RoPE requires one complete action/state register block"
                        )
                    if packed_freqs is None:
                        packed_freqs = gather_packed_rope_frequencies(
                            freqs,
                            self.freqs_action,
                            self.freqs_state,
                            packed_state.original_indices,
                            video_seq_len=video_seq_len,
                            num_action_tokens=action_length,
                            num_state_tokens=num_state_tokens,
                            action_state_index=(
                                (current_start_frame - 1)
                                // self.num_frame_per_block
                            ),
                        )
                assert packed_state is not None
                assert packed_current_profile is not None
                assert packed_freqs is not None
                history_keep_ratio, current_keep_ratio = packed_layer_ratios[
                    block_index
                ]
                layer_head_groups = packed_layer_head_groups.get(block_index)
                layer_current_video_tokens_by_ratio = (
                    {
                        ratio: packed_current_profile.video_tokens_for_ratio(ratio)
                        for _, _, ratio in layer_head_groups
                        if ratio is not None
                    }
                    if layer_head_groups is not None
                    else None
                )
                packed_current_video_tokens = (
                    packed_current_profile.video_tokens_for_ratio(current_keep_ratio)
                )
                active_length = packed_state.active_length(packed_current_video_tokens)
                block.self_attn.packed_dense_action_history = (
                    self._packed_dense_action_history_for_layer(block_index)
                )
                block.self_attn.packed_max_action_current = (
                    self._packed_max_action_current_for_layer(block_index)
                )
                max_action_current_active = (
                    block.self_attn.packed_max_action_current
                    and packed_state.maximum_video_tokens
                    > packed_current_video_tokens
                )
                maximum_action_current_x = (
                    packed_state.packed_x
                    if max_action_current_active
                    else None
                )
                maximum_action_current_e0 = (
                    packed_state.packed_e0
                    if max_action_current_active
                    else None
                )
                maximum_action_current_freqs = (
                    packed_freqs
                    if max_action_current_active
                    else None
                )
                try:
                    updated_packed = block.forward_packed(
                        packed_state.active_x(packed_current_video_tokens),
                        packed_state.active_e0(packed_current_video_tokens),
                        packed_freqs[:, :active_length],
                        action_register_length=action_register_length,
                        context=context,
                        kv_cache=kv_cache[block_index],
                        history_indices=packed_history_indices[history_keep_ratio],
                        history_token_count=packed_history_token_count,
                        head_groups=layer_head_groups,
                        history_indices_by_ratio=(
                            packed_history_indices
                            if layer_head_groups is not None
                            else None
                        ),
                        current_video_tokens_by_ratio=(
                            layer_current_video_tokens_by_ratio
                            if layer_head_groups is not None
                            else None
                        ),
                        maximum_current_x=maximum_action_current_x,
                        maximum_current_e0=maximum_action_current_e0,
                        maximum_current_freqs=maximum_action_current_freqs,
                    )
                finally:
                    # Fresh per-DiT routing produces fresh index tensors.  A
                    # cache keyed by those pointers cannot hit on the next
                    # denoise call and otherwise retains every gathered K/V
                    # generation for the whole trajectory.
                    if not self.anchor_sparse_reuse_denoise:
                        block.self_attn.clear_anchor_sparse_history_cache()
                packed_state.update_active(
                    updated_packed,
                    packed_current_video_tokens,
                )
                packed_offset = (
                    block_index - self.anchor_sparse_dense_prefix_layers + 1
                )
                should_propagate_packed = (
                    self.anchor_sparse_propagate_radius > 0
                    and (
                        packed_offset % self.anchor_sparse_propagate_every == 0
                        or block_index == sparse_end - 1
                    )
                )
                if should_propagate_packed:
                    if self.anchor_sparse_config is None:
                        raise RuntimeError(
                            "Packed propagation requires an anchor sparse config"
                        )
                    x = packed_state.recover_propagated(
                        config=self.anchor_sparse_config,
                        radius=self.anchor_sparse_propagate_radius,
                    )
                    packed_state = None
                    self._anchor_sparse_last_packed_propagation_count += 1
                continue

            if packed_state is not None:
                # The only materialized full-sequence restoration is the Dense
                # suffix/output boundary (or the end of an all-packed stack).
                x = packed_state.recover_full()
                packed_state = None

            x, updated_kv_cache, anchor_route_indices = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
                anchor_route_indices=anchor_route_indices,
                current_video_indices=current_video_indices,
                current_attention_query_indices=current_attention_query_indices,
                update_kv_cache=update_kv_cache,
            )
            if update_kv_cache:
                if updated_kv_cache is None:
                    raise RuntimeError("KV cache update was requested but a block returned None")
                updated_kv_caches.append(updated_kv_cache)
            if (
                block_index == 0
                # The initial (and cache-rewind) pass uses the blockwise
                # conditioning path above, which intentionally does not expose
                # action-conditioned key scores.  Keep that pass fully dense;
                # current-token routing starts only once causal KV history exists.
                and action_conditioned_causal_routing_is_eligible(
                    current_start_frame,
                    action_register_length,
                )
                # Video-only cache-fill calls run before action denoising and do
                # not carry action/state registers. Their KV update stays dense;
                # the following WAM call builds the action-conditioned route.
                and self.anchor_sparse_config is not None
                and not self.anchor_sparse_packed_middle
                and (
                    (
                        self.anchor_sparse_current_keep_ratio < 1.0
                        and current_video_indices is None
                    )
                    or (
                        self.anchor_sparse_current_attention
                        and self.anchor_sparse_attention_query_keep_ratio < 1.0
                        and current_attention_query_indices is None
                    )
                )
            ):
                route = block.self_attn.last_anchor_route
                if route is None:
                    raise RuntimeError(
                        "The dense prefix layer did not produce scores for current-token routing"
                    )
                if video_seq_len % self.frame_seqlen != 0:
                    raise ValueError(
                        "Current-token routing requires complete video frames, got "
                        f"{video_seq_len} tokens for frame_seqlen={self.frame_seqlen}"
                    )
                current_frames = video_seq_len // self.frame_seqlen
                current_scores = route.scores[:, -current_frames:]
                if (
                    self.anchor_sparse_current_keep_ratio < 1.0
                    and current_video_indices is None
                ):
                    current_video_indices = build_current_video_query_route(
                        current_scores,
                        self.anchor_sparse_config,
                        keep_ratio=self.anchor_sparse_current_keep_ratio,
                    ).detach()
                if (
                    self.anchor_sparse_current_attention
                    and self.anchor_sparse_attention_query_keep_ratio < 1.0
                    and current_attention_query_indices is None
                ):
                    if (
                        current_video_indices is not None
                        and self.anchor_sparse_attention_query_keep_ratio
                        == self.anchor_sparse_current_keep_ratio
                    ):
                        current_attention_query_indices = current_video_indices
                    else:
                        current_attention_query_indices = build_current_video_query_route(
                            current_scores,
                            self.anchor_sparse_config,
                            keep_ratio=self.anchor_sparse_attention_query_keep_ratio,
                        ).detach()

        if packed_state is not None:
            x = packed_state.recover_full()

        if (
            self.anchor_sparse_reuse_denoise
            and anchor_route_cache_key is not None
            and anchor_route_indices is not None
        ):
            self._anchor_sparse_route_cache = anchor_route_indices.detach()
            self._anchor_sparse_current_route_cache = (
                current_video_indices.detach()
                if current_video_indices is not None
                else None
            )
            self._anchor_sparse_current_attention_route_cache = (
                current_attention_query_indices.detach()
                if current_attention_query_indices is not None
                else None
            )
            self._anchor_sparse_route_cache_key = anchor_route_cache_key

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches


    def _forward_inference_trt(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen 
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
            update_kv_cache=False,
        ) 

        return x_video, action_noise_pred

    def _forward_inference_trt_droid(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen 
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
            update_kv_cache=False,
        ) 

        return x_video, action_noise_pred


    def _forward_inference(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
        embodiment_id=None,
        update_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            timestep (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            action (Tensor, *optional*):
                Action tensor of shape [B, H, D]
            state (Tensor, *optional*):
                State tensor of shape [B, H, D]
            embodiment_id (Tensor, *optional*):
                Embodiment ID tensor of shape [B]
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            clip_feature (Tensor, *optional*):
                CLIP image features for image-to-video mode
            timestep_action (Tensor, *optional*):
                Action timestep tensor of shape [B]
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """      
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        # Concat [x; y] only when pretrained that way (14B). 5B uses latent only, first-frame via CLIP.
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
            update_kv_cache=update_kv_cache,
        )

        # Copy the updated KV caches back to the original KV cache.
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        #for block_index, updated_kv_cache in enumerate(updated_kv_caches):
        #    kv_cache[block_index] = updated_kv_cache.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None

        # Concat [x; y] only when pretrained that way (14B). 5B uses latent only, first-frame via CLIP.
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)

        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=0,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        F = timestep.shape[1]

        # time embeddings
        if action is not None:
            embodiment_id = torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            action_length = action_features.shape[1]
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            action_length = None
            state_features = None
            action_register = None
            action_register_length = None

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        timestep_original = timestep.clone()

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        if clean_x is not None:
            if y is not None and self.concat_first_frame_latent:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len

            x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)
            assert aug_t is not None

            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean)
            e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
            is_tf=clean_x is not None,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache, anchor_route_indices = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                assert anchor_route_indices is None
                return outputs
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x, updated_kv_cache, anchor_route_indices = block(x, **kwargs)
                assert updated_kv_cache is None
                assert anchor_route_indices is None

        if clean_x is not None:
            x = x[:, clean_x.shape[1]:]

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_size):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patchified features, with shape [B, L, C_out * prod(patch_size)].
            grid_size (Tensor):
                Spatial-temporal grid dimensions before patching, with shape [3]
                (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1
        ).reshape(f * h * w, 1, -1)

        return freqs

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
