# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import contextlib
import torch
from torch.profiler import profile, ProfilerActivity
import time
from typing import Optional
import os

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import transformer_engine
    from groot.vla.model.dreamzero.modules.cudnn_attention import DotProductAttention
    TRANSFORMER_ENGINE_AVAILABLE = True
except ModuleNotFoundError:
    TRANSFORMER_ENGINE_AVAILABLE = False

import warnings


def _gpu_supports_flash_attention():
    """FlashAttention requires Ampere (compute capability 8.0) or newer."""
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8
    except Exception:
        return False


def _sdpa_attention_fallback(
    q, k, v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """PyTorch SDPA fallback for GPUs that don't support FlashAttention (e.g. pre-Ampere)."""
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention on this GPU. '
            'It can have a slight impact on quality.'
        )
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * softmax_scale
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[tuple[int, int]] = None,
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: Optional[int] = None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    version:        int. 2 for flash attention 2, 3 for flash attention 3.

    Returns:
        x:              [B, Lq, Nq, C2].
    """
    if window_size is None:
        window_size = (-1, -1)
    if version is None:
        version = 3

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # Use PyTorch SDPA on pre-Ampere GPUs (FlashAttention requires Ampere or newer)
    if not _gpu_supports_flash_attention():
        return _sdpa_attention_fallback(
            q, k, v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )
    zeros = torch.zeros([1], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0).to(torch.int32)
    cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0).to(torch.int32)

    # apply attention
    if version == 3 and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        raise ValueError(f"Invalid version: {version}")

    # output
    return x.type(out_dtype)


class AttentionModule(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.,
        softmax_scale: Optional[float] = None,
        q_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[tuple[int, int]] = None,
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backend: Optional[str] = None,
    ):
        super().__init__()
        if backend is None:
            backend = "torch"

        if os.getenv("ATTENTION_BACKEND") is not None:
            backend = os.getenv("ATTENTION_BACKEND")
        else:
            backend = "FA2"

        # Check for TensorRT at runtime, not import time
        if os.getenv("ENABLE_TENSORRT", "False").lower() == "true":
            backend = "torch"

        # Fall back to FA backend if TE is specified but not available
        if backend == "TE" and not TRANSFORMER_ENGINE_AVAILABLE:
            print("Warning: Transformer Engine is not available. Falling back to FA2 backend.")
            backend = "FA2"

        assert backend in ["torch", "FA2", "FA3", "TE", "torch_onnx"]
        self.backend = backend
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.q_scale = q_scale
        self.causal = causal
        self.window_size = window_size if window_size is not None else (-1, -1)
        self.deterministic = deterministic
        self.compute_dtype = dtype

        if backend == "torch":
            def _torch_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                q = q.transpose(1, 2).to(dtype)
                k = k.transpose(1, 2).to(dtype)
                v = v.transpose(1, 2).to(dtype)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

                out = out.transpose(1, 2).contiguous()
                return out.to(out_dtype)
            self.attn_func = _torch_impl

        elif  backend == "torch_onnx":
            def _torch_onnx_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                # use torch.nn.functional.scaled_dot_product_attention for tensorrt export

                # The input is (s, n, d), but sdpa needs (b, h, s, d).
                # We add a batch dimension and transpose.
                q = q.unsqueeze(0).transpose(1, 2).to(dtype)
                k = k.unsqueeze(0).transpose(1, 2).to(dtype)
                v = v.unsqueeze(0).transpose(1, 2).to(dtype)

                # Fix for ONNX export: repeat k and v to match q's batch size in cross-attention
                if q.shape[0] != k.shape[0] and k.shape[0] == 1:
                    k = k.repeat(q.shape[0], 1, 1, 1)
                    v = v.repeat(q.shape[0], 1, 1, 1)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

                # Transpose back to (b, s, n, d) format.
                out = out.transpose(1, 2).contiguous()
                return out.to(out_dtype)
            self.attn_func = _torch_onnx_impl

        elif backend == "TE" and TRANSFORMER_ENGINE_AVAILABLE:
            self.attn_backend = DotProductAttention(
                num_attention_heads=num_heads,
                kv_channels=head_dim,
                qkv_format="bshd",
                attn_mask_type="causal" if causal else "no_mask",
                window_size=window_size,
                attention_dropout=dropout_p,
            )

            def _te_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                return self.attn_backend(
                    query_layer=q.to(dtype),
                    key_layer=k.to(dtype),
                    value_layer=v.to(dtype),
                ).to(out_dtype)
            self.attn_func = _te_impl

        elif backend == "FA2" or backend == "FA3":
            def _flash_attn_impl(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                q_lens: Optional[torch.Tensor], k_lens: Optional[torch.Tensor],
            ) -> torch.Tensor:
                return flash_attention(
                    q=q, k=k, v=v,
                    q_lens=q_lens, k_lens=k_lens,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    q_scale=q_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                    dtype=dtype,
                    version=3 if backend == "FA3" else 2,
                )
            self.attn_func = _flash_attn_impl

        else:
            raise ValueError(f"Invalid backend: {backend}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
    ):
        if (
            self.backend == "torch" or
            self.backend == "torch_onnx" or
            (self.backend == "TE" and TRANSFORMER_ENGINE_AVAILABLE)
        ):
            if q_lens is not None or k_lens is not None:
                warnings.warn(
                    'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
                )
            return self.attn_func(q, k, v)  # type: ignore[call-arg]
        else:
            return self.attn_func(q, k, v, q_lens, k_lens)  # type: ignore[call-arg]

    def forward_varlen_head_sequences(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Run one varlen kernel with each original head as a batch sequence.

        Inputs are already packed as ``[total_tokens, 1, head_dim]``. This
        lets heterogeneous M1 head groups share one FA2 launch without padding
        different history/current budgets to a common shape.
        """

        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("varlen head sequences must be [T, 1, D]")
        if q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
            raise ValueError("varlen head sequences require one kernel head")
        if k.shape != v.shape:
            raise ValueError("varlen key and value shapes must match")
        if q_lens.ndim != 1 or k_lens.ndim != 1:
            raise ValueError("varlen sequence lengths must be vectors")
        if q_lens.shape != k_lens.shape:
            raise ValueError("varlen query/key batches must align")

        use_flash = (
            q.device.type == "cuda"
            and _gpu_supports_flash_attention()
            and self.backend in ("FA2", "FA3")
        )
        if use_flash:
            half_dtypes = (torch.float16, torch.bfloat16)
            q = q if q.dtype in half_dtypes else q.to(self.compute_dtype)
            k = k if k.dtype in half_dtypes else k.to(self.compute_dtype)
            v = v if v.dtype in half_dtypes else v.to(self.compute_dtype)
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            if self.q_scale is not None:
                q = q * self.q_scale
            q_lens = q_lens.to(device=q.device, dtype=torch.int32)
            k_lens = k_lens.to(device=q.device, dtype=torch.int32)
            zeros = torch.zeros(1, device=q.device, dtype=torch.int32)
            cu_seqlens_q = torch.cat((zeros, q_lens)).cumsum(0).to(torch.int32)
            cu_seqlens_k = torch.cat((zeros, k_lens)).cumsum(0).to(torch.int32)
            if self.backend == "FA3" and FLASH_ATTN_3_AVAILABLE:
                return flash_attn_interface.flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                    deterministic=self.deterministic,
                )[0]
            if not FLASH_ATTN_2_AVAILABLE:
                raise RuntimeError("FA2 varlen head sequences require flash-attn")
            return flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout_p,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )

        # CPU and non-FA backends are correctness fallbacks used by unit tests.
        outputs = []
        q_offset = 0
        k_offset = 0
        for query_length, key_length in zip(
            q_lens.tolist(),
            k_lens.tolist(),
            strict=True,
        ):
            query = q[q_offset:q_offset + query_length].transpose(0, 1)
            key = k[k_offset:k_offset + key_length].transpose(0, 1)
            value = v[k_offset:k_offset + key_length].transpose(0, 1)
            output = torch.nn.functional.scaled_dot_product_attention(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                dropout_p=self.dropout_p,
                is_causal=self.causal,
                scale=self.softmax_scale,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
            q_offset += query_length
            k_offset += key_length
        return torch.cat(outputs, dim=0)
