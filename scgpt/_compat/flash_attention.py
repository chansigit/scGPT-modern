"""
Compatibility shim: old flash-attn v1.x ``FlashMHA`` API, reimplemented on top
of modern flash-attn.

scGPT's model code (``scgpt/model/model.py`` etc.) imports::

    from flash_attn.flash_attention import FlashMHA

This module no longer exists in flash-attn v2+ — the class was removed and the
public API became functional (``flash_attn_func``, ``flash_attn_varlen_func``).
This shim lets scGPT keep running on modern flash-attn (v2.x on Ampere/Ada,
v4 / CuTeDSL on Hopper/Blackwell) **without touching the model code beyond
the one-line import**.

Goals:
  - Preserve the exact ``__init__`` signature and keyword arguments.
  - Preserve the submodule names ``Wqkv`` and ``out_proj`` so existing scGPT
    checkpoints (``best_model.pt``) load with no key remapping.
  - Preserve the ``forward(x, key_padding_mask, need_weights)`` signature and
    return type ``(output, None)``.
  - Accept the same key_padding_mask convention that scGPT's
    ``FlashTransformerEncoderLayer`` produces: **True = valid token, False =
    padding** (scGPT flips the PyTorch-standard mask before calling here).

Backend selection (checked once, at first forward call):
  1. flash-attn-4 (``flash_attn.cute``) — CuTeDSL kernels for Hopper/Blackwell
  2. flash-attn 3 (``flash_attn_3``) — hopper-optimized FA3, built from source
     against the Dao-AILab repo's ``hopper/`` subdirectory. Strictly faster than
     FA2 on H100/sm_90a thanks to WGMMA + TMA + warp specialization.
  3. flash-attn v2+ (``flash_attn.flash_attn_func`` / ``flash_attn_varlen_func``)
  4. ImportError — fall through, the model-level ``flash_attn_available`` flag
     in ``scgpt/model/model.py`` will have already tripped and the pure-PyTorch
     transformer path will be used instead.

Version adapters:
  FA3 dropped the ``dropout_p`` kwarg (FA3 has no native attention dropout).
  When the v3 backend is selected, the resolver wraps ``flash_attn_func`` and
  ``flash_attn_varlen_func`` in thin closures that silently drop the kwarg so
  the outer ``FlashMHA.forward()`` can keep passing it uniformly. scGPT's
  default configs use ``attention_dropout=0.0`` so this is a no-op at runtime;
  a warning is emitted once if a nonzero dropout is ever requested on v3.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
# Backend detection (lazy — only runs at first forward pass).
# --------------------------------------------------------------------------- #

_flash_attn_func: Optional[Callable[..., Any]] = None
_flash_attn_varlen_func: Optional[Callable[..., Any]] = None
_FA_BACKEND: Optional[str] = None


def _wrap_v3(fn_func: Callable[..., Any], fn_varlen: Callable[..., Any]) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Adapt flash-attn 3 functions to the v2 calling convention.

    FA3 removed the ``dropout_p`` kwarg. The returned wrappers accept it for
    API parity but warn (once) if a nonzero value is supplied, since FA3 has
    no native attention dropout path.
    """
    import warnings as _warnings

    _warned = {"on": False}

    def _maybe_warn(dropout_p: float) -> None:
        if dropout_p and not _warned["on"]:
            _warnings.warn(
                "scgpt._compat.flash_attention: flash-attn 3 has no native "
                "attention dropout; dropout_p=%r will be silently ignored. "
                "Use FA2 (`pip install flash-attn`) if attention dropout is "
                "required during training." % (dropout_p,),
                RuntimeWarning,
                stacklevel=3,
            )
            _warned["on"] = True

    def _fa3_dense(q, k, v, dropout_p=0.0, causal=False, **kwargs):
        _maybe_warn(dropout_p)
        return fn_func(q, k, v, causal=causal, **kwargs)

    def _fa3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                   dropout_p=0.0, causal=False, **kwargs):
        _maybe_warn(dropout_p)
        return fn_varlen(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            causal=causal, **kwargs,
        )

    return _fa3_dense, _fa3_varlen


def _resolve_backend() -> Tuple[Callable[..., Any], Callable[..., Any], str]:
    """Pick the best available flash-attn implementation.

    Order of preference: flash-attn-4 CuTeDSL → flash-attn 3 (hopper) →
    flash-attn v2+ → raise. Result is cached in module globals after the
    first successful call.
    """
    global _flash_attn_func, _flash_attn_varlen_func, _FA_BACKEND

    cached_fn = _flash_attn_func
    cached_vfn = _flash_attn_varlen_func
    cached_backend = _FA_BACKEND
    if cached_fn is not None and cached_vfn is not None and cached_backend is not None:
        return cached_fn, cached_vfn, cached_backend

    fn: Optional[Callable[..., Any]] = None
    vfn: Optional[Callable[..., Any]] = None
    backend: Optional[str] = None

    # Preferred: flash-attn-4 (CuTeDSL, Hopper/Blackwell optimized).
    try:
        from flash_attn.cute import flash_attn_func as _f  # type: ignore[import-not-found]
        from flash_attn.cute import flash_attn_varlen_func as _fv  # type: ignore[import-not-found]
        fn, vfn, backend = _f, _fv, "v4-cute"
    except ImportError:
        pass

    # Next preference: flash-attn 3 (hopper) — source-built from Dao-AILab
    # flash-attention repo's hopper/ subdirectory. H100 sm_90a optimized.
    if fn is None:
        try:
            from flash_attn_3 import flash_attn_func as _f3  # type: ignore[import-not-found]
            from flash_attn_3 import flash_attn_varlen_func as _fv3  # type: ignore[import-not-found]
            fn, vfn = _wrap_v3(_f3, _fv3)
            backend = "v3-hopper"
        except ImportError:
            pass

    # Fallback: flash-attn v2+ mainline (Ampere / Ada / Hopper).
    if fn is None:
        try:
            from flash_attn import flash_attn_func as _f2  # type: ignore[import-not-found]
            from flash_attn import flash_attn_varlen_func as _fv2  # type: ignore[import-not-found]
            fn, vfn, backend = _f2, _fv2, "v2"
        except ImportError:
            pass

    if fn is None or vfn is None or backend is None:
        raise ImportError(
            "scgpt._compat.flash_attention: could not import flash-attn. "
            "None of flash-attn-4 (flash_attn.cute), flash-attn 3 "
            "(flash_attn_3), or flash-attn v2+ (flash_attn.flash_attn_func) "
            "is available. Install one of: "
            "`pip install flash-attn-4` (once NVIDIA fixes packaging), "
            "`cd flash-attention/hopper && python setup.py install` for v3, "
            "or `pip install flash-attn` for v2."
        )

    _flash_attn_func, _flash_attn_varlen_func, _FA_BACKEND = fn, vfn, backend
    return fn, vfn, backend


def get_backend_name() -> str:
    """Return the selected backend tag: ``v4-cute`` / ``v3-hopper`` / ``v2`` / ``none``.

    Triggers backend resolution as a side effect.
    """
    try:
        _resolve_backend()
    except ImportError:
        return "none"
    return _FA_BACKEND or "none"


# --------------------------------------------------------------------------- #
# Padding helpers (variable-sequence-length path).
#
# Written to not depend on ``flash_attn.bert_padding``, which existed in
# flash-attn v1/v2 but is not guaranteed in flash-attn-4. ~15 lines of numpy-
# style indexing is all we need.
# --------------------------------------------------------------------------- #

def _unpad(
    x: torch.Tensor, key_padding_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Remove padding tokens from a (B, S, ...) tensor.

    Args:
        x: tensor of shape (B, S, ...)
        key_padding_mask: bool tensor (B, S); **True = valid, False = padding**

    Returns:
        x_unpad:     (nnz, ...)
        cu_seqlens:  (B+1,) int32 — cumulative sequence lengths
        max_seqlen:  int — longest valid seqlen in the batch
        indices:     (nnz,) int64 — positions in flat (B*S) space (for repad)
    """
    seqlens = key_padding_mask.sum(dim=-1, dtype=torch.int32)          # (B,)
    max_seqlen = int(seqlens.max().item())
    cu_seqlens = F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0))  # (B+1,)

    indices = key_padding_mask.flatten().nonzero(as_tuple=False).flatten()
    x_flat = x.flatten(0, 1)                                          # (B*S, ...)
    x_unpad = x_flat.index_select(0, indices)                         # (nnz, ...)
    return x_unpad, cu_seqlens, max_seqlen, indices


def _pad(
    x_unpad: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seqlen: int,
) -> torch.Tensor:
    """Inverse of :func:`_unpad` — scatter valid tokens back into (B, S, ...)."""
    trailing = x_unpad.shape[1:]
    out = x_unpad.new_zeros((batch_size * seqlen, *trailing))
    out.index_copy_(0, indices, x_unpad)
    return out.view(batch_size, seqlen, *trailing)


# --------------------------------------------------------------------------- #
# The shim class — same name, same API, same parameter names as the legacy
# flash-attn v1 FlashMHA.
# --------------------------------------------------------------------------- #

class FlashMHA(nn.Module):
    """Drop-in replacement for ``flash_attn.flash_attention.FlashMHA``.

    Parameters are identical to the legacy class. The learnable submodules
    ``Wqkv`` (in-proj) and ``out_proj`` are named exactly as in the original
    so that pretrained scGPT checkpoints load without any key remapping.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        assert batch_first, "FlashMHA compat shim requires batch_first=True"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads
        assert head_dim % 8 == 0 and head_dim <= 256, (
            "flash-attn supports head_dim divisible by 8, up to 256 "
            f"(got head_dim={head_dim})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        self.batch_first = batch_first
        self.dropout_p = attention_dropout

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, S, embed_dim) — fp16 or bf16, CUDA.
            key_padding_mask: optional (B, S) bool. **True = valid, False = padding.**
                (scGPT's FlashTransformerEncoderLayer already flips the PyTorch
                convention before calling here.)
            need_weights: not supported by flash-attn; asserted False.

        Returns:
            (output, None) where output has shape (B, S, embed_dim). The
            second element is always ``None`` — flash-attn does not materialize
            attention probabilities.
        """
        assert not need_weights, "FlashMHA shim does not return attention weights"
        assert x.is_cuda, "flash-attn requires CUDA tensors"

        flash_attn_func, flash_attn_varlen_func, _ = _resolve_backend()

        B, S, _ = x.shape
        # (B, S, 3*D) -> (B, S, 3, H, Dh) -> (q, k, v) each (B, S, H, Dh).
        # Wqkv runs at the module's dtype (typically fp32 under scgpt.tasks.*
        # which leaves the model unpromoted). We cast the q/k/v activations to
        # fp16 for the kernel and cast back before out_proj — mirrors flash-
        # attn v1's internal behavior so the shim stays drop-in.
        qkv = self.Wqkv(x)
        orig_dtype = qkv.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(torch.float16)

        qkv = rearrange(qkv, "b s (three h d) -> b s three h d",
                        three=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        dropout_p = self.dropout_p if self.training else 0.0

        if key_padding_mask is None:
            # Fast path: no padding, plain flash attention.
            out = flash_attn_func(
                q, k, v, dropout_p=dropout_p, causal=self.causal,
            )                                                 # (B, S, H, Dh)
        else:
            # Variable-length path: unpad, run varlen kernel, scatter back.
            q_u, cu_seqlens, max_s, indices = _unpad(q, key_padding_mask)
            k_u, _,          _,     _       = _unpad(k, key_padding_mask)
            v_u, _,          _,     _       = _unpad(v, key_padding_mask)
            out_u = flash_attn_varlen_func(
                q_u, k_u, v_u,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_s,
                max_seqlen_k=max_s,
                dropout_p=dropout_p,
                causal=self.causal,
            )
            out = _pad(out_u, indices, B, S)                  # (B, S, H, Dh)

        out = rearrange(out, "b s h d -> b s (h d)")
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return self.out_proj(out), None


__all__ = ["FlashMHA", "get_backend_name"]
