import math
from typing import Optional

import torch
import triton
import triton.language as tl


"""
Fused scaled dot-product attention in Triton.

This implementation focuses on a single batch and head dimension at a time and
operates on (L, D) shaped Q, K, V tensors, where:

- L: sequence length
- D: head dimension

The kernel:
1. Computes QK^T in blocks
2. Applies scaling and (optional) causal masking
3. Computes softmax along the key dimension
4. Multiplies the attention weights by V to produce the output

This is not a fully production-optimized FlashAttention implementation, but it
captures the main structure and provides a realistic starting point for
cross-backend experimentation on CUDA and ROCm.
"""


@triton.jit
def _fused_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_q_l, stride_q_d,
    stride_k_l, stride_k_d,
    stride_v_l, stride_v_d,
    stride_o_l, stride_o_d,
    L, D,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Each program instance computes attention for a block of queries (BLOCK_M)
    over a block of keys (BLOCK_N), with BLOCK_D being the head dimension tile.
    """

    # Program IDs
    pid_m = tl.program_id(0)  # along sequence dimension (queries)
    # We tile only along M for simplicity; N is looped over.

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                    # [BLOCK_D]

    # Mask to stay within sequence length
    mask_m = offs_m < L

    # Pointers to Q block: shape [BLOCK_M, BLOCK_D]
    q_ptrs = Q_ptr + offs_m[:, None] * stride_q_l + offs_d[None, :] * stride_q_d
    q = tl.where(mask_m[:, None] & (offs_d[None, :] < D), tl.load(q_ptrs, mask=mask_m[:, None]), 0.0)

    # We will compute softmax over all N via a streaming approach
    # Initialize max and sum for softmax (per query position)
    qk_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    qk_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Scaling factor for QK^T
    scale = 1.0 / math.sqrt(float(D))

    # Loop over keys in BLOCK_N tiles
    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = offs_n < L

        # Load K tile: [BLOCK_N, BLOCK_D]
        k_ptrs = K_ptr + offs_n[:, None] * stride_k_l + offs_d[None, :] * stride_k_d
        k = tl.where(mask_n[:, None] & (offs_d[None, :] < D), tl.load(k_ptrs, mask=mask_n[:, None]), 0.0)

        # Compute qk: [BLOCK_M, BLOCK_N]
        # q: [M, D], k: [N, D] => qk = q @ k^T
        qk = tl.dot(q, tl.trans(k)) * scale  # [M, N]

        # Optional causal masking: disallow looking ahead
        if causal:
            # For each query position i in offs_m, keys beyond i are masked.
            query_pos = offs_m[:, None]
            key_pos = offs_n[None, :]
            causal_mask = key_pos > query_pos
            qk = tl.where(causal_mask, -float("inf"), qk)

        # Numerically stable softmax accumulation
        current_max = tl.maximum(qk_max[:, None], tl.max(qk, axis=1)[:, None])[:, 0]
        # Rescale old sum with new max
        exp_old = tl.exp(qk_max - current_max)
        exp_new = tl.exp(qk - current_max[:, None])
        qk_sum = exp_old * qk_sum + tl.sum(exp_new, axis=1)
        qk_max = current_max

    # Second pass: compute normalized attention * V
    # Reinitialize output accumulator
    o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = offs_n < L

        # Load K again
        k_ptrs = K_ptr + offs_n[:, None] * stride_k_l + offs_d[None, :] * stride_k_d
        k = tl.where(mask_n[:, None] & (offs_d[None, :] < D), tl.load(k_ptrs, mask=mask_n[:, None]), 0.0)

        # Recompute qk tile
        qk = tl.dot(q, tl.trans(k)) * scale

        if causal:
            query_pos = offs_m[:, None]
            key_pos = offs_n[None, :]
            causal_mask = key_pos > query_pos
            qk = tl.where(causal_mask, -float("inf"), qk)

        # Softmax normalize using accumulated max/sum
        exp_scores = tl.exp(qk - qk_max[:, None])
        probs = exp_scores / qk_sum[:, None]  # [M, N]

        # Load V: [BLOCK_N, BLOCK_D]
        v_ptrs = V_ptr + offs_n[:, None] * stride_v_l + offs_d[None, :] * stride_v_d
        v = tl.where(mask_n[:, None] & (offs_d[None, :] < D), tl.load(v_ptrs, mask=mask_n[:, None]), 0.0)

        # Accumulate output: o += probs @ v
        o += tl.dot(probs, v)

    # Store result
    o_ptrs = O_ptr + offs_m[:, None] * stride_o_l + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, o, mask=mask_m[:, None] & (offs_d[None, :] < D))


def fused_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Fused attention wrapper for a single (batch, head) slice.

    Args:
        q, k, v: Tensors of shape [L, D] on CUDA or ROCm device (torch device type "cuda")
        causal: whether to apply causal mask (no attending to future tokens)
        block_m, block_n, block_d: Triton tile sizes

    Returns:
        o: Tensor of shape [L, D] with attention output.
    """
    assert q.shape == k.shape == v.shape, "q, k, v must have same shape [L, D]"
    assert q.is_cuda, "q must be on a CUDA/ROCm device"
    L, D = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)

    grid = (triton.cdiv(L, block_m),)

    _fused_attn_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        L, D,
        causal=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    return o


def fused_attention_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Simple batched wrapper for attention over [B, H, L, D] shaped tensors.
    Iterates over batch and head dimensions and calls the single-head kernel.

    This is not the most efficient possible implementation (loop in Python),
    but it is straightforward and keeps the kernel focused on the [L, D] case.
    """
    assert q.shape == k.shape == v.shape, "q, k, v must have same shape [B, H, L, D]"
    assert q.dim() == 4, "Expected input shape [B, H, L, D]"

    B, H, L, D = q.shape
    device = q.device
    o = torch.empty_like(q, device=device)

    for b in range(B):
        for h in range(H):
            o[b, h] = fused_attention_triton(
                q[b, h], k[b, h], v[b, h], causal=causal
            )

    return o
