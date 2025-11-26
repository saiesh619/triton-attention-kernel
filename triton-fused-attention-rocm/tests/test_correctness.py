import math
import torch
import torch.nn.functional as F

from kernels.fused_attention import fused_attention_batched


def _torch_reference_attention(q, k, v, causal: bool):
    """
    Uses PyTorch's scaled dot-product attention as the reference.
    Expects q, k, v of shape [B, H, L, D].
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Rearrange to [L, B*H, D] or use built-in SDP if available
    # Here we use manual attention for clarity.
    q_ = q * scale  # [B, H, L, D]
    k_t = k.transpose(-1, -2)  # [B, H, D, L]

    scores = torch.matmul(q_, k_t)  # [B, H, L, L]

    if causal:
        mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)  # [B, H, L, L]
    out = torch.matmul(attn, v)          # [B, H, L, D]
    return out


def test_fused_attention_close_to_torch():
    device = torch.device("cuda")

    B, H, L, D = 2, 2, 128, 64

    torch.manual_seed(0)
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    for causal in [False, True]:
        ref = _torch_reference_attention(q, k, v, causal=causal)
        triton_out = fused_attention_batched(q, k, v, causal=causal)

        max_diff = (ref - triton_out).abs().max().item()
        print(f"[TEST] causal={causal}, max_diff={max_diff}")

        # Threshold can be tuned based on actual kernel implementation.
        assert max_diff < 1e-3, f"Max difference too high: {max_diff}"
