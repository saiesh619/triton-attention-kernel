import time
from contextlib import contextmanager

import torch

from kernels.fused_attention import fused_attention_batched


@contextmanager
def timed(msg: str):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    print(f"{msg}: {(end - start) * 1000.0:.3f} ms")


def benchmark_attention(B=2, H=8, L=1024, D=64, causal=False, warmup=5, iters=20):
    device = torch.device("cuda")

    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    # Warmup PyTorch reference
    from tests.test_correctness import _torch_reference_attention
    for _ in range(warmup):
        _ = _torch_reference_attention(q, k, v, causal=causal)

    # Warmup Triton
    for _ in range(warmup):
        _ = fused_attention_batched(q, k, v, causal=causal)

    # Benchmark PyTorch reference attention
    with timed("[BENCH] PyTorch attention"):
        for _ in range(iters):
            _ = _torch_reference_attention(q, k, v, causal=causal)

    # Benchmark Triton fused attention
    with timed("[BENCH] Triton fused attention"):
        for _ in range(iters):
            _ = fused_attention_batched(q, k, v, causal=causal)


if __name__ == "__main__":
    benchmark_attention()
