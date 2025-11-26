# Benchmark Results

This file captures simple runtime comparisons between:

- PyTorch reference scaled dot-product attention
- Triton fused attention kernel (CUDA backend)
- Triton fused attention kernel (ROCm backend, if available)

## Example Configuration

- Batch size (B): 2
- Heads (H): 8
- Sequence length (L): 1024
- Head dimension (D): 64
- Dtype: float32

## Placeholder Results

- PyTorch attention: TODO ms
- Triton fused attention (CUDA): TODO ms
- Triton fused attention (ROCm): TODO ms
