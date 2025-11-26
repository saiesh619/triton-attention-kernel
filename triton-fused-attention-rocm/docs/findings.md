# Findings & Design Notes

## Kernel Design

- Fused QK^T, softmax, and V multiplication into a single Triton kernel.
- Structured around [BLOCK_M, BLOCK_N, BLOCK_D] tiles to balance occupancy and
  memory bandwidth.

## Numerical Behavior

- Compared against PyTorch's manual attention implementation.
- Max absolute difference threshold targeted: < 1e-3 for FP32.

## Cross-Backend Behavior

- CUDA vs ROCm performance, occupancy, and codegen differences will be recorded here.
- Any ROCm-specific tuning (e.g., block sizes, vectorization) will be documented.
