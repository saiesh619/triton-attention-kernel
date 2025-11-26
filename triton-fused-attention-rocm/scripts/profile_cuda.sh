#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[PROFILE] Nsight Systems placeholder for fused attention on CUDA"
# Example once Nsight is installed:
# nsys profile -o "${ROOT_DIR}/analysis/fused_attn_cuda" \
#   python "${ROOT_DIR}/benchmarks/benchmark_attention.py"

echo "Update this script with your local Nsight setup to collect real traces."
