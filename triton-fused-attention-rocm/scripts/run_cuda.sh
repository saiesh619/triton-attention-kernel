#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[RUN] Triton fused attention benchmark on CUDA backend"
python "${ROOT_DIR}/benchmarks/benchmark_attention.py"
