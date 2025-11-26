#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# On ROCm, PyTorch still uses device type "cuda" but maps to AMD hardware via ROCm.
# This script is mostly a semantic placeholder for now.
echo "[RUN] Triton fused attention benchmark on ROCm backend (PyTorch ROCm build required)"
python "${ROOT_DIR}/benchmarks/benchmark_attention.py"
