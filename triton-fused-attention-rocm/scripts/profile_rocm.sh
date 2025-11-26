#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[PROFILE] rocprof placeholder for fused attention on ROCm"
# Example once rocprof is available:
# rocprof --stats --hip-trace \
#   --out "${ROOT_DIR}/analysis/fused_attn_rocm.json" \
#   python "${ROOT_DIR}/benchmarks/benchmark_attention.py"

echo "Update this script with your local ROCm profiler commands."
