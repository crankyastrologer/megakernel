#!/usr/bin/env bash
# Run the vLLM benchmark with the venv/bin (ninja) and CUDA toolkit on PATH.
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
# Avoid FlashInfer JIT sampler compilation; greedy decode is unchanged.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec .venv-vllm/bin/python bench/vllm_bench.py "$@"
