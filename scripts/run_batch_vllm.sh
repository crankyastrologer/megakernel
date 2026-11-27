#!/usr/bin/env bash
# Run the vLLM batch sweep with the vLLM venv and CUDA on PATH.
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec .venv-vllm/bin/python scripts/batch_sweep_vllm.py "$@"
