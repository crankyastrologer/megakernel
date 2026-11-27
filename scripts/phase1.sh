#!/usr/bin/env bash
# Phase 1: baselines + vLLM + roofline.
#   MODELS=smollm2-135m,tinyllama-1.1b PLENS=128,512,2048 STEPS=50 ./scripts/phase1.sh
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PATH="$ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
mkdir -p results

MODELS=${MODELS:-smollm2-135m,qwen2.5-1.5b}
PLENS=${PLENS:-128,512,2048}
STEPS=${STEPS:-}
WARMUP=${WARMUP:-}

STEP_ARG=""
[ -n "$STEPS" ] && STEP_ARG="--steps $STEPS"
WARM_ARG=""
[ -n "$WARMUP" ] && WARM_ARG="--warmup $WARMUP"

VERS=$("$ROOT/.venv/bin/python" -c "import torch,triton,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__,'transformers',transformers.__version__)")

{
  echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "models: $MODELS"
  echo "prompt_lengths: $PLENS"
  echo "measure_steps: ${STEPS:-config default}"
  echo "warmup_steps: ${WARMUP:-config default}"
  echo "compile_mode: ${COMPILE_MODE:-default}  (reduce-overhead segfaults on this stack)"
  echo "main venv: $VERS"
  echo "build: MODELS=$MODELS PLENS=$PLENS STEPS=$STEPS ./scripts/phase1.sh"
} | tee results/phase1_versions.txt

echo
echo "=== vLLM baseline ==="
if [ -x .venv-vllm/bin/python ]; then
  ./scripts/run_vllm_bench.sh --models "$MODELS" --prompt-lengths "$PLENS" $STEP_ARG 2>&1 \
    | tee results/phase1_vllm.log
else
  echo "vllm venv missing; skipping" | tee results/phase1_vllm.log
fi

echo
echo "=== torch decode modes ==="
"$ROOT/.venv/bin/python" scripts/phase1.py --models "$MODELS" --prompt-lengths "$PLENS" \
  $STEP_ARG $WARM_ARG --profile 2>&1 | tee results/phase1_run.log

echo
echo "=== token parity gate ==="
"$ROOT/.venv/bin/python" tests/test_token_parity.py 2>&1 | tee results/phase1_parity.log
