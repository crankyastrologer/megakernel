#!/usr/bin/env bash
# Phase 0: environment, CUDA graph, cooperative launch, profiler.
set -u

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
export PATH="/usr/local/cuda/bin:$PATH"
mkdir -p results

echo "=== nvidia-smi ==="
nvidia-smi | tee results/phase0_nvidia_smi.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem \
  --format=csv,noheader | tee -a results/phase0_nvidia_smi.txt

echo
echo "=== tool versions ==="
$PY - <<'PYEOF'
import torch, triton, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("triton", triton.__version__)
print("transformers", transformers.__version__)
PYEOF

echo
echo "=== python env + cuda graph + profiler ==="
$PY scripts/phase0.py 2>&1 | tee results/phase0_python.log
PY_RC=${PIPESTATUS[0]}

echo
echo "=== cooperative kernel (nvcc) ==="
NVCC_RC=1
if nvcc -arch=sm_89 -rdc=true -O2 -o results/coop_test kernels/cuda/coop_test.cu 2>&1 | tee results/phase0_nvcc.log; then
  NVCC_RC=0
  results/coop_test 2>&1 | tee results/phase0_coop.log
  COOP_RC=${PIPESTATUS[0]}
else
  COOP_RC=1
  echo "nvcc compile failed" | tee results/phase0_coop.log
fi

echo
echo "=== memory bandwidth ==="
$PY bench/membw.py 2>&1 | tee results/phase0_membw.log

echo
echo "=== nsys (best effort) ==="
NSYS_RC=1
if command -v nsys >/dev/null 2>&1; then
  nsys --version | tee results/phase0_nsys.log
  nsys profile --force-overwrite=true -t cuda -o results/nsys_smoke $PY bench/nsys_smoke.py \
    >> results/phase0_nsys.log 2>&1
  if [ -f results/nsys_smoke.nsys-rep ]; then
    nsys stats --report cuda_gpu_kern_sum results/nsys_smoke.nsys-rep >> results/phase0_nsys.log 2>&1
    if grep -q "does not contain CUDA kernel data" results/phase0_nsys.log; then
      NSYS_RC=1
    elif grep -q "cuda_gpu_kern_sum" results/phase0_nsys.log; then
      NSYS_RC=0
    fi
  fi
else
  echo "nsys: not found" | tee results/phase0_nsys.log
fi

echo
echo "=== phase 0 summary ==="
printf 'python(graph+profiler): %s\n' "$([ $PY_RC -eq 0 ] && echo PASS || echo FAIL)"
printf 'nvcc compile:           %s\n' "$([ $NVCC_RC -eq 0 ] && echo PASS || echo FAIL)"
printf 'cooperative launch:     %s\n' "$([ $COOP_RC -eq 0 ] && echo PASS || echo FAIL)"
printf 'nsys gpu trace:         %s\n' "$([ $NSYS_RC -eq 0 ] && echo PASS || echo 'FAIL/UNAVAILABLE (see results/phase0_nsys.log)')"

if [ $PY_RC -eq 0 ] && [ $NVCC_RC -eq 0 ] && [ $COOP_RC -eq 0 ]; then
  echo "PHASE0: PASS"
  exit 0
else
  echo "PHASE0: FAIL"
  exit 1
fi
