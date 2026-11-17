#!/usr/bin/env bash
# Install vLLM into its own venv (vLLM pins its own torch/transformers).
set -u
cd "$(dirname "$0")/.."

if [ ! -x .venv-vllm/bin/python ]; then
  uv venv --python 3.11 .venv-vllm
fi

uv pip install --python .venv-vllm/bin/python vllm==0.33.0 2>&1 | tee results/vllm_install.log
RC=${PIPESTATUS[0]}
echo "install_rc=$RC" | tee -a results/vllm_install.log

if [ $RC -ne 0 ]; then
  exit $RC
fi

.venv-vllm/bin/python - <<'PY' 2>&1 | tee -a results/vllm_install.log
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import vllm
print("vllm", vllm.__version__)
PY
