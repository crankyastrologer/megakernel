#!/usr/bin/env bash
# Build llama.cpp with CUDA at the pinned commit (idempotent).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_COMMIT="${LLAMA_COMMIT:-b7ad48ebd}"
JOBS="${JOBS:-6}"
CUDA_ARCH="${CUDA_ARCH:-89}"
export PATH="$ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Install it with: uv pip install --python .venv/bin/python cmake ninja" >&2
  exit 1
fi

if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git fetch --all --quiet || true
git checkout "$LLAMA_COMMIT"

cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF
cmake --build build -j "$JOBS"

echo "llama.cpp built at $LLAMA_DIR/build/bin"
