#!/usr/bin/env bash
# Download the SmolLM2-135M GGUF baselines into models/gguf/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models/gguf"
BASE="https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main"
FILES=(
  "SmolLM2-135M-Instruct-F16.gguf"
  "SmolLM2-135M-Instruct-Q4_K_M.gguf"
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  if [ -f "$DEST/$f" ]; then
    echo "already present: $DEST/$f"
  else
    echo "downloading $f"
    curl -L --fail -o "$DEST/$f" "$BASE/$f"
  fi
done
ls -lh "$DEST"
