import os
import shutil
import subprocess
import sys

print("docker:", shutil.which("docker"))
try:
    out = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20)
    print("docker server:", out.stdout.strip(), "rc", out.returncode, out.stderr.strip()[:200])
except Exception as e:
    print("docker error:", repr(e))

print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")))
for p in [os.path.expanduser("~/.cache/huggingface/hub")]:
    print(p, "exists", os.path.isdir(p))
    if os.path.isdir(p):
        print("  entries:", sorted(os.listdir(p))[:20])

import urllib.request

for url in [
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json",
]:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            print(url, "->", r.status, len(r.read()))
    except Exception as e:
        print(url, "-> ERR", repr(e))
