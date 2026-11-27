import json
import urllib.request

CANDIDATES = [
    "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
    "bartowski/SmolLM2-135M-Instruct-GGUF",
    "ggml-org/SmolLM2-135M-Instruct-GGUF",
    "unsloth/SmolLM2-135M-Instruct-GGUF",
]

for repo in CANDIDATES:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            files = json.loads(r.read())
        ggufs = [f["path"] for f in files if f["path"].endswith(".gguf")]
        print(f"OK   {repo}  ({len(ggufs)} gguf)")
        for f in ggufs[:20]:
            print("     ", f)
    except Exception as e:
        print(f"FAIL {repo}  {getattr(e, 'code', type(e).__name__)}")
