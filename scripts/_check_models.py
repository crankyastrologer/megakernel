import json
import urllib.request

CANDIDATES = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

for mid in CANDIDATES:
    url = f"https://huggingface.co/{mid}/resolve/main/config.json"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=20) as r:
            cfg = json.loads(r.read())
        print(f"OK   {mid}  max_pos={cfg.get('max_position_embeddings')}  layers={cfg.get('num_hidden_layers')}  hidden={cfg.get('hidden_size')}")
    except Exception as e:
        code = getattr(e, "code", None)
        print(f"FAIL {mid}  http={code}  {type(e).__name__}")
