import json
import urllib.request

for pkg in ["vllm"]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30) as r:
        data = json.load(r)
    info = data["info"]
    print(pkg, "latest:", info["version"], "requires_python:", info.get("requires_python"))
    rd = info.get("requires_dist") or []
    for line in rd:
        if any(k in line.lower() for k in ["torch", "transformers", "triton", "xformers", "flashinfer", "numpy"]):
            print("  ", line)
    print("recent releases:", sorted(data["releases"].keys())[-12:])
