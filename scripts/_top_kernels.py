import json
import sys

path = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
with open(path) as f:
    ev = json.load(f)["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
agg = {}
for e in k:
    name = e.get("name", "")
    agg.setdefault(name, [0.0, 0])
    agg[name][0] += float(e.get("dur", 0))
    agg[name][1] += 1
print(f"kernels/step ~ {len(k)/n_steps:.0f}")
for name, (dur, count) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:18]:
    print(f"{dur/n_steps/1e3:8.3f} ms  x{count/n_steps:7.1f}/step  {name[:100]}")
