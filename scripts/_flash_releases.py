import json
import urllib.request

url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=5"
req = urllib.request.Request(url, headers={"User-Agent": "mega4060"})
with urllib.request.urlopen(req, timeout=30) as r:
    releases = json.loads(r.read())
for rel in releases[:3]:
    print("RELEASE", rel["tag_name"])
    for a in rel.get("assets", []):
        n = a["name"]
        if "cp311" in n and "linux_x86_64" in n:
            print("   ", n)
