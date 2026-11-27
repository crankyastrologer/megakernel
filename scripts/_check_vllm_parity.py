import hashlib
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.prompts import make_prompt_ids

MID = "HuggingFaceTB/SmolLM2-135M-Instruct"

with open(os.path.join(ROOT, "results", "vllm.json")) as f:
    d = json.load(f)
rec = d["smollm2-135m|128"]

tok = AutoTokenizer.from_pretrained(MID)
ids = make_prompt_ids(tok, 128)[0].tolist()
h = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]
print("main prompt hash:", h, "vllm hash:", rec["prompt_hash"], "match:", h == rec["prompt_hash"])
print("main head:", ids[:8])
print("vllm head:", rec["prompt_ids_head"])

model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
inp = torch.tensor([ids], device="cuda")
with torch.inference_mode():
    out = model.generate(input_ids=inp, max_new_tokens=20, do_sample=False, use_cache=True, pad_token_id=0)
ref = out[0, 128:].tolist()
print("ref tokens:", ref)
print("vllm tokens:", rec["tokens"])
print("match:", ref == rec["tokens"])
print("ref text:", repr(tok.decode(ref)))
print("vllm text:", repr(tok.decode(rec["tokens"])))
