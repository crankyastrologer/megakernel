"""vLLM offline benchmark (run with .venv-vllm python).

Measures steady-state TPOT from engine metrics (first_token_ts -> last_token_ts),
which excludes prefill, using CUDA graphs enabled (enforce_eager=False).
Writes results/vllm.json keyed by "<model>|<prompt_len>".
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench.prompts import make_prompt_ids  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def main():
    """Benchmark vLLM offline TPOT for the configured models and prompts.

    Loads each model with CUDA graphs enabled, warms up, repeats greedy
    generation, and derives TPOT from engine timestamps. Writes
    ``results/vllm.json`` keyed by ``"<model>|<prompt_len>"``.

    CLI flags: ``--models``, ``--prompt-lengths``, ``--steps``, ``--reps``,
    ``--gpu-mem``, ``--debug``.

    Returns:
        None.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--prompt-lengths", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "models.yaml")) as f:
        cfg = yaml.safe_load(f)
    keys = args.models.split(",") if args.models else list(cfg["models"].keys())
    rt = cfg["runtime"]
    plens = [int(x) for x in args.prompt_lengths.split(",")] if args.prompt_lengths else rt["prompt_lengths"]
    steps = args.steps or rt["measure_steps"]

    import torch
    import transformers
    import vllm
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "vllm.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {}

    data.setdefault("_meta", {})
    data["_meta"].update(
        {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "enforce_eager": False,
            "reps": args.reps,
            "steps": steps,
            "gpu_memory_utilization": args.gpu_mem,
        }
    )

    for key in keys:
        mid = cfg["models"][key]["id"]
        mcfg = AutoConfig.from_pretrained(mid)
        derived = getattr(mcfg, "max_position_embeddings", None) or 32768
        fit_plens = [p for p in plens if p + steps <= derived]
        skipped = [p for p in plens if p not in fit_plens]
        if skipped:
            print(
                f"WARNING {key}: skipping prompt lengths {skipped} "
                f"(prompt+{steps} > max_position_embeddings={derived})",
                flush=True,
            )
        if not fit_plens:
            print(f"WARNING {key}: no prompt length fits max context {derived}; skipping model", flush=True)
            continue
        max_model_len = min(max(fit_plens) + steps + 16, derived)
        print(f"\n=== vLLM {key} ({mid}) max_model_len={max_model_len} ===", flush=True)
        t0 = time.perf_counter()
        llm = LLM(
            model=mid,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=False,
            disable_log_stats=False,
            seed=0,
        )
        load_s = time.perf_counter() - t0
        print(f"engine ready in {load_s:.1f}s", flush=True)
        tok = llm.get_tokenizer()

        for plen in fit_plens:
            ids = make_prompt_ids(tok, plen)[0].tolist()
            prompt_hash = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]
            llm.generate(
                prompts=[TokensPrompt(prompt_token_ids=ids)],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
                use_tqdm=False,
            )

            params = SamplingParams(
                temperature=0.0, max_tokens=steps, ignore_eos=True, detokenize=False, seed=0
            )
            tpots = []
            walls = []
            last_tokens = []
            for r in range(args.reps):
                t = time.perf_counter()
                outs = llm.generate(
                    prompts=[TokensPrompt(prompt_token_ids=ids)],
                    sampling_params=params,
                    use_tqdm=False,
                )
                walls.append(time.perf_counter() - t)
                o = outs[0]
                gen = o.outputs[0]
                toks = list(gen.token_ids)
                m = o.metrics
                if args.debug and r == 0 and plen == plens[0]:
                    print("DEBUG metrics type:", type(m), flush=True)
                    print("DEBUG metrics repr:", m, flush=True)
                    if m is not None:
                        print("DEBUG metrics vars:", vars(m), flush=True)
                    print("DEBUG output attrs:", [a for a in vars(o) if "metric" in a or "time" in a], flush=True)
                    print("DEBUG n tokens:", len(toks), "finish:", gen.finish_reason, flush=True)
                n = len(toks)
                tpot = None
                if m is not None and n > 1:
                    a = getattr(m, "first_token_ts", None)
                    b = getattr(m, "last_token_ts", None)
                    if a is not None and b is not None and b > a:
                        tpot = (b - a) / (n - 1) * 1e3
                if tpot is None:
                    tpot = getattr(m, "mean_time_per_output_token", None)
                    if tpot is not None:
                        tpot *= 1e3
                tpots.append(tpot)
                last_tokens = toks

            valid = [x for x in tpots if x is not None]
            tpot_ms = statistics.median(valid) if valid else None
            rec = {
                "model": key,
                "model_id": mid,
                "prompt_len": plen,
                "steps": steps,
                "reps": args.reps,
                "tpot_ms": round(tpot_ms, 4) if tpot_ms else None,
                "tok_per_s": round(1e3 / tpot_ms, 3) if tpot_ms else None,
                "tpot_all_ms": [round(x, 4) if x else None for x in tpots],
                "wall_s_all": [round(w, 4) for w in walls],
                "tokens": last_tokens,
                "prompt_hash": prompt_hash,
                "prompt_ids_head": ids[:8],
                "engine_ready_s": round(load_s, 2),
            }
            data[f"{key}|{plen}"] = rec
            print(
                f"-- plen {plen}: TPOT {rec['tpot_ms']} ms  {rec['tok_per_s']} tok/s"
                f"  reps {rec['tpot_all_ms']}  wall {rec['wall_s_all']}",
                flush=True,
            )

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
