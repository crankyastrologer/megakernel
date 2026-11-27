# Batch-size sweep — HF eager, HF graph, vLLM

**Model:** SmolLM2-135M-Instruct, bf16, greedy, identical prompt per request.
**Hardware:** RTX 4060 Laptop, 24 SMs, 32 MB L2, WSL2.
**Steps:** 50 decode tokens; HF = 1 run, vLLM = median of 3 reps.
**Metrics:** per-step latency (each request gets one token per step), aggregate
decode throughput, and step-level roofline percentages.

> **Read this first.** This is a *scaling* sweep, not a replacement for the
> Part 1 B=1 numbers. vLLM was loaded with `max_model_len=2114` for the whole
> sweep, so its B=1 result (2.86 ms) differs from Part 1 (2.30 ms at 128). HF
> graph B=1 also varies with clock state (3.15 ms here vs 3.39 ms in Part 1).
> Use the **within-sweep B=1→8 trend** and the Part 1 numbers for absolute B=1
> claims.

Raw data:

- `results/batch_hf_eager.json`
- `results/batch_hf_graph.json`
- `results/batch_vllm.json`

---

## 1. Aggregate decode throughput (tok/s)

| B | HF eager 128 | HF graph 128 | vLLM 128 | HF eager 512 | HF graph 512 | vLLM 512 | HF eager 2048 | HF graph 2048 | vLLM 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.0 | 317.5 | 349.7 | 53.7 | 314.5 | 414.3 | 56.6 | 313.9 | 366.3 |
| 2 | 65.9 | 551.1 | 784.1 | 104.4 | 552.4 | 798.0 | 110.0 | 486.3 | 651.0 |
| 4 | 139.8 | 1075.1 | 1601.1 | 217.1 | 1030.8 | 1429.7 | 217.1 | 878.8 | 1217.5 |
| 8 | 232.5 | 2138.7 | **2953.9** | 312.8 | 1907.8 | **2610.6** | 354.4 | 1467.7 | **2051.1** |

For reference, the fused batch-1 path is **588 tok/s at 128** (1.70 ms) and
**503 tok/s at 2048** (1.99 ms). It does not support B>1.

At B=1 the fused path is the fastest; at B=8 vLLM is 5.0× faster in aggregate
throughput at 128 and 4.1× at 2048.

---

## 2. Per-request latency (ms/token)

Each request gets one token per step, so the step time is the per-request
latency. HF eager is host-bound and excluded here (18–52 ms/step).

| B | HF graph 128 | vLLM 128 | HF graph 2048 | vLLM 2048 |
|---:|---:|---:|---:|---:|
| 1 | 3.150 | 2.860 | 3.186 | 2.730 |
| 2 | 3.629 | 2.551 | 4.112 | 3.072 |
| 4 | 3.721 | 2.498 | 4.552 | 3.285 |
| 8 | 3.741 | 2.708 | 5.451 | 3.900 |

The HF graph degrades more with batch at 2048; vLLM stays within ~1.4× of its
B=1 latency while serving 8 requests.

---

## 3. Step-level roofline at 2048

Per step, weights are read once (269 MB) and KV is read B times (47.9 MB per
request at 2048). Total-traffic % uses both.

| B | HF graph weight % | HF graph total % | vLLM weight % | vLLM total % |
|---:|---:|---:|---:|---:|
| 1 | 32.0 | 37.7 | 37.4 | 44.0 |
| 2 | 24.8 | 33.6 | 33.2 | 45.0 |
| 4 | 22.4 | 38.4 | 31.1 | 53.1 |
| 8 | 18.7 | 45.3 | 26.2 | **63.4** |

Note what happens:

- **Weight-only % falls with B** because the weight read is fixed per step while
  the step gets slower (more KV, attention, and compute).
- **Total-traffic % rises with B** because KV bytes scale with B and dominate
  more of the step. At B=8/2048, vLLM reaches **63.4% of the total roofline**.
- The batch-1 problem is therefore a **latency/launch problem, not a bandwidth
  wall**: once there is enough batch to amortize weights, vLLM approaches the
  memory roofline.

At 128, vLLM's total-traffic % is 36.2 / 41.1 / 43.0 / 41.7 for B=1/2/4/8.

---

## 4. What this means

1. **Batching amortizes the 269 MB weight read.** The per-token weight floor is
   `1.021 / B` ms (0.128 ms at B=8), but the step also pays KV/attention/compute,
   so real B=8 throughput is 2.0–3.0k tok/s rather than the ~7k weights-only
   bound.
2. **vLLM is the throughput baseline.** At B=8 it serves 2954 tok/s at 128 and
   2051 tok/s at 2048, while keeping per-request latency at 2.7–3.9 ms.
3. **HF eager is not a serving baseline.** It is host-bound: 18–52 ms/step, and
   its step-level utilization is ~3–11% even at B=8.
4. **HF graph is the framework fallback.** It removes the host gap but the
   batch>1 path falls back to SDPA (the split-K fast-attn kernel is batch-1),
   so its per-request latency grows at 2048 and its total-traffic utilization
   tops out at 45%.
5. **The fused batch-1 kernel is a latency play.** At B=1 it is 1.70 ms (588
   tok/s) at 128, faster than vLLM's 2.86 ms in this sweep. It has no batch
   path, so it cannot compete on aggregate throughput.
6. **Phase 3 should stay batch-1.** The batch data shows the GPU can reach ~63%
   of the total roofline when given B=8; the batch-1 megakernel's remaining gap
   is scheduling/ramp, which is exactly Phase 3's target. Adding batch support
   would be a different project (and would compete with vLLM, not with the
   batch-1 latency story).

---

## 5. Caveats

- **HF graph B>1 uses SDPA fallback.** `bench/fastattn.py` now guards the split-K
  patch with `batch == 1`; B>1 uses the normal masked SDPA path. A batch-aware
  split-K could improve the HF graph numbers, so treat them as a fallback, not
  the best possible HF batch baseline.
- **vLLM max_model_len = 2114 for all runs.** This changes CUDA-graph capture and
  attention kernel selection versus the Part 1 per-prompt runs. Do not mix the
  B=1 vLLM number here with the Part 1 2.30 ms at 128.
- **HF graph B=1 is clock-sensitive.** It measured 3.15 ms here and 3.39 ms in
  Part 1; the within-sweep B scaling is the reliable signal.
- **Identical prompts.** This isolates batching from prompt-length imbalance and
  scheduling; real serving mixes lengths.
- **Greedy only.** No sampling, no speculative decoding.
- **No fused batch>1 path.** The fused decoder is single-token by construction.

---

## 6. Commands

```bash
cd /home/ansh/mega4060

# HF eager + graph batch sweep
.venv/bin/python scripts/batch_sweep_hf.py \
  --batches 1,2,4,8 --prompt-lengths 128,512,2048 \
  --modes eager,graph --steps 50 \
  --json results/batch_hf.json

# vLLM batch sweep (separate venv)
./scripts/run_batch_vllm.sh \
  --batches 1,2,4,8 --prompt-lengths 128,512,2048 \
  --steps 50 --reps 3 --json results/batch_vllm.json
```

The HF sweep writes one JSON per mode when run separately; the combined
`results/batch_hf.json` is the union if run with `--modes eager,graph`.
