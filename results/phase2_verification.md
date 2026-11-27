# Phase 2 verification and benchmark protocol

What was checked, where the code lives, how the numbers were produced, and what
is still uncertain.

**Scope:** SmolLM2-135M-Instruct, bf16, batch 1, greedy, RTX 4060 Laptop, WSL2.
**Canonical headline:** 3.39 ms HF graph -> 1.70 ms fused at 128 tokens, 50-token
greedy parity exact. New medians at 512/2048: 1.73 / 1.99 ms.

---

## 1. Verification code

| check | file | what it verifies | last result |
|---|---|---|---|
| RoPE | `tests/test_rope_kv.py` | q/k vs HF `apply_rotary_pos_emb` at positions 0/1/127/128/2047; v exact; cache slot; untouched slots; pos-0 identity | PASS, max 0.015625, v exact |
| QKV GEMV | `tests/test_qkv_gemv.py` | packed qkv vs `F.linear(F.rms_norm(...))`; P=1 and P=4 partial sums | PASS, max 0.015625 |
| o/down GEMV | `tests/test_gemv_res.py` | y vs `res + F.linear`; partial sums vs `sum(y_bf16^2)` | PASS, y 0.031/0.125 (1 ULP), ss diff <= 0.002 |
| gate/up + SiLU | `tests/test_gate_up.py` | SwiGLU vs torch; P=1 and P=18; `bump_seq` | PASS, max 0.007812; bump 5->6 |
| end-to-end | `tests/test_fused_parity.py` | 50 greedy tokens at prompt 128, fused vs manual eager | PASS, first diff `None`, 1.70 ms |
| end-to-end sweep | `scripts/phase2_verify.py` | 50 tokens at 128/512/2048; median/p95; both roofline percentages; JSON | PASS all lengths |
| trace analysis | `scripts/analyze_trace.py` | per-kernel times; HF q/k/v/o/gate/up/down split; fused o/down split | reproduces comparison doc |

Part 1's `tests/test_token_parity.py` remains the eager/static/graph/compile gate
on SmolLM2; it is not part of the fused path.

### Unit-test tolerances

The unit tests compare bf16 tensors produced by different reduction orders, so
they use tolerances at the bf16-ULP level:

- RoPE / qkv: `0.02` (observed max `0.015625`).
- gate/up: `0.1` (observed `0.007812`).
- gemv_res: y `< 0.5` (observed `0.031` o, `0.125` down); partial sums `< 1e-2`
  (observed `1.2e-4` and `2e-3`).

The real gate is not the tolerance: it is **exact greedy token parity** in
`test_fused_parity.py` / `phase2_verify.py`.

---

## 2. End-to-end verification output

`scripts/phase2_verify.py --prompt-lengths 128,512,2048 --steps 50
--modes graph,graph_fused`:

| plen | mode | median ms | p95 ms | parity | % weight | % total |
|---:|---|---:|---:|---|---:|---:|
| 128 | graph | 3.386 | 3.394 | True | 30.1 | 30.5 |
| 128 | graph_fused | **1.711** | 1.732 | True | 59.7 | 60.4 |
| 512 | graph | 2.943 | 3.005 | True | 34.7 | 36.3 |
| 512 | graph_fused | **1.726** | 1.731 | True | 59.2 | 61.9 |
| 2048 | graph | 3.439 | 3.492 | True | 29.7 | 35.0 |
| 2048 | graph_fused | **2.003** | 2.029 | True | 51.0 | 60.0 |

The dedicated runs used in the docs measured **1.70 / 1.73 / 1.99 ms**; this
sweep is within 0.01-0.015 ms of those values.

### HF baseline variance (important)

The HF graph is more variable than the fused path:

| measurement | 128 | 512 | 2048 |
|---|---:|---:|---:|
| Part 1 canonical (`phase1.csv`) | 3.39 | 3.43 | 3.72 |
| `phase2_verify` same process | 3.386 | **2.943** | 3.439 |
| fresh HF-512 run | — | **3.171** | — |

A 512-token HF graph that is faster than the 128-token run is not a context
effect. The fresh 512 run logged `2310 MHz SM / 7001 MHz mem / 13.2 W` after
timing; the GPU was not in the same sustained state as the Part 1 runs. The
fused path is weight-dominated and stayed stable across processes (1.70-1.73).

Consequence: quote the **Part 1 canonical HF numbers** for the headline, and
treat the fused absolute numbers (which are stable) as the reliable result. The
speedup vs HF is 1.98x at 128 against 3.39, and in the range 1.7-1.9x at longer
context depending on the HF clock state.

---

## 3. Timing protocol (full step)

Code: `bench/harness.py`.

1. **Steady-clock burn:** `harness.burn()` runs 2048x2048 bf16 matmuls for 0.3 s
   (`BURN_SECONDS`). WSL2 cannot lock clocks, so this is the standard from Part 1.
2. **Warm-up:** 10 decode steps are discarded by `time_stats(times, 10)`.
3. **Measurement:** 50 decode steps, each wrapped in CUDA events:
   ```python
   t0.record(); step_fn(); t1.record()
   ```
   One synchronize at the end; per-step time = `t0.elapsed_time(t1)`. Tokens are
   cloned for parity; no `.item()` inside the loop.
4. **Statistics:** median (primary), mean, p95, min, max over the measured steps.

For `graph_fused`, `bench/fused.py::FusedDecoder.__init__` does:

```text
HF prefill -> StaticCache
pack weights -> [L,960,576] qkv, o, gate, up, down
build RoPE tables
run 2 eager warm-up steps (JIT compile)
reset seq = prompt_len, reset token = prefill argmax
capture one step in torch.cuda.CUDAGraph
replay once -> second token
```

The timed loop then replays the graph. The graph is self-feeding: argmax writes
the token buffer that `embed_lookup` reads on the next replay.

**L2 note:** the full-step timing does **not** flush L2 between replays. The
weight working set is 269 MB (> 32 MB L2), so each step is mostly a cold weight
stream, but the last ~32 MB of the previous step can be resident. The per-kernel
microbenchmarks use an explicit read-only flush.

---

## 4. Profiling protocol

Code: `bench/breakdown.py`.

- Build the requested mode with `harness.build_decoder`.
- Warm up, burn, time 10 steps with CUDA events (wall).
- Profile 5 steps with `torch.profiler` (`CPU + CUDA`), export a chrome trace.
- Classify kernels:
  - `attention`: names containing flash/attention/attn/sdpa
  - `gemm_weights`: gemm/sgemm/cutlass/cublas/gemv/wmma/gate_up/swiglu/addmm/addmv
  - `sampling`: argmax/topk/sort
  - `memcpy`: memcpy/memset
  - `elementwise_other`: the rest
- Report wall, kernel sum, host gap (`wall - kernel`), kernels/step, categories.

CUPTI inflates kernel times by ~10% under WSL; use the trace for attribution,
the CUDA-event wall time for the headline.

Trace files:

```text
results/trace_breakdown_graph_128.json        (HF)
results/trace_breakdown_graph_fused_128.json  (fused)
results/breakdown_graph_128.json
results/breakdown_graph_fused_128.json
```

Reproduce the comparison doc's per-projection split:

```bash
.venv/bin/python scripts/analyze_trace.py results/trace_breakdown_graph_128.json --steps 8
.venv/bin/python scripts/analyze_trace.py results/trace_breakdown_graph_fused_128.json --steps 5
```

The HF split assumes 210 `gemvx` events per step in q/k/v/o/gate/up/down order;
the fused split assigns alternating `gemv_res` events to o and down.

---

## 5. Per-kernel microbenchmark protocol (tuning only)

Used for the qkv norm split, gate/up tiling, and the down regression. Not the
source of the headline numbers.

```text
- 30 layers, real weights, one kernel per layer in a CUDA graph
- read-only L2 flush: torch.sum(256 MB bf16 buffer) before each replay
  (a zero_ write flush leaves dirty lines and inflated the qkv number)
- 15-20 replays, median
- cold data, so the result is HBM-bound
```

Why the flush: 30 qkv matrices are 33.2 MB, almost exactly the 32 MB L2. Without
a flush the loop measured 0.516 ms = 244% of roofline. With a read-only flush it
measured 0.31 ms at BK=64; the sweep then found BK=128/warps=8 for down.

These are tuning numbers. The comparison doc's per-kernel values come from the
full-step profiler trace, because that includes the real cache state and layer
sequence.

---

## 6. Roofline protocol

Code: `bench/membw.py`, `bench/roofline.py`.

- `bench/membw.py` streams a 512 MB bf16 buffer with a Triton read kernel; six
  block/warp configs, 50 iterations, median. Best: **263.5 GB/s** at 9022 MHz.
- Weight floor: `roofline.weight_bytes(model)` = **269.025 MB** -> **1.021 ms**.
- KV bytes: `2 * layers * kv_heads * head_dim * seq_len * 2 B`.
  - 128: 3.6 MB; 512: 12.5 MB; 2048: 47.9 MB.
- Weight-only % = `weight_bytes / time / 263.5`.
- Total-traffic % = `(weight_bytes + kv_bytes) / time / 263.5`.

The total-traffic percentage is the fair context-scaling metric: it stays at
~60-62% across 128/512/2048, while the weight-only percentage falls from 60% to
51% because the denominator ignores the growing KV traffic.

**Clock caveat:** `nvidia-smi` reported **7001 MHz memory** after several decode
runs, not the 9022 MHz used for the 263.5 GB/s denominator. If the memory clock
was not sustained, the absolute roofline percentages are conservative. The
relative engine comparison shares the same conditions.

---

## 7. Known limitations / what a reviewer should probe

1. **HF baseline variance.** Use the Part 1 canonical HF numbers (3.39 / 3.43 /
   3.72) for the headline; the same-process HF graph can measure 2.94-3.43 ms at
   512 depending on clock state. Re-run HF and fused interleaved with clocks
   logged for a strict comparison.
2. **Memory clock.** 7001 MHz observed after runs vs 9022 MHz in `membw.json`.
   A stock/sustained-clock study is still pending.
3. **Profiler inflation.** Kernel times are ~10% higher than wall; don't compare
   profiled kernel sums to CUDA-event latency without that caveat.
4. **Prefix token count.** For 50 steps, fused returns 52 tokens (2 prefix + 50)
   and eager returns 51 (1 prefix + 50). Parity compares the first 51, which
   covers the prefill token and all 50 generated tokens; the extra fused token is
   ignored. This is intentional, but don't compare list lengths.
5. **No per-step L2 flush** in the full-step timing; the microbench does flush.
6. **Head is still torch.** Final RMSNorm + `F.linear` + `argmax`; the fused
   layer path is what is verified, not a fused head.
7. **One model / one prompt / greedy only.** No sampling, batch > 1, or Qwen
   fused path.
8. **Parity is prompt-specific.** Exact 50-token parity on the fixed prompt is
   strong evidence, not a proof for all inputs.

---

## 8. Commands

```bash
cd /home/ansh/mega4060

# unit + end-to-end gates
.venv/bin/python tests/test_rope_kv.py
.venv/bin/python tests/test_qkv_gemv.py
.venv/bin/python tests/test_gemv_res.py
.venv/bin/python tests/test_gate_up.py
.venv/bin/python tests/test_fused_parity.py

# full parity/timing sweep (writes results/phase2_verify.json)
.venv/bin/python scripts/phase2_verify.py \
  --prompt-lengths 128,512,2048 --steps 50 --modes graph,graph_fused \
  --json results/phase2_verify.json

# profiling
.venv/bin/python bench/breakdown.py --mode graph_fused --prompt-len 128 --steps 10 --profile-steps 5
.venv/bin/python bench/breakdown.py --mode graph --prompt-len 128
.venv/bin/python scripts/analyze_trace.py results/trace_breakdown_graph_fused_128.json --steps 5
.venv/bin/python scripts/analyze_trace.py results/trace_breakdown_graph_128.json --steps 8

# roofline
.venv/bin/python bench/membw.py
```

The Part 1 baseline commands (vLLM, llama.cpp) are in `README.md` and
`results/part1_notes.md`.
