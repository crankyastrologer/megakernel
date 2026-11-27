# Phase 2 engineering notebook — Triton fusion

Running log for Phase 2. Companion to `PLAN.md`. Updated at each checkpoint
(CP0–CP4) and whenever a measurement changes a decision.

- **Status:** CP3 down tuning 2026-11-26; 1.70 / 1.73 / 1.99 ms, parity green.
- **Last updated:** 2026-11-26.
- **Scope:** SmolLM2-135M, batch-1 greedy decode, bf16, RTX 4060 Laptop.
- **Target:** 1.5–2.1 ms/token with exact 50-token greedy parity.

Kernel comparison: `results/phase2_comparison.md` (HF vs fused, stage by stage).
Verification and benchmark protocol: `results/phase2_verification.md`.
Batch scaling: `results/batch_sweep.md` (HF eager/graph/vLLM at B=1/2/4/8).

Purpose: keep the debugging trail, the numbers behind each decision, and the
architecture rationale in one place. The article should use this as the source
for "what went wrong" and "why the kernel looks like this."

---

## 1. Architecture snapshot

### Vanilla vs fused

HF graph, per decoder layer (~38 kernels, every intermediate materialized):

```text
x -> norm: pow/mean/rsqrt/mul/cast/mul -> xn
  -> q_proj / k_proj / v_proj (3 cuBLAS GEMVs) -> views/transposes
  -> RoPE: cat/neg, mul, mul, add
  -> index_copy(k), index_copy(v)
  -> SDPA -> reshape/contiguous -> o_proj -> residual add
  -> norm (again) -> gate/up GEMVs -> silu -> mul -> down GEMV -> residual add
```

Fused, per layer (7 kernels, only reductions hit memory):

```text
x -> qkv_gemv (norm scaling + packed qkv) -> qkv
  -> rope_kv_append (rotate q/k, write k/v cache)
  -> split-K attention + reduce
  -> gemv_res (o_proj + residual, writes norm partials)
  -> rms_gate_up_swiglu (norm scaling + both GEMVs + SiLU*up)
  -> gemv_res (down + residual, writes norm partials)
```

Plus `embed_lookup` once and a two-stage `lm_head + argmax` at the end.
Target: ~215 kernels/step instead of 1,372.

### The boundary rule

> A kernel boundary is a reduction boundary.

Pointwise ops, broadcasts, residuals, activations, and scaling fuse into the
producer/consumer. A stage that needs a full reduction over an intermediate
produced across CTAs stays a boundary (or needs a grid barrier, which is Phase 3).

Consequences:

- `RMSNorm + QKV` is **not** fully fused: the sum-of-squares is a reduction.
  Compute it once in the producer epilogue (or a 1-CTA stats kernel); fuse only
  the scaling.
- `gate/up + down` cannot fuse: every down row needs all 1536 intermediates.
- `attention + o_proj` cannot fuse: o_proj reduces over all 9 heads.
- `lm_head + argmax` fuses as a two-stage reduction (local max per block, then
  a tiny final reduce).

### Kernel map (target 135M: H=576, I=1536, L=30, V=49152, 9 q / 3 kv heads, D=64)

| # | kernel | grid | notes |
|---|---|---|---|
| 1 | `embed_lookup` | 1 | one row gather; tied with lm_head |
| 2 | `qkv_gemv` + `norm_stats`/partials | 30–60 | scaling + packed 960×576 GEMV |
| 3 | `rope_kv_append` | 15 | 9 q heads, 3 k, 3 v; writes cache at `seq-1` |
| 4 | split-K attention + reduce | 9×16 + 9 | reuse `bench/fastattn.py` |
| 5 | `gemv_res` (o) | 18 | residual + partial sums for next norm |
| 6 | `rms_gate_up_swiglu` | 24–48 | scaling + two GEMVs + SiLU·up |
| 7 | `gemv_res` (down) | 18 | residual + partials for next layer |
| 8–9 | `lm_head + argmax` | 768 + 1 | bf16 logits, first-index tie-break |

### Docstring convention

Every `@triton.jit` kernel documents **Inputs**, **Outputs**, **Math**, and
**Grid**. This is for reviewer sanity and for the writeup appendix; the math
section must state the rounding points (bf16 storage vs fp32 accumulation),
because those are where parity is won or lost.

---

## 2. Bug and trap catalogue

| # | Symptom | Cause | Fix | Lesson |
|---|---|---|---|---|
| 1 | Triton compile error on `offs_n` | `tl.arrange` typo (5 sites) | `tl.arange` | Triton has no `arrange`; errors point at generated code, read the source line |
| 2 | `tl.lsum` not found | typo | `tl.sum` | — |
| 3 | `arange's arguments must be of type tl.constexpr` | `half = D // 2` lost constexpr-ness; `tl.arange(0, half)` rejected | use `tl.arange(0, D // 2)` inline | constexpr does not survive every local assignment |
| 4 | Fused MLP token mismatch in the first spike | forgot the RMSNorm weight `gamma` | multiply `xhat` by `norm_w` | compare per-stage max-abs before token parity |
| 5 | Would-be RoPE parity failure | adjacent-pair rotation instead of `rotate_half` halves | 64-dim vector split into `[0:32]` and `[32:64]`; `cos/sin` are duplicated halves | Llama uses the half-split convention |
| 6 | Would-be cache corruption | one program writing `seq+1` inside the append kernel | increment `seq` once per step before the layers; append at `seq-1` | no block ordering inside a kernel; a late block can read the incremented value |
| 7 | Would-be wrong positions | incrementing `seq` per layer | increment once per **token** | cache position is per token, not per layer |
| 8 | QKV measured at 244% of roofline (0.516 ms) | 30 qkv matrices = 33.2 MB ≈ 32 MB L2; loop re-read from L2 | flush L2 with a read-only 256 MB buffer between replays | single-layer and L2-hot microbenchmarks lie |
| 9 | Cold qkv looked worse than it should | flushed with `flush.zero_()` (128 MB write); dirty-line writeback competed with qkv | use a read-only flush (`torch.sum` of a 256 MB bf16 buffer) | the flush method is part of the experiment |
| 10 | QKV eager timing 0.42–0.52 ms | 30 eager Triton launches, ~10 µs Python overhead each | capture the 30-layer loop in a CUDA graph | measure graph replay for GPU time; eager for host overhead |
| 11 | QKV only 40% of roofline | norm reduction fused into every block; 60 blocks each do a 576-element sum before any weight load | move sum-of-squares out: `norm_stats` now, producer partials next | fusing a reduction into every consumer block serializes the stream |
| 12 | `padding_idx=2` concern | assumed HF might zero the pad row | verified `F.embedding` forward is a plain gather even for token 2 | test, don't assume |
| 13 | Double-read risk for lm_head | tied embeddings | same `data_ptr`; use one tensor, no clone | verify `tie_word_embeddings` |
| 14 | bf16 parity drift | mismatched rounding points in RMSNorm | round `x*rstd` to bf16, then multiply by bf16 `gamma` | match the reference's storage boundaries |
| 15 | `gemv_res` raised `NameError: pid is not defined` | the docstring edit replaced the signature and accidentally dropped `pid = tl.program_id(0)` | restored the line; added `tests/test_gemv_res.py` | after a mechanical edit, compile and run every affected kernel, not just `py_compile` |
| 16 | 2048 timing 2.34 ms vs 1.98 ms | thermal/order variance; `nvidia-smi` showed mem clock 7001 MHz after the run, so the 263.5 GB/s denominator was not sustained | re-measured in a fresh process: 2.06 ms | log clocks; re-run suspicious points in isolation |
| 17 | down+res 0.406 ms vs 0.278 GEMM + ~0.06 residual | BK=64 gave 24 loop iterations and poor streaming at N=576 | BK=128 + 8 warps -> 0.321 ms | sweep BK for large-K GEMVs; larger K needs larger BK |

---

## 3. Chronological log

### 2026-11-19 — feasibility spikes

- Built a 30-layer fused Triton chain with real weights (269 MB working set),
  CUDA-graph captured, attention stubbed:
  - torch HF-style chain: **2.32 ms/step**
  - fused Triton: **1.29 ms/step (79.3% of 263.5 GB/s)**
  - + measured 0.17 ms attention ⇒ ~1.46 ms; realistic with rope/embed/head 1.5–1.8 ms.
- Fused MLP only (norm+gate/up+SwiGLU, down+residual), real model/prompt:
  **50/50 greedy tokens identical** to manual eager, despite bf16-ULP diffs.
- Conclusion: exact token parity is achievable; Triton GEMV is competitive.

### 2026-11-25 — first kernels / typos

- `kernels/triton/decode.py` created with `embed_lookup` and `rms_qkv`.
- Five `tl.arrange` typos and one `tl.lsum` typo. Fixed.
- `embed_lookup` verified bitwise exact against `embed_tokens.weight[token]`;
  shape `[49152, 576]` bf16; `padding_idx=2`; tied with lm_head; token 2 is a
  plain gather, not zeroed.

### 2026-11-26 — rope, qkv parity, L2 trap, norm finding

- `rope_kv_append` implemented and tested at positions 0/1/127/128/2047:
  - q/k max diff **0.015625** (2 bf16 ULP) vs HF `apply_rotary_pos_emb`
  - v exact, cache placement exact, untouched slots zero
  - position 0 is exact identity
- `rms_qkv` parity vs `F.linear(F.rms_norm(...))`: **0.015625** max (q/k),
  **0.000488** (v) at BN=32/64.
- First qkv timing was 0.516 ms (244% roofline): 33.2 MB working set sits at
  the 32 MB L2 boundary. With a read-only flush:
  - `rms_qkv`: 0.312 ms (40.3%)
  - pure GEMV (norm removed): 0.212 ms (59.4%)
  - pure weight read: 0.209 ms (60.3%); best config 0.177 ms (71.1%, BN=16, warps=8)
- Diagnosis: the per-block norm reduction stalls the weight stream (~3.3 µs/layer,
  ~0.10 ms total). `num_stages` had no effect.
- Added `norm_stats` + `qkv_gemv` (P partial sums):
  - stats + gemv: **0.258 ms (48.8%)**
  - gemv only: 0.217 ms (58.0%)
  - producer partials projected **~0.22 ms**
- Added tests: `tests/test_rope_kv.py`, `tests/test_qkv_gemv.py` (P=1, P=4).
- Added `gemv_res` (o_proj/down_proj + residual + next-norm partials).
  Parity: y within 1 bf16 ULP (o 0.03125, down 0.125); `sum(ss)` matches
  `sum(y_bf16^2)` to <= 2e-3.
- Added docstrings to every kernel: Inputs, Outputs, Math, Grid.
- The docstring edit dropped `pid = tl.program_id(0)` in `gemv_res`; the new
  parity test caught it immediately. Restored.
- Added `tests/test_gemv_res.py`.
- Updated `PLAN.md` kernel table + CP0 note.

### 2026-11-26 — CP2: full fused step

- Implemented `bench/fused.py`: packed weights, StaticCache prefill, static
  buffers, `bump_seq`, the 30-layer loop, CUDA graph, and a self-feeding token
  buffer (argmax writes the buffer embed reads on the next replay).
- Added `graph_fused` to `bench/harness.py`.
- **50-token greedy parity at 128 / 512 / 2048** (first diff: none).
- Timings: **1.79 / 1.81 / 2.06 ms** (56.9 / 56.4 / 49.5% roofline).
- Breakdown at 128: 254 kernels; weights 1.68 ms, attention 0.20,
  elementwise 0.09, sampling 0.013, host ~0. At 2048 attention 0.51.
- Added `tests/test_fused_parity.py` (50 steps at 128) and fused breakdown
  categories. The final norm/lm_head/argmax is still the HF/torch path.

### 2026-11-26 — down regression fixed

- Swept `gemv_res` for down (K=1536, N=576): BK=64 -> 0.443 ms, BK=128 ->
  0.314 ms (BN=32, warps=8). Applied BN=32/BK=128/warps=8; P stays 18.
- New medians: **1.70 / 1.73 / 1.99 ms** (60.2 / 59.1 / 51.4%); 50-token
  parity green at all three.
- Fused breakdown: weights 1.591 (now ~= HF 1.595 pure-GEMM total),
  attention 0.196, elementwise 0.089, down 0.321.

---

## 4. Numbers to reuse (with conditions)

All timings: batch 1, greedy, bf16, CUDA graph, cold L2 via read-only 256 MB
flush, median over 20–50 replays, 263.5 GB/s roofline.

| measurement | value | condition |
|---|---:|---|
| Part 1 graph, 128 tok | 3.39 ms (30.1%) | HF path + split-K attention |
| Part 1 breakdown 128 | 1.59 GEMM / 1.33 elementwise / 0.17 attn / 0.01 sample / 0.26 host | 1,372 kernels |
| Feasibility fused chain | 1.29 ms (79.3%) | 269 MB, attention stubbed |
| Feasibility torch chain | 2.32 ms (43.9%) | same reduced op set |
| RoPE max diff | 0.015625 | 2 bf16 ULP vs HF |
| qkv parity | 0.015625 | q/k; v 0.000488 |
| qkv fused norm | 0.312 ms (40.3%) | 30 layers, cold |
| qkv stats + gemv | 0.258 ms (48.8%) | P=1 |
| qkv pure gemv | 0.217 ms (58.0%) | P=1, no stats |
| qkv pure read | 0.209 ms (60.3%) | BN=16, warps=4 |
| qkv pure read best | 0.177 ms (71.1%) | BN=16, warps=8 |
| same-W warm | 0.240 ms | shows fixed per-kernel cost |
| `P=4` partials parity | 0.015625 | producer path validated |
| `gemv_res` y parity | 0.03125 (o), 0.125 (down) | 1 bf16 ULP, random inputs |
| `gemv_res` partial sums | <= 2e-3 | `sum(ss)` vs `sum(y_bf16^2)` |
| fused 128 | 1.70 ms (60.2%) | `graph_fused`, 50 steps |
| fused 512 | 1.73 ms (59.1%) | `graph_fused`, 50 steps |
| fused 2048 | 1.99 ms (51.4%) | fresh process, 50 steps |
| fused kernels | 254/step | vs 1,372 vanilla |
| fused breakdown 128 | weights 1.59, attn 0.20, elementwise 0.09, down 0.321, host ~0 | profiled; CUPTI-inflated |

Roofline arithmetic:

- 269 MB weights → **1.021 ms** floor.
- qkv: 30 × 1.106 MB = 33.2 MB → 0.126 ms floor.
- Fused projection: 1.25–1.30 (weights) + 0.17 (attention) + ~0.05–0.10 (misc)
  ≈ **1.5–1.7 ms**.

Elementwise-cost arithmetic (important for the writeup):

```text
1.33 ms / 1100 kernels ≈ 1.2 µs per elementwise kernel
those kernels move a few KB → ~10 ns of bandwidth each
=> >99% of the elementwise time is fixed per-kernel cost, not bytes
```

---

## 5. Decisions log

| decision | rationale |
|---|---|
| Standalone fused decoder, HF prefill | control packing, buffers, cache, and graph; mirrors Phase 3 |
| Output-row tiling for GEMV | batch 1 has no token parallelism; split the weight matrix |
| BN=16 for qkv, warps=8 | best cold read; 60 CTAs vs 15 at BN=64 |
| No split-K for qkv | 30–60 CTAs already fill 24 SMs; split-K adds a reduction |
| Split-K for attention | q_len=1 gives no query parallelism; parallelize context |
| Norm reduction in producer/stats | per-block redundant reduction stalls the weight stream |
| Keep HF graph baseline untouched | honest comparison; don't break Part 1 |
| Two-stage lm_head+argmax | argmax is a reduction over 49K outputs across 768 CTAs |
| Graph first, tie-break to bf16 | HF returns bf16 logits; first index wins ties |
| Exact 50-token parity gate | Phase 1 precedent; token parity is the correctness bar |
| 135M only for fusion | vLLM already at 93–95% on Qwen; not worth it |
| BK=128/warps=8 for down | 24 -> 12 K iterations; 0.406 -> 0.321 ms |
| Self-feeding token buffer | argmax writes the buffer embed reads next replay; no host copy |

---

## 6. Open issues / next steps

- [x] `gemv_res` with partial-sum epilogue (o_proj, down_proj); parity test added.
- [x] `rms_gate_up_swiglu` with `(ss_ptr, P, PB)` interface; parity test added.
- [x] `bump_seq`; tested in `tests/test_gate_up.py`.
- [x] `bench/fused.py`: weight packing, prefill handoff, static cache, step, graph.
- [x] 50-token parity at 128/512/2048.
- [x] End-to-end timing; after down tuning: 1.70 / 1.73 / 1.99 ms.
- [ ] Fuse the final norm + lm_head + argmax (currently HF norm + torch).
- [ ] `scripts/phase2.py` -> `results/phase2.csv`.
- [x] Down: BK=128/warps=8 (0.406 -> 0.321 ms).
- [ ] Tune o tiles and attention `SPLITS` at 2048.
- [ ] Residual qkv gap: 0.22 ms vs 0.177 ms pure read (load order, tiling, `tl.dot`).
- [ ] Update this notebook at CP3.

---

## 7. Writeup hooks

Candidate sections/anecdotes for the article:

1. **"The 1.2 microsecond kernel."** 1,100 elementwise kernels, 1.33 ms; bytes
   would take ~10 ns. Fusion is about boundaries, not arithmetic.
2. **"Fusing a reduction is a trap."** The norm-in-GEMV experiment: 0.312 vs
   0.212 ms; why the reduction must move to the producer.
3. **"244% of the memory roofline."** How a 33 MB working set hid in a 32 MB L2,
   and why every kernel measurement needs a cold-cache protocol.
4. **"Small GEMVs have a speed limit."** 1.1 MB per layer caps at 60–71% even
   for a pure read; this motivates the Phase 3 persistent/prefetch design.
5. **"A kernel boundary is a reduction boundary."** The rule that explains the
   seven-kernel layer and why gate/up+down and attention+o cannot fuse.
6. **"The same model, four schedules."** eager 20.7 → graph 3.39 → fused ~1.6 →
   persistent (Phase 3).
