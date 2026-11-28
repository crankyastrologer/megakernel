# HF vs fused Triton decode — kernel comparison

**Model:** SmolLM2-135M-Instruct, bf16, batch 1, greedy.
**Hardware:** RTX 4060 Laptop, 24 SMs, 32 MB L2, WSL2.
**Roofline:** 263.5 GB/s sustained read (Part 1), 1.021 ms weight floor.
**Prompt:** 128 tokens unless stated. Decode-only; prefill excluded.

Verification and benchmark protocol: `results/phase2_verification.md`.
Batch scaling: `results/batch_sweep.md`.
Figures: `results/figures_phase2.md`.

This document compares the vanilla HF CUDA-graph decode step with the fused
Triton step in `bench/fused.py`. It lists the old kernels, the new kernels,
the measured time for each, and the steps that produced the reduction.

> **How to read the numbers.** GEMM and category times are CUDA kernel durations
> from `torch.profiler` chrome traces; CUPTI inflates them by ~10% under WSL.
> Wall times are CUDA-event timings. Use kernel times for *attribution* and wall
> times for the headline. The HF per-projection split is exact: 210 `gemvx`
> kernels per step are assigned by the known q/k/v/o/gate/up/down order. The
> fused per-kernel split is exact by kernel name; `gemv_res` is split into o and
> down by its alternating position in the layer.

---

## 1. Headline

| metric | vanilla HF graph | fused Triton | change |
|---|---:|---:|---:|
| wall ms/token | **3.36** | **1.70** | **-1.66 ms (1.98×)** |
| % of 263.5 GB/s | 30.1% | 60.2% | +30.1 pts |
| kernels/step | **1,372** | **254** | -1,118 |
| weight/GEMM kernels | 211 | 121 | -90 |
| elementwise kernels | 1,100 | 71 | -1,029 |
| attention kernels | 60 | 60 | 0 |
| sampling kernels | 1 | 1 | 0 |
| host gap | 0.26 ms | ~0 | -0.26 |

At 512 and 2048 tokens the fused path measures **1.73 ms** and **1.99 ms**
(59.1% and 51.4%). 50-token greedy parity is exact at all three lengths.

For reference: vLLM 2.30 / 2.18 / 2.41 ms; llama.cpp F16+FA
2.32 / 2.37 / 2.54 ms. The fused path is 1.35× faster than vLLM at 128,
1.26× at 512, and 1.21× at 2048.

---

## 2. The vanilla HF decode step

### Per-layer pipeline (~37–45 kernels)

```text
x
├─ RMSNorm: pow -> mean -> rsqrt -> add(eps) -> mul -> cast -> mul(gamma)   ~6
├─ q_proj -> view/transpose/cast                                          1 GEMV
├─ k_proj -> view/transpose/cast                                          1 GEMV
├─ v_proj -> view/transpose/cast                                          1 GEMV
├─ RoPE: neg -> cat -> mul -> mul -> add                                   ~5
├─ KV append: arange -> add -> add_ -> index_copy(k) -> index_copy(v)      ~5
├─ attention: split-K + reduce (already patched)                           2
├─ o_proj -> reshape -> residual add                                       1 GEMV + ~2
├─ RMSNorm (again)                                                        ~6
├─ gate_proj                                                              1 GEMV
├─ up_proj                                                                1 GEMV
├─ SiLU -> mul                                                            ~2
├─ down_proj -> residual add                                              1 GEMV + ~1
└─ (once per step) final RMSNorm -> lm_head -> argmax                      ~8 + 1 GEMV
```

Every intermediate is materialized. Most of the 1,100 elementwise kernels move
only a few KB, so their cost is per-kernel fixed cost, not bandwidth:
`1.33 ms / 1,100 ≈ 1.2 µs per kernel`.

### Top HF kernels (128 tokens, per step)

| kernel | role | kernels | ms/step |
|---|---|---:|---:|
| `gemvx::kernel` | all 7 projections | 210 | **1.339** |
| `BinaryFunctor` elementwise | mul/add (RoPE, norm, SiLU) | 120 | 0.164 |
| `CUDAFunctor_add` | residual + RoPE adds | 120 | 0.124 |
| `direct_copy` | copies/casts | 62 | 0.119 |
| `reduce MeanOps` | RMSNorm mean | 61 | 0.103 |
| `vectorized BinaryFunctor` | muls (norm gamma, SiLU·up, RoPE) | 91 | 0.103 |
| `bfloat16_copy` | dtype casts / copies | 93 | 0.094 |
| `CatArrayBatchedCopy` | `rotate_half` concat | 60 | 0.083 |
| `index_copy` | KV cache append | 60 | 0.082 |
| `neg` | `rotate_half` negation | 60 | 0.072 |
| `float BinaryFunctor` | fp32 mul/add | 62 | 0.070 |
| `pow` | RMSNorm | 61 | 0.062 |
| `rsqrt` | RMSNorm | 61 | 0.061 |
| `add float` | RMSNorm eps | 61 | 0.057 |
| `silu` | SwiGLU | 30 | 0.039 |
| `add long` / `add long self` | cache position / length | 61 | 0.065 |
| `arange` | cache position | 31 | 0.025 |
| `cutlass ... gemm` | lm_head | 1 | **0.256** |
| `argmax` | sampling | 1 | 0.011 |

The elementwise total is **1.330 ms across 1,100 kernels**. The same work in the
fused path is **0.089 ms across 71 kernels**.

### HF per-projection time (exact, from the trace)

| stage | ms/step | note |
|---|---:|---|
| q_proj | 0.151 | |
| k_proj | 0.080 | GQA: 192 outputs |
| v_proj | 0.081 | GQA: 192 outputs |
| **qkv total** | **0.312** | 3 kernels |
| o_proj | 0.152 | |
| gate_proj | 0.299 | |
| up_proj | 0.298 | |
| **gate+up** | **0.597** | 2 kernels |
| down_proj | 0.278 | |
| lm_head | 0.256 | cutlass, different kernel |
| **GEMM total** | **1.595** | 211 kernels |

---

## 3. The fused decode step

### Per-layer pipeline (7 kernels)

```text
x
├─ qkv_gemv: norm scaling + packed [Wq; Wk; Wv] GEMV -> [960]          1
├─ rope_kv_append: rotate q/k, write k/v cache                         1
├─ split-K attention + reduce                                          2
├─ gemv_res: o_proj + residual + next-norm partial                    1
├─ rms_gate_up_swiglu: norm scaling + gate/up GEMVs + SiLU·up          1
└─ gemv_res: down_proj + residual + next-norm partial                 1
```

Plus `embed_lookup`, `bump_seq`, `norm_stats` (layer 0), and the final
norm + lm_head + argmax once per token.

### Fused kernel time (128 tokens, per step)

| kernel | kernels | ms/step | role |
|---|---:|---:|---|
| `rms_gate_up_swiglu` | 30 | **0.5853** | gate+up GEMV, SiLU·up, norm scaling |
| `gemv_res` (o) | 30 | 0.1736 | o_proj + residual + partials |
| `gemv_res` (down) | 30 | 0.3209 | down_proj + residual + partials |
| `qkv_gemv` | 30 | 0.2193 | packed qkv, norm scaling |
| `_decode_attn_split` | 30 | 0.1548 | split-K attention |
| `_decode_attn_reduce` | 30 | 0.0407 | combine splits |
| `rope_kv_append` | 30 | 0.0382 | RoPE + KV append |
| `attn_f32 -> bf16` cast | 30 | 0.0363 | torch cast (next: fuse) |
| `cutlass` lm_head | 1 | 0.2913 | torch F.linear |
| `argmax` | 1 | 0.0133 | torch |
| head RMSNorm | ~6 | ~0.008 | torch |
| `embed_lookup` + `bump_seq` + `norm_stats` | 3 | 0.0040 | |

Kernel-time total ~1.89 ms (profiled); wall 1.69–1.70 ms. The gap is CUPTI
inflation.

### Category comparison (exact)

| category | HF graph | fused | delta |
|---|---:|---:|---:|
| weights (GEMM/GEMV) | 1.5946 | 1.5906 | -0.004 |
| elementwise/other | 1.3297 | 0.0888 | **-1.241** |
| attention | 0.1684 | 0.1955 | +0.027 |
| sampling | 0.0114 | 0.0133 | +0.002 |
| memcpy | — | 0.0012 | +0.001 |
| host gap | 0.2563 | ~0 | **-0.256** |
| **wall** | **3.3603** | **1.6947** | **-1.666** |

The weights category is now essentially equal to HF's pure-GEMM total, even
though the fused kernels include residual, SiLU, and partial-sum epilogues.
The elementwise category and host gap are where the 1.66 ms came from.

---

## 4. Stage-by-stage comparison

GEMM times are exact; elementwise allocations are approximate because HF spreads
one logical stage across several kernels.

| stage | HF | fused | what happened |
|---|---:|---:|---|
| input RMSNorm | ~0.19 | ~0.001 | reduction moved to producer/stats; scaling fused into qkv |
| qkv projection | 0.312 | 0.219 | 3 GEMVs -> 1 packed GEMV (30 CTAs); -0.093 |
| RoPE + KV append | ~0.30 | 0.038 | one 15-program kernel; -0.26 |
| attention | 0.168 | 0.195 | same split-K kernels; difference is profiler noise |
| o_proj + residual | ~0.24 | 0.174 | residual in the GEMV epilogue |
| post-attention RMSNorm | ~0.19 | ~0.001 | reduction from o+res partials |
| gate/up + SiLU | ~0.88 | 0.585 | norm + both GEMVs + SiLU·up in one kernel |
| down_proj + residual | ~0.34 | 0.321 | BK=128 + 8 warps; now ≈ HF |
| final norm + lm_head + argmax | ~0.29 | ~0.31 | still the torch path |
| embedding | 0.002 | 0.002 | gather unchanged |
| host gap | 0.256 | ~0 | graph + no host sync |
| **total** | **3.36** | **1.70** | |

---

## 5. What we changed, in order

1. **Start from the CUDA graph, not eager.** The graph already removed the
   17 ms host gap in Part 1. Phase 2 attacks the 1.33 ms of elementwise kernels.

2. **Pack q/k/v into one weight matrix.** Three 9/3/3-tile GEMVs become one
   30-CTA GEMV that reads `x` once. qkv: 0.312 -> 0.218 ms.

3. **Fuse the norm scaling, but not the reduction.** This was the biggest
   surprise. Fusing the 576-element sum-of-squares into every QKV block made
   each block wait before its weight stream started:
   - fused norm: 0.312 ms over 30 layers
   - pure GEMV: 0.212 ms
   - 1-CTA stats + GEMV: 0.258 ms
   - producer partials + GEMV: ~0.22 ms projected
   The final design has `gemv_res` write one partial per output block; the next
   GEMV sums ~18 floats and starts streaming immediately.

4. **Fuse RoPE and KV append into one kernel.** 9 q heads + 3 k heads + 3 v
   heads = 15 programs. q is rotated in place; k is rotated directly into the
   cache; v is copied. ~0.30 -> 0.038 ms.

5. **Keep the split-K attention.** It was already flat and parity-tested
   (0.17 ms at 128, 0.50 ms at 2048). No change.

6. **Put the residual in the GEMV epilogue.** `gemv_res` computes
   `y = res + W x` in place. o+res: ~0.24 -> 0.174 ms.

7. **Fuse gate/up + SiLU·up.** One kernel computes both GEMVs for the same
   output rows, then `silu(g)*u` in registers. The 3.54 MB of weights dominate;
   the norm scaling and activation are free. Gate/up stage: ~0.88 -> 0.585 ms.

8. **Fuse down + residual.** One kernel, in place. It includes the next-norm
   partials. The first version (BK=64, 4 warps) regressed: 0.406 vs 0.278
   (GEMM) + ~0.06 (add). With BK=128 and 8 warps it is 0.321 ms, roughly the
   HF stage total. The lesson was the K-loop tile, not the fusion.

9. **Make the graph self-feeding.** The final argmax writes the same token
   buffer that `embed_lookup` reads on the next replay. No host copy, no
   `.item()`, no sync. Host gap ~0.

10. **Collapse the elementwise kernels.** 1,100 -> 71 kernels; 1.33 -> 0.089 ms.
    The remaining 71 are 30 rope kernels, 30 attention-output casts, the head
    norm, and three one-off kernels.

---

## 6. What regressed, and what we rejected

| attempt | result | lesson |
|---|---|---|
| Fuse RMSNorm reduction into every QKV block | 0.312 vs 0.212 ms pure GEMV | a reduction fused into every consumer serializes the stream |
| Fuse down+residual with BK=64 | 0.406 vs ~0.34 HF stage | fixed: BK=128 + 8 warps -> 0.321; sweep BK for large-K GEMVs |
| One block per token for GEMV | 1 SM reads 1.1 MB while 23 idle | decode parallelism is over the weight matrix, not tokens |
| Single-layer microbenchmarks | 0.516 ms qkv = 244% of roofline | 33 MB working set sits in 32 MB L2; flush and measure 30 layers |
| Eager fused timing | 0.42–0.52 ms qkv | 30 Python launches ~10 µs each; measure graph replay |
| Fuse gate/up with down | not attempted | down needs all 1,536 intermediates; reduction boundary |
| Fuse attention with o_proj | not attempted | o_proj needs all 9 heads; reduction boundary |

---

## 7. What is left

At 128 tokens, the fused step is **1.70 ms**:

```text
weights (profiled)     1.591
attention              0.195
elementwise            0.089
sampling               0.014
host                   ~0
--------------------------------
kernel sum (profiled)  1.889
wall                   1.69-1.70
```

The weight floor is **1.021 ms**, so weight streaming is the remaining lever:

- `down+res` 0.321 ms (was 0.406; now ≈ HF)
- `gate/up` 0.585 ms (73% of its 0.403 floor)
- `qkv` 0.218 ms (vs 0.177 pure-read ceiling)
- `lm_head` 0.291 ms (torch F.linear; fuse argmax later)
- attention at 2048 grows 0.195 -> 0.505 ms; tune `SPLITS`/`BLOCK_N`

Phase 3's persistent/prefetch kernel targets the gap between the 1.675 ms
measured and the 1.021 ms floor. The remaining 0.09 ms elementwise is no longer
the story.

---

## 8. Reproduction

```bash
cd /home/ansh/mega4060

# 50-token parity (fused vs manual eager)
.venv/bin/python tests/test_fused_parity.py

# breakdown at 128 and 2048
.venv/bin/python bench/breakdown.py --mode graph_fused --prompt-len 128
.venv/bin/python bench/breakdown.py --mode graph_fused --prompt-len 2048

# HF baseline breakdown (for the comparison)
.venv/bin/python bench/breakdown.py --mode graph --prompt-len 128
```

Trace files:

- `results/trace_breakdown_graph_128.json` (HF)
- `results/trace_breakdown_graph_fused_128.json` (fused)
- `results/breakdown_graph_128.json`
- `results/breakdown_graph_fused_128.json`

---

## 9. One-paragraph summary for the article

> The vanilla HF decode step spends 1.33 ms of a 3.36 ms token on roughly 1,100
> tiny elementwise kernels, and another 1.59 ms on GEMMs that never see a full
> memory pipeline. The fused Triton step keeps the same arithmetic but changes
> the boundaries: q/k/v become one packed GEMV, RoPE and KV append become one
> 15-program kernel, the MLP becomes two kernels with the activation and
> residual in their epilogues, and the RMSNorm reduction is computed once in the
> producer. The result is 254 kernels instead of 1,372, elementwise time of
> 0.09 ms instead of 1.33 ms, and 1.70 ms/token instead of 3.36 — a 1.98×
> speedup that beats vLLM by 1.35× at the same context.
