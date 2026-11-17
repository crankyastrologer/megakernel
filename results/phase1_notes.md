# Phase 1 notes — baselines and the go/no-go number

Date: 2026-11-13
Location: WSL2 `/home/ansh/mega4060`

> **Update (Part 1):** the cross-framework summary in this file uses the **final
> fixed path**. The 135M CUDA-graph and static rows in the Phase-1 tables *below*
> are the *pre-fix* baseline. A context-scaling bug in the static-cache attention
> was fixed with a split-K Triton decode kernel; the 135M graph is now flat at
> **3.39 / 3.43 / 3.72 ms** (128/512/2048). See `part1_notes.md` for the fix,
> the llama.cpp baseline, and the decode breakdown. vLLM/eager/Qwen numbers are
> unaffected.

> **Overclock disclosure:** all numbers here were measured with MSI Afterburner
> offsets of **+1000 MHz memory** and **+158 MHz core**. Stock peak is
> 256 GB/s; the overclocked theoretical peak is ~288.8 GB/s; sustained read is
> 263.5 GB/s. Relative comparisons share the same overclock; absolute roofline
> percentages use 263.5 GB/s. See `part1_notes.md` and the README.

Main venv: Python 3.11.15, torch 2.9.1+cu126, triton 3.5.1, transformers 5.22.2
vLLM venv: `.venv-vllm`, torch 2.14.0+cu130, vllm 0.33.0, transformers 5.20.2

## Cross-framework summary (final numbers)

SmolLM2-135M, bf16/F16, batch-1 greedy: median ms/token and **% of the 263.5 GB/s
weight-read roofline**. Weight bytes 0.26903 GB; Q4 uses 0.10366 GB. llama.cpp is
`llama-bench tg64 @ d{128,512,2048}`; `+FA` = flash attention on (`-fa 1`).

| runtime | 128 | 512 | 2048 |
|---|---|---|---|
| llama.cpp F16 +FA | 2.32 (44.0%) | 2.37 (43.1%) | 2.54 (40.2%) |
| llama.cpp F16 -FA | 2.78 (36.7%) | 2.75 (37.1%) | 2.91 (35.1%) |
| llama.cpp Q4_K_M +FA | 1.75 (22.5%) | 1.70 (23.2%) | 1.88 (20.9%) |
| vLLM (bf16, CUDA graphs) | 2.30 (44.5%) | 2.18 (46.8%) | 2.41 (42.3%) |
| ours: CUDA graph | 3.39 (30.1%) | 3.43 (29.8%) | 3.72 (27.5%) |
| ours: torch.compile | 7.40 (13.8%) | 7.85 (13.0%) | 6.94 (14.7%) |
| ours: eager | 20.71 (4.9%) | 21.65 (4.7%) | 21.84 (4.7%) |
| ours: static (no graph) | 26.57 (3.8%) | 26.80 (3.8%) | 28.46 (3.6%) |

Qwen2.5-1.5B (3.08743 GB weights); llama.cpp was not run on this model:

| runtime | 128 | 512 | 2048 |
|---|---|---|---|
| vLLM (bf16, CUDA graphs) | 12.49 (93.8%) | 12.34 (95.0%) | 12.54 (93.4%) |
| ours: CUDA graph | 14.45 (81.1%) | 15.46 (75.8%) | 18.68 (62.7%) |
| ours: torch.compile | 14.47 (81.0%) | 14.79 (79.2%) | 17.26 (67.9%) |
| ours: eager | 24.81 (47.2%) | 20.77 (56.4%) | 21.48 (54.6%) |
| ours: static (no graph) | 31.27 (37.5%) | 25.30 (46.3%) | 26.56 (44.1%) |

Sources: `results/phase1.csv` (torch/vLLM; graph/static/compile use the fixed path),
`results/llamacpp.csv` (llama.cpp), `results/vllm.json`. The context-scaling fix
and interpretation are in `results/part1_notes.md`.

**Takeaways:** on 135M, vLLM and llama.cpp F16 are tied at ~40–47% of roofline,
while our fixed graph is ~27–30%. The recoverable costs are the 1.33 ms of
elementwise work and 0.57 ms of GEMV inefficiency; the megakernel target is
1.8–2.1 ms. On Qwen, vLLM is at 93–95% (already at the achievable roofline), so
it is the contrast case, not the optimization target.

## Result: Phase 1 PASS

All modes are greedy, same tokenized prompt, same output length (50 measured
decode steps after 10 warmup). vLLM ran with CUDA graphs enabled
(`enforce_eager=False`; engine reported `CUDAGraphMode.FULL_AND_PIECEWISE`).
TPOT is taken from vLLM engine timestamps (`first_token_ts` -> `last_token_ts`),
excluding prefill.

### Roofline denominator: 263.5 GB/s

Phase 0's 187.6 GB/s was a cold/short-buffer artifact. A sustained 512 MB
Triton streaming-read loop boosts the GPU (SM 2100 MHz, mem 9022 MHz, 33 W) and
measures **263.5 GB/s** (theoretical at 9022 MHz: 288.7 GB/s; ~91% efficiency).
Idle clocks are 1620/7001 MHz and are not representative of decode.

### SmolLM2-135M (bf16, 0.269 GB weights)

| prompt | eager ms | static ms | CUDA graph ms | compile ms | vLLM ms | vLLM % roofline |
|---|---|---|---|---|---|---|
| 128 | 20.71 | 25.50 | 4.80 | 8.62 | **2.296** | **44.5%** |
| 512 | 21.65 | 25.94 | 4.30 | 8.82 | **2.184** | **46.8%** |
| 2048 | 21.84 | 26.25 | 6.69 | 8.92 | **2.414** | **42.3%** |

### Qwen2.5-1.5B-Instruct (bf16, 3.087 GB weights)

| prompt | eager ms | static ms | CUDA graph ms | compile ms | vLLM ms | vLLM % roofline |
|---|---|---|---|---|---|---|
| 128 | 24.81 | 31.27 | 14.45 | 14.47 | **12.494** | **93.8%** |
| 512 | 20.77 | 25.30 | 15.46 | 14.79 | **12.336** | **95.0%** |
| 2048 | 21.48 | 26.56 | 18.68 | 17.26 | **12.539** | **93.4%** |

## Go/no-go decision

| Model | vLLM % of roofline | Gate | Decision |
|---|---|---|---|
| SmolLM2-135M | 42-47% | < 70% | **Pursue "faster than vLLM"** |
| Qwen2.5-1.5B | 93-95% | > 85% | Educational/roofline contrast only |

**Headline target: SmolLM2-135M at batch 1.** vLLM is only ~44% of the HBM
roofline there, so a fused/megakernel decode step has room. For the 1.5B model
vLLM is essentially at the achievable roofline; do not chase tok/s on it.

Current gap to close on 135M:

- vLLM: 2.30 ms/token (44.5% roofline)
- our CUDA graph: 4.80 ms/token (21.3% roofline)
- weight-read floor: 0.269 GB / 263.5 GB/s = **1.02 ms/token**

To beat vLLM we need < 2.30 ms; to hit the 1.2-1.5x headline we need
**1.5-1.9 ms/token**, i.e. 2.0-3.2x faster than the current graph and 55-68%
of the roofline. That is the Phase 2/3 target.

## Where the time goes (torch.profiler chrome traces)

Kernels per decode step and the host gap (unprofiled median minus profiled GPU
kernel time):

| model | mode | kernels/step | gap ms/step |
|---|---|---|---|
| SmolLM2-135M | eager | 1222 | 17.2 |
| SmolLM2-135M | static | 1469 | 21.1 |
| SmolLM2-135M | CUDA graph | 1469 | ~0 |
| Qwen2.5-1.5B | eager | 1170 | 12.1 |
| Qwen2.5-1.5B | static | 1373 | 13.8 |
| Qwen2.5-1.5B | CUDA graph | 1373 | ~0 |

Interpretation:

- Eager decode is **CPU/launch-bound on WSL**: ~1.2K kernels/step and 12-21 ms
  of host gap. CUDA graphs remove almost all of it (4.3x faster on 135M).
- Even under a graph, ~1.5K kernels/step remain. At 135M those kernels only
  reach 21% of roofline, because the per-kernel work is tiny and the launch/
  tail effects dominate. This is exactly the fusion opportunity.
- Profiled GPU kernel times are inflated under WSL/CUPTI (e.g. Qwen graph shows
  23.6 ms kernel time vs 14.5 ms measured wall). Use the CUDA-event timings for
  latency and the trace only for kernel counts/ordering.

## Correctness notes

- `tests/test_token_parity.py` passes for eager, static, CUDA graph, and
  compile on SmolLM2-135M.
- **HF `model.generate` diverges from manual decode for Qwen2.5-1.5B at
  prompt 128, token 9.** Manual eager, StaticCache + CUDA graph, and vLLM agree
  exactly with each other (pairwise first-diff = None). The parity reference in
  `scripts/phase1.py` is therefore manual eager; the `generate` divergence is
  recorded in `phase1_meta.json`.
- vLLM token ids match the torch reference for both models at all prompt
  lengths (after switching to the manual-eager reference).

## Engineering caveats

- **`torch.compile(mode="reduce-overhead")` segfaults** (exit 139) on this
  stack: transformers v5 static-cache in-place mutation makes Inductor skip
  cudagraphs ("mutated inputs"), then the compiled run crashes. The `compile`
  rows use `mode="default"`. `reduce-overhead` is a documented negative result.
- **Model substitution:** Llama-3.2-1B is gated (HTTP 401) and TinyLlama's
  context is only 2048 tokens (cannot decode after a 2048-token prompt), so the
  1B-class model is `Qwen/Qwen2.5-1.5B-Instruct` (32K context, ungated).
- vLLM needs its own venv (torch 2.14/cu130). Run via
  `scripts/run_vllm_bench.sh`, which sets `CUDA_HOME`, puts the venv `ninja` on
  PATH, and sets `VLLM_USE_FLASHINFER_SAMPLER=0` (avoids FlashInfer JIT while
  leaving greedy output unchanged).
- nsys still has no CUDA kernel data in WSL (Phase 0); torch.profiler traces
  are the primary source.

## Artifacts

- `results/phase1.csv` — 30 rows (2 models x 3 prompts x 5 modes), parity flags
- `results/phase1_profile.csv` — kernels/step and gap at prompt 128
- `results/phase1_meta.json` — versions, clocks, per-model configs, generate divergence
- `results/vllm.json` — vLLM TPOT, token ids, prompt hashes
- `results/trace_<model>_<mode>.json` — chrome traces
- `results/phase1_<model>_latency.png`, `results/phase1_<model>_roofline.png`

## Reproduce

```bash
cd /home/ansh/mega4060
./scripts/install_vllm.sh          # once; separate vLLM venv
./scripts/phase1.sh                # vLLM + torch modes + parity gate
# if a long run is interrupted, resume without losing rows:
.venv/bin/python scripts/phase1.py --models qwen2.5-1.5b \
  --prompt-lengths 128,512,2048 --modes eager,static,graph --profile --resume
```

## Next: Phase 2

Target < 2.3 ms/token on SmolLM2-135M. Start with Triton fusion
(RMSNorm+QKV epilogue, RoPE+KV append, RMSNorm+gate/up+SwiGLU+down, final
norm+lm_head+argmax) and compare every step against the Phase 1 CUDA-graph
baseline. Do not revisit the 1.5B model for tok/s; it is already ~95% of
roofline under vLLM.
