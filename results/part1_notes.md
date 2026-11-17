# Part 1 notes — llama.cpp baseline, context-scaling fix, decode breakdown

Date: 2026-11-13/14
Project: `/home/ansh/mega4060`
Model: SmolLM2-135M-Instruct (bf16 torch, F16/Q4_K_M GGUF)
Roofline denominator: **263.5 GB/s** sustained read (see `phase1_notes.md`)

> **Overclock disclosure (recorded after the runs):** all measurements used MSI
> Afterburner offsets of **+1000 MHz memory** and **+158 MHz core**. Stock
> memory peak for this unit is **256 GB/s**; the overclocked theoretical peak is
> **~288.8 GB/s**, and the measured sustained read is **263.5 GB/s (91.2%)**.
> Every runtime shared the same overclock, so relative comparisons are
> unaffected. Absolute roofline percentages use 263.5 GB/s; multiply by 0.912 to
> express them against the 288.8 GB/s theoretical peak.

This is the Part 1 package: external baseline (llama.cpp), the CUDA-graph
context-scaling bug + fix, and a component breakdown of the 4.8 ms decode step
that defines the megakernel target.

---

## 1. llama.cpp baseline

Built from source (`~/llama.cpp`, commit `b7ad48ebd`) with CUDA 13.0 / sm_89.
Models from `unsloth/SmolLM2-135M-Instruct-GGUF`:

- `SmolLM2-135M-Instruct-F16.gguf` (259 MB)
- `SmolLM2-135M-Instruct-Q4_K_M.gguf` (101 MB)

Benchmark tool: `llama-bench`, `tg64 @ d{128,512,2048}` (decode 64 tokens at KV
depth d), 5 reps, CUDA backend, all layers on GPU. Raw output:
`results/llamacpp_f16.md`, `results/llamacpp_q4.md`.

### F16, flash attention on (`-fa 1`) — the fair comparison

| depth | t/s | ms/token | GB/s | % of 263.5 |
|---|---|---|---|---|
| 128 | 430.60 | 2.322 | 115.8 | **44.0%** |
| 512 | 422.53 | 2.367 | 113.7 | **43.1%** |
| 2048 | 393.27 | 2.543 | 105.8 | **40.2%** |

Without flash attention (`-fa 0`): 2.782 / 2.750 / 2.912 ms (35-37% roofline).

### Q4_K_M, flash attention on

| depth | t/s | ms/token | GB/s | % of 263.5 |
|---|---|---|---|---|
| 128 | 571.52 | 1.750 | 59.2 | **22.5%** |
| 512 | 588.69 | 1.699 | 61.0 | **23.2%** |
| 2048 | 530.84 | 1.884 | 55.0 | **20.9%** |

(Using the 0.1037 GB GGUF file size; weight bytes are ~0.100 GB.)

Without flash attention: 2.339 / 2.039 / 2.138 ms.

### Framing decision

llama.cpp F16 (FA) lands at **40-44% of the HBM roofline**, essentially tied
with vLLM (42-47%) and far below the 60% threshold. This is the
**"nobody is close on small models"** outcome: the megakernel is the main story,
not a small edge over one runtime. The Q4 case is even further below roofline
(~21%), so fusion has more headroom there; Part 1 keeps the bf16/F16 model.

Combined 135M picture at context 128 (median ms/token):

| runtime | ms/token | % roofline |
|---|---|---|
| weight-read floor | 1.02 | 100% |
| llama.cpp F16 FA | 2.32 | 44.0% |
| vLLM (bf16, CUDA graphs) | 2.30 | 44.5% |
| our CUDA graph, fixed | **3.39** | **30.1%** |
| our CUDA graph, before fix | 4.80 | 23.6% |

---

## 2. Context-scaling fix in the CUDA graph

### Symptom

Our graph path was not flat: 4.80 ms at 128 → 6.69 ms at 2048, while vLLM
(2.30 → 2.41) and llama.cpp (2.32 → 2.54) are nearly flat. The breakdown showed
attention growing from 0.33 ms to **2.99 ms** over that range.

### Cause

With transformers v5 `StaticCache`, the model builds a 4D causal mask and passes
it to `scaled_dot_product_attention`. PyTorch then selects the **mem-efficient
CUTLASS** attention kernel, which is ~4x slower than the flash backend for a
single-query decode and scales poorly with KV length (microbench at L=2048:
142 µs vs 37 µs per layer). A constant-key sentinel trick to drop the mask does
not work: a negative sentinel key gives a **positive** score when `sum(q) < 0`.

### Fix

A split-K Triton decode-attention kernel (`bench/fastattn.py`) patched into
`LlamaAttention.forward` for the `q_len==1 + StaticCache` case:

- 16 KV splits per query head, online softmax, then a tiny reduction kernel.
- Reads the current cache length from the device `cumulative_length` scalar, so
  it is CUDA-graph-safe (fixed shapes, runtime length).
- Prefill keeps the original masked path (causality is required there).
- Token parity against manual eager: **exact** at all prompt lengths.

### Result

| prompt | before fix | after fix | growth 128→2048 |
|---|---|---|---|
| 128 | 4.80 | **3.39** | — |
| 512 | 4.30 | **3.43** | +0.04 |
| 2048 | 6.69 | **3.72** | **+0.33** |

Compared with llama.cpp (+0.22 ms) and vLLM (+0.11 ms), the path is now flat to
within ~0.2 ms of llama.cpp over a 16x context change. Attention at 2048 fell
from 2.99 ms to **0.50 ms**.

Measurement note: timings now include a 0.3 s GPU burn before the timed loop
(`harness.burn`) so the GPU is at steady clocks. Without it, the short-128
prompt runs at lower clocks and looks slower than 2048.

---

## 3. Where the 4.8 ms went (and where the 3.4 ms goes)

Chrome-trace kernel categories, CUDA graph, prompt 128 (median ms/step):

| component | before fix | after fix |
|---|---|---|
| weight streaming (GEMM/GEMV) | 1.62 | 1.59 |
| elementwise/other (1.1K kernels) | 1.68 | 1.33 |
| attention | 0.33 | 0.17 |
| sampling (argmax) | 0.012 | 0.011 |
| host gap (launch/sync) | 1.09 | 0.26 |
| **total (wall)** | **4.76** | **3.36** |
| weight-read floor | 1.02 | 1.02 |

At 2048 (after fix): GEMM 1.92, elementwise 1.41, attention 0.50, sampling 0.012,
host ~0, total 3.74 ms.

### Reading the breakdown

- **Weight streaming is close-ish but not at roofline**: 1.59 ms vs a 1.02 ms
  floor at 128 (64% efficient). Batch-1 GEMV cannot saturate HBM by itself.
- **Elementwise is the largest recoverable chunk**: 1.33 ms across ~1,100
  kernels (RMSNorm, RoPE, SwiGLU, residual adds, copies, cats, KV index_copy).
  Fusing these into GEMM epilogues and the KV append removes both their work and
  their launch/tail overhead.
- **Attention is now small** (0.17 ms at 128, 0.50 ms at 2048).
- **Sampling is negligible** (argmax, 0.011 ms). Do not build a special path for it.
- **Host gap is ~0** under the graph; CUDA graphs already solved launch overhead.
  The remaining cost is GPU work, not Python.

### Megakernel target (from this breakdown)

At 128, a perfect megakernel would pay ~1.02 ms for weights and a little more for
attention/host. A realistic target is **1.8-2.1 ms**:

- remove most of the 1.33 ms elementwise by fusing norms/RoPE/SwiGLU/residual/
  argmax into the GEMM epilogues;
- keep weight streaming near the floor (1.0-1.2 ms);
- keep the fixed attention (~0.2-0.5 ms).

That would be **1.1-1.3x faster than vLLM/llama.cpp** (2.3-2.5 ms) and
**1.7-2.1x faster than the current graph**, at 50-58% of roofline. The stretch
1.5x headline (1.53 ms) needs the GEMM path itself closer to the 1.02 ms floor.

---

## 4. Artifacts

- `results/llamacpp_f16.md`, `results/llamacpp_q4.md` — raw llama-bench tables
- `results/phase1.csv` — torch/vLLM rows (graph/static now use the fixed path)
- `results/breakdown_graph_128.json`, `results/breakdown_graph_2048.json`
- `results/trace_breakdown_graph_{128,2048}.json` — chrome traces
- `bench/fastattn.py` — split-K Triton decode attention + patch
- `bench/breakdown.py` — kernel-category profiler
- `scripts/build_llamacpp.sh` — reproducible llama.cpp CUDA build

## 5. Reproduce

```bash
# llama.cpp
bash scripts/build_llamacpp.sh
cd ~/llama.cpp
./build/bin/llama-bench -m /home/ansh/mega4060/models/gguf/SmolLM2-135M-Instruct-F16.gguf \
  -p 0 -n 64 -d 128,512,2048 -r 5 -fa 1 -o md

# torch graph with the fix
cd /home/ansh/mega4060
.venv/bin/python scripts/phase1.py --models smollm2-135m \
  --prompt-lengths 128,512,2048 --modes graph --resume
.venv/bin/python bench/breakdown.py --mode graph --prompt-len 128
```

## 6. Caveats

- llama.cpp was built at commit `b7ad48ebd` with CUDA 13.0; `-fa 1` is the fair
  comparison to vLLM. llama.cpp's default greedy path is close but not
  token-identical to the torch reference (different quantized/F16 kernels).
- The Triton attention is a correctness-first implementation. It is already flat,
  but a production megakernel should fold it into the persistent kernel and tune
  splits/block sizes.
- StaticCache non-graph ("static") is still launch-bound (~27 ms) and is not a
  meaningful baseline; use eager or the CUDA graph.
