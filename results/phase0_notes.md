# Phase 0 notes — environment and sanity

Date: 2026-11-13 (UTC)
Location: WSL2 Ubuntu, native folder `/home/ansh/mega4060`, `uv`-managed venv (`.venv`, Python 3.11.15).

## Status: PASS

| Check | Result |
|---|---|
| GPU visible from WSL2 | PASS — RTX 4060 Laptop GPU, 8188 MiB |
| torch CUDA build works | PASS — torch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2 |
| Triton present | PASS — 3.5.1 |
| Trivial CUDA graph capture + replay | PASS — `before=3.0`, `after=4.0` |
| `torch.profiler` CUDA trace | PASS — `results/phase0_trace.json` contains CUDA kernels |
| Cooperative kernel launch | PASS — grid 144 (6 blocks/SM x 24 SM), 256 threads, grid.sync() OK |
| `nsys` GPU kernel trace | UNAVAILABLE — report generated but "does not contain CUDA kernel data" |
| Memory bandwidth microbench | read-only 187.6 GB/s, copy 182.1 GB/s |

## Hardware (actual, differs from PLAN.md assumptions)

| Item | PLAN.md assumed | Measured |
|---|---|---|
| GPU | RTX 4060 8 GB | **RTX 4060 Laptop GPU**, 8 GB |
| SMs | 24 | 24 |
| SM count / cc | sm_89 | 8.9 |
| L2 | ~24 MB | **32 MB** |
| Power cap | n/a | **38 W** (laptop SKU, power-limited) |
| Driver | n/a | 617.42 (KMD), CUDA UMD 13.4; `nvidia-smi` 615.78.05 |
| CUDA toolkit | 12.x | Toolkit 13.0 installed; torch runtime 12.6 |
| Clocks | n/a | SM 1620 MHz (max 3105), mem 7001 MHz (max 8001) |
| Display active | AMD iGPU expected | Confirmed: 0 MiB used, 0% util, fully free |

Roofline denominator: ~~use ~188 GB/s~~ **superseded in Phase 1: use 263.5 GB/s
sustained read.** The 187.6 GB/s `torch.sum` result here was a cold/short-buffer
artifact; a 512 MB Triton streaming-read loop (Phase 1) boosts the GPU to
9022 MHz / 33 W and measures 263.5 GB/s. See `results/phase1_notes.md`.

## Profiling

- **Primary: `torch.profiler`** (CUDA activity). Works in WSL2; produces a valid
  Chrome trace with kernel names and durations.
- **`nsys` 2025.3.2**: `nsys profile -t cuda` completes and writes `.nsys-rep`, but
  `nsys stats --report cuda_gpu_kern_sum` reports
  `does not contain CUDA kernel data`. This matches the PLAN.md warning about
  partial Nsight support in WSL2. Do not rely on nsys for kernel timelines; use
  `torch.profiler` traces for Phase 1/2/3.

## Cooperative launch caveat

`torch.cuda.get_device_properties(0)` does not expose `cooperative_launch`, so the
`env.json` field reads `false` (getattr default). The CUDA runtime query in
`kernels/cuda/coop_test.cu` reports `cooperativeLaunch: 1`, and a real
`cudaLaunchCooperativeKernel` + `cg::this_grid().sync()` test passed. Phase 3C
(whole-model cooperative kernel) is therefore viable.

## Environment / versions

```
python       3.11.15
torch        2.9.1+cu126
torch cuda   12.6
cudnn        91002
triton       3.5.1
transformers 5.22.2   <-- note: v5 major version, API differs from 4.x
numpy        2.4.6
accelerate   1.15.0
safetensors  0.8.0
pandas       3.1.1
matplotlib   3.11.2
nvidia-ml-py 13.615.71
```

Caveat for Phase 1: `transformers==5.22.2` is a major-version jump. Confirm
`StaticCache` / CUDA-graph APIs before writing `bench/graph.py`; if the v5 API is
awkward, pin `transformers` to a 4.x release and record the change.

## Artifacts

- `results/env.json` — full probe (env + graph + profiler)
- `results/membw.json` — bandwidth microbench
- `results/phase0_trace.json` — Chrome trace
- `results/phase0_nvidia_smi.txt`, `phase0_python.log`, `phase0_coop.log`, `phase0_nsys.log`
- `scripts/phase0.py`, `scripts/phase0.sh`, `bench/membw.py`, `kernels/cuda/coop_test.cu`

## Reproduce

```bash
cd /home/ansh/mega4060
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python transformers safetensors accelerate pandas matplotlib nvidia-ml-py psutil
./scripts/phase0.sh
```

## Next: Phase 1 gate

Run baselines (eager, torch.compile, CUDA graph, vLLM) on SmolLM2-135M then 1B,
measure TPOT and `% of 188 GB/s` roofline. Do not write kernels before that number.
