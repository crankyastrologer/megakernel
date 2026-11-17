"""Measure achievable HBM read bandwidth.

A torch.sum reduction underestimates peak read BW on this GPU; this uses a
Triton streaming-read kernel (one partial sum per block) plus copy/sum references.
"""

import json
import os
import subprocess
import time

import torch
import triton
import triton.language as tl

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@triton.jit
def _read_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Stream-read ``BLOCK`` elements and store their block sum.

    Args:
        x_ptr (tl.pointer): Input vector (bf16/fp16/fp32).
        out_ptr (tl.pointer): One ``float32`` partial sum per program.
        n_elements (int): Number of valid input elements.
        BLOCK (tl.constexpr): Elements handled per program.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    acc = tl.sum(x.to(tl.float32))
    tl.store(out_ptr + pid, acc)


def timed(fn, iters=50, warmup=10):
    """Time ``fn`` on the GPU and return the median wall time.

    Args:
        fn (Callable[[], None]): GPU work to time; must enqueue on the current stream.
        iters (int): Timed iterations. Defaults to 50.
        warmup (int): Untimed iterations. Defaults to 10.

    Returns:
        float: Median seconds per call.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def triton_read_gbs(x, block=8192, warps=8):
    """Measure streaming-read bandwidth with the Triton kernel.

    Args:
        x (torch.Tensor): Flat, contiguous input tensor on CUDA.
        block (int): Elements per Triton program. Defaults to 8192.
        warps (int): Warps per program. Defaults to 8.

    Returns:
        tuple[float, int, int]: ``(GB/s, block, warps)``.
    """
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    out = torch.empty(grid[0], device=x.device, dtype=torch.float32)

    def run():
        """Launch one streaming-read pass."""
        _read_kernel[grid](x, out, n, BLOCK=block, num_warps=warps)

    t = timed(run)
    return n * x.element_size() / t / 1e9, block, warps


def clocks():
    """Query current SM/memory clocks, power and temperature via nvidia-smi.

    Returns:
        str: Comma-separated ``clocks.sm, clocks.mem, power.draw, temperature.gpu``,
        or an ``unavailable: ...`` string if nvidia-smi fails.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu", "--format=csv,noheader"],
            text=True,
        )
        return out.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    """Benchmark copy/sum/Triton reads and write ``results/membw.json``."""
    os.makedirs(RESULTS, exist_ok=True)
    clocks_before = clocks()
    print("clocks before:", clocks_before, flush=True)
    n = 256 * 1024 * 1024  # 256M bf16 = 512 MB (> L2 by 16x)
    dtype = torch.bfloat16
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.empty_like(x)

    t_copy = timed(lambda: y.copy_(x))
    copy_bytes = 2 * n * x.element_size()
    copy_gbs = copy_bytes / t_copy / 1e9

    t_sum = timed(lambda: torch.sum(x, dtype=torch.float32))
    sum_bytes = n * x.element_size()
    sum_gbs = sum_bytes / t_sum / 1e9

    candidates = []
    for block, warps in [(4096, 8), (8192, 8), (16384, 8), (8192, 4), (32768, 8), (65536, 8)]:
        gbs, b, w = triton_read_gbs(x, block, warps)
        print(f"triton read BLOCK={b} warps={w}: {gbs:.2f} GB/s", flush=True)
        candidates.append(gbs)
    peak = max(candidates)

    clk_after = clocks()
    mem_clock = None
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.mem", "--format=csv,noheader,nounits"], text=True
        ).strip()
        mem_clock = float(q)
    except Exception:
        pass
    theoretical = mem_clock * 0.032 if mem_clock else None

    out = {
        "buffer_mib": n * x.element_size() // 2**20,
        "copy_ms": round(t_copy * 1e3, 4),
        "copy_gbs": round(copy_gbs, 2),
        "sum_ms": round(t_sum * 1e3, 4),
        "sum_gbs": round(sum_gbs, 2),
        "triton_read_variants_gbs": [round(c, 2) for c in candidates],
        "read_gbs": round(peak, 2),
        "mem_clock_mhz": mem_clock,
        "theoretical_gbs_at_mem_clock": round(theoretical, 2) if theoretical else None,
        "clocks_before": clocks_before,
        "clocks_after": clk_after,
        "note": "read_gbs = best sustained Triton streaming-read result; roofline denominator",
    }
    path = os.path.join(RESULTS, "membw.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
