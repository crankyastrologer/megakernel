"""Kernel-count / gap measurement via torch.profiler chrome trace."""

import json
import os
import time

import torch


def profile_steps(step_fn, n_steps, trace_path, warmup_steps=2):
    """Profile decode steps and summarize GPU kernels and host gap.

    Runs ``warmup_steps`` untimed calls, then profiles ``n_steps`` calls with
    ``torch.profiler``, exports a chrome trace, and parses kernel/memcpy events
    from it.

    Args:
        step_fn (Callable[[], object]): One decode step (advances the cache).
        n_steps (int): Number of profiled steps.
        trace_path (str): Output path for the chrome trace JSON.
        warmup_steps (int): Untimed steps before profiling. Defaults to 2.

    Returns:
        dict: ``steps``, ``profiled_wall_ms_per_step``, ``kernels_per_step``,
        ``kernel_ms_per_step``, ``memcpy_ms_per_step``,
        ``profiler_gap_ms_per_step``, and ``trace``.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup_steps):
        step_fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        wall0 = time.perf_counter()
        for _ in range(n_steps):
            step_fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall0

    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    prof.export_chrome_trace(trace_path)

    with open(trace_path) as f:
        events = json.load(f).get("traceEvents", [])

    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    memcpys = [e for e in events if e.get("cat") in ("gpu_memcpy", "Memcpy") and e.get("ph") == "X"]
    kernel_us = sum(float(e.get("dur", 0.0)) for e in kernels)
    memcpy_us = sum(float(e.get("dur", 0.0)) for e in memcpys)
    wall_us = wall * 1e6

    return {
        "steps": n_steps,
        "profiled_wall_ms_per_step": wall_us / n_steps / 1e3,
        "kernels_per_step": len(kernels) / n_steps,
        "kernel_ms_per_step": kernel_us / n_steps / 1e3,
        "memcpy_ms_per_step": memcpy_us / n_steps / 1e3,
        "profiler_gap_ms_per_step": (wall_us - kernel_us - memcpy_us) / n_steps / 1e3,
        "trace": trace_path,
    }
