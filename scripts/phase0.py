"""Phase 0: environment probe, CUDA graph smoke test, profiler trace.

Writes results/env.json and results/phase0_trace.json, prints a JSON summary.
"""

import json
import os
import platform
import subprocess
import sys
import time

import torch


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def ver(mod):
    """Return a module's ``__version__`` or ``"unknown"``.

    Args:
        mod (module): Imported module.

    Returns:
        str: Version string or ``"unknown"``.
    """
    try:
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def pkg_version(name):
    """Return an installed distribution's version or a placeholder.

    Args:
        name (str): Distribution name, e.g. ``"numpy"``.

    Returns:
        str: Version string or ``"not installed"``.
    """
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not installed"


def cuda_toolkit_version():
    """Return the ``nvcc --version`` release line, if nvcc is available.

    Returns:
        str: Toolkit release line, or an ``nvcc unavailable: ...`` string.
    """
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if "release" in line:
                return line.strip()
    except Exception as exc:
        return f"nvcc unavailable: {exc}"
    return "unknown"


def nvml_info():
    """Collect driver, memory, power and clock information via NVML.

    Returns:
        dict: NVML fields such as ``driver_version``, ``memory_total_mib``,
        ``power_limit_w``, ``clocks_*`` and ``max_clocks_*``. Unsupported
        fields are recorded as ``"unavailable: ..."``; a top-level ``error``
        is set if NVML cannot be initialized.
    """
    info = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        info["nvml_name"] = pynvml.nvmlDeviceGetName(h)
        info["memory_total_mib"] = pynvml.nvmlDeviceGetMemoryInfo(h).total // 2**20
        info["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        try:
            info["power_draw_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception as exc:
            info["power_draw_w"] = f"unavailable: {exc}"
        try:
            info["clocks_graphics_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
            info["clocks_sm_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            info["clocks_mem_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        except Exception as exc:
            info["clocks"] = f"unavailable: {exc}"
        try:
            info["max_clocks_sm_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
            info["max_clocks_mem_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        except Exception as exc:
            info["max_clocks"] = f"unavailable: {exc}"
        pynvml.nvmlShutdown()
    except Exception as exc:
        info["error"] = str(exc)
    return info


def cuda_graph_test():
    """Capture and replay a trivial CUDA graph to prove graph support.

    Returns:
        dict: ``{"ok": True, "before", "after", "increment"}`` on success, or
        ``{"ok": False, "error": ...}`` on failure.
    """
    result = {"ok": False}
    try:
        x = torch.zeros(1, device="cuda", dtype=torch.float32)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                x.add_(1.0)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x.add_(1.0)

        before = x.item()
        g.replay()
        torch.cuda.synchronize()
        after = x.item()
        result = {"ok": True, "before": before, "after": after, "increment": after - before}
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)}
    return result


def profiler_test():
    """Profile a few matmuls and export a chrome trace.

    Returns:
        dict: ``ok``, ``trace`` path, ``trace_bytes`` and the profiler summary
        table. On failure, ``{"ok": False, "error": ...}``.
    """
    result = {"ok": False}
    try:
        from torch.profiler import ProfilerActivity, profile

        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()

        trace_path = os.path.join(RESULTS, "phase0_trace.json")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                torch.mm(a, b)
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        prof.export_chrome_trace(trace_path)
        result = {
            "ok": os.path.exists(trace_path) and os.path.getsize(trace_path) > 0,
            "trace": trace_path,
            "trace_bytes": os.path.getsize(trace_path) if os.path.exists(trace_path) else 0,
            "table": table,
        }
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)}
    return result


def main():
    """Run all Phase 0 checks and write ``results/env.json``.

    Collects GPU/toolchain/NVML info, runs the CUDA-graph and profiler smoke
    tests, prints a JSON summary, and writes the combined result.

    Returns:
        int: 0 if every check passed, 1 otherwise.
    """
    os.makedirs(RESULTS, exist_ok=True)
    if not torch.cuda.is_available():
        print(json.dumps({"fatal": "torch.cuda.is_available() is False"}))
        return 1

    props = torch.cuda.get_device_properties(0)
    gpu = {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_mib": props.total_memory // 2**20,
        "l2_cache_mib": round(getattr(props, "L2_cache_size", 0) / 2**20, 2),
        "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", None),
        "regs_per_sm": getattr(props, "regs_per_multiprocessor", None),
        "shared_mem_per_sm_kib": getattr(props, "shared_memory_per_multiprocessor", 0) // 1024,
        "shared_mem_per_block_kib": getattr(props, "shared_memory_per_block", 0) // 1024,
        "cooperative_launch": bool(getattr(props, "cooperative_launch", False)),
    }

    import transformers
    import triton

    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": ver(torch),
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "cuda_toolkit": cuda_toolkit_version(),
        "triton": ver(triton),
        "transformers": ver(transformers),
        "numpy": pkg_version("numpy"),
        "accelerate": pkg_version("accelerate"),
        "safetensors": pkg_version("safetensors"),
        "pandas": pkg_version("pandas"),
        "matplotlib": pkg_version("matplotlib"),
        "nvidia_ml_py": pkg_version("nvidia-ml-py"),
        "gpu": gpu,
        "nvml": nvml_info(),
    }

    print("== env ==")
    print(json.dumps(env, indent=2))

    graph = cuda_graph_test()
    print("== cuda graph ==")
    print(json.dumps(graph, indent=2))

    prof = profiler_test()
    print("== profiler ==")
    print(prof.pop("table", ""))
    print(json.dumps(prof, indent=2))

    summary = {"env": env, "cuda_graph": graph, "profiler": prof}
    env_path = os.path.join(RESULTS, "env.json")
    with open(env_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {env_path}")

    ok = graph.get("ok") and prof.get("ok")
    print(f"PHASE0_PY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
