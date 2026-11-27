import statistics
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PLEN = 128
STEPS = 50
WARM = 5
DTYPE = torch.bfloat16


def clocks():
    """Return current SM/memory clocks, power and temperature as a CSV line.

    Returns:
        str: nvidia-smi CSV line, or ``repr(error)`` on failure.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu", "--format=csv,noheader"],
            text=True,
        )
        return out.strip()
    except Exception as e:
        return repr(e)


def make_ids(tok, n):
    """Tokenize the experiment text and return ``n`` tokens on CUDA.

    Args:
        tok (transformers.PreTrainedTokenizerBase): Tokenizer.
        n (int): Number of prompt tokens.

    Returns:
        torch.LongTensor: Shape ``(1, n)`` on CUDA.
    """
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy, engineers built small language models that ran on laptops. "
    ) * 200
    return tok(text, return_tensors="pt").input_ids[:, :n].cuda()


def run_eager(model, ids, n):
    """Time ``n`` eager decode steps with per-step CUDA events.

    Args:
        model (transformers.PreTrainedModel): Causal LM on CUDA.
        ids (torch.LongTensor): Prompt ids ``(1, prompt_len)``.
        n (int): Decode steps.

    Returns:
        tuple[list[float], float]: Per-step ms and total wall ms.
    """
    evs = []
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t_all0 = torch.cuda.Event(True)
        t_all0.record()
        t0 = torch.cuda.Event(True)
        for i in range(n):
            t0.record()
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            t1 = torch.cuda.Event(True)
            t1.record()
            evs.append((t0, t1))
            t0 = t1
        t_all1 = torch.cuda.Event(True)
        t_all1.record()
        torch.cuda.synchronize()
    wall = t_all0.elapsed_time(t_all1)
    per = [a.elapsed_time(b) for a, b in evs]
    return per, wall


def run_static(model, ids, n):
    """Time ``n`` static-cache decode steps with per-step CUDA events.

    Args:
        model (transformers.PreTrainedModel): Causal LM on CUDA.
        ids (torch.LongTensor): Prompt ids ``(1, prompt_len)``.
        n (int): Decode steps.

    Returns:
        tuple[list[float], float]: Per-step ms and total wall ms.
    """
    evs = []
    with torch.inference_mode():
        cache = StaticCache(config=model.config, max_cache_len=PLEN + n + WARM + 16)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t_all0 = torch.cuda.Event(True)
        t_all0.record()
        for i in range(n):
            t0 = torch.cuda.Event(True)
            t0.record()
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            t1 = torch.cuda.Event(True)
            t1.record()
            evs.append((t0, t1))
        t_all1 = torch.cuda.Event(True)
        t_all1.record()
        torch.cuda.synchronize()
    per = [a.elapsed_time(b) for a, b in evs]
    return per, t_all0.elapsed_time(t_all1)


def profile_kernels(fn, iters=5):
    """Profile ``fn`` and return average GPU time and kernel launches per call.

    Args:
        fn (Callable[[], None]): Workload to profile.
        iters (int): Iterations inside the profiler. Defaults to 5.

    Returns:
        tuple[float, int, torch.profiler.profile]: ``(cuda_us_per_call,
        kernels_per_call, profiler)``.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    total_cuda = sum(e.self_device_time_total for e in ka)
    calls = sum(e.count for e in ka if e.self_device_time_total > 0)
    return total_cuda / iters, calls / iters, prof


def main():
    """Compare eager vs static-cache decode latency and kernel counts."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda")
    model.eval()
    ids = make_ids(tok, PLEN)
    print("clocks before:", clocks())

    per, wall = run_eager(model, ids, STEPS)
    per = per[WARM:]
    print(
        "eager: wall/step %.3f ms  event med %.3f  p95 %.3f  min %.3f"
        % (wall / STEPS, statistics.median(per), sorted(per)[int(0.95 * len(per))], min(per))
    )

    per, wall = run_static(model, ids, STEPS)
    per = per[WARM:]
    print(
        "static: wall/step %.3f ms  event med %.3f  p95 %.3f  min %.3f"
        % (wall / STEPS, statistics.median(per), sorted(per)[int(0.95 * len(per))], min(per))
    )

    # profile eager kernels
    state = {}

    def one_eager():
        """Run one prefill plus three eager decode steps for profiling."""
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(3):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)

    cuda_us, calls, prof = profile_kernels(one_eager, iters=1)
    print("eager profile (full prefill+3 steps): cuda %.0f us, gpu kernels %d" % (cuda_us, calls))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    print("clocks after:", clocks())


if __name__ == "__main__":
    main()
