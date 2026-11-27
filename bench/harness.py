"""Phase 1 decode harness: eager, static-cache, CUDA-graph, torch.compile modes.

All modes perform greedy decoding on the same tokenized prompt and expose
per-step CUDA-event timings plus generated token ids for parity checks.

`build_decoder` returns (prefix_tokens, step_fn, cleanup) so the same stepper can
be used for timing and for kernel profiling.
"""

import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from bench.prompts import PROMPT_TEXT, make_prompt_ids as _make_prompt_ids

DTYPE = torch.bfloat16


def load_model(model_id):
    """Load a causal LM and tokenizer in bf16 on CUDA and patch fast attention.

    For Llama-family models, enables the split-K Triton decode-attention patch
    unless ``FAST_ATTN=0`` is set in the environment.

    Args:
        model_id (str): Hugging Face model id.

    Returns:
        tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
        ``(model, tokenizer)``. The tokenizer's pad token defaults to eos/unk.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, device_map="cuda")
    model.eval()
    if os.environ.get("FAST_ATTN", "1") != "0" and getattr(model.config, "model_type", None) == "llama":
        from bench.fastattn import patch_llama_fast_decode

        patch_llama_fast_decode()
    return model, tok


def make_prompt_ids(tok, prompt_len):
    """Build the fixed prompt and move it to CUDA.

    Args:
        tok (transformers.PreTrainedTokenizerBase): Tokenizer for the model.
        prompt_len (int): Number of prompt tokens.

    Returns:
        torch.LongTensor: Shape ``(1, prompt_len)`` on the current CUDA device.
    """
    return _make_prompt_ids(tok, prompt_len).cuda()


@torch.inference_mode()
def reference_tokens(model, ids, n_new):
    """Greedy-decode reference tokens with Hugging Face ``generate``.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids of shape ``(1, prompt_len)``.
        n_new (int): Number of new tokens to generate.

    Returns:
        list[int]: Generated token ids (prompt excluded).
    """
    out = model.generate(
        input_ids=ids,
        max_new_tokens=n_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
    )
    return out[0, ids.shape[1] :].tolist()


def _make_eager(model, ids, n_steps):
    """Build an eager decode stepper using a dynamic KV cache.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        n_steps (int): Planned number of decode steps (unused by eager).

    Returns:
        tuple[list[torch.Tensor], Callable[[], torch.Tensor], None]: One prefix
        token, the step function, and no cleanup info.
    """
    out = model(input_ids=ids, use_cache=True)
    state = {"past": out.past_key_values, "cur": out.logits[:, -1].argmax(-1, keepdim=True)}
    first = state["cur"].clone()

    def step():
        """Run one eager decode step and return the new token ``(1, 1)``."""
        o = model(input_ids=state["cur"], past_key_values=state["past"], use_cache=True)
        state["past"] = o.past_key_values
        state["cur"] = o.logits[:, -1].argmax(-1, keepdim=True)
        return state["cur"]

    return [first], step, None


def _make_static(model, ids, n_steps):
    """Build an eager decode stepper using a preallocated ``StaticCache``.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        n_steps (int): Number of decode steps the cache must hold.

    Returns:
        tuple[list[torch.Tensor], Callable[[], torch.Tensor], dict]: One prefix
        token, the step function, and ``{"max_cache_len": int}``.
    """
    plen = ids.shape[1]
    maxlen = plen + n_steps + 8
    cache = StaticCache(config=model.config, max_cache_len=maxlen)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    from bench.fastattn import fill_cache_sentinels

    fill_cache_sentinels(cache)
    state = {"cur": out.logits[:, -1].argmax(-1, keepdim=True)}
    first = state["cur"].clone()

    def step():
        """Run one static-cache decode step and return the new token ``(1, 1)``."""
        o = model(input_ids=state["cur"], past_key_values=cache, use_cache=True)
        state["cur"] = o.logits[:, -1].argmax(-1, keepdim=True)
        return state["cur"]

    return [first], step, {"max_cache_len": maxlen}


def _make_graph(model, ids, n_steps):
    """Capture a CUDA graph for one ``StaticCache`` decode step.

    Prefills the cache, captures the single-token forward + argmax, replays once
    (capture records without executing), and returns a replay-based stepper.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        n_steps (int): Number of decode steps the cache must hold.

    Returns:
        tuple[list[torch.Tensor], Callable[[], torch.Tensor], dict]: Two prefix
        tokens (prefill + first replay), the step function, and info with
        ``max_cache_len``, the graph, and the cache.
    """
    plen = ids.shape[1]
    maxlen = plen + n_steps + 16
    cache = StaticCache(config=model.config, max_cache_len=maxlen)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    from bench.fastattn import fill_cache_sentinels

    fill_cache_sentinels(cache)
    first = out.logits[:, -1].argmax(-1, keepdim=True).clone()
    static_ids = first.clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = model(input_ids=static_ids, past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)

    g.replay()
    second = nxt.clone()

    def step():
        """Feed the previous token into the graph and replay one decode step."""
        static_ids.copy_(nxt)
        g.replay()
        return nxt

    return [first, second], step, {"max_cache_len": maxlen, "graph": g, "cache": cache}


def _make_compile(model, ids, n_steps, mode=None):
    """Build a ``torch.compile`` decode stepper over a ``StaticCache``.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        n_steps (int): Number of decode steps the cache must hold.
        mode (str | None): ``torch.compile`` mode. Defaults to ``$COMPILE_MODE``
            or ``"default"`` (``reduce-overhead`` segfaults on this stack).

    Returns:
        tuple[list[torch.Tensor], Callable[[], torch.Tensor], dict]: One prefix
        token, the step function, and info with ``max_cache_len`` and the
        compiled module.
    """
    mode = mode or os.environ.get("COMPILE_MODE", "default")
    compiled = torch.compile(model, mode=mode)
    plen = ids.shape[1]
    maxlen = plen + n_steps + 8
    cache = StaticCache(config=model.config, max_cache_len=maxlen)
    out = compiled(input_ids=ids, past_key_values=cache, use_cache=True)
    from bench.fastattn import fill_cache_sentinels

    fill_cache_sentinels(cache)
    state = {"cur": out.logits[:, -1].argmax(-1, keepdim=True).clone()}
    first = state["cur"]

    def step():
        """Run one compiled decode step and return the new token ``(1, 1)``."""
        torch.compiler.cudagraph_mark_step_begin()
        o = compiled(input_ids=state["cur"], past_key_values=cache, use_cache=True)
        state["cur"] = o.logits[:, -1].argmax(-1, keepdim=True).clone()
        return state["cur"]

    return [first], step, {"max_cache_len": maxlen, "compiled": compiled}


def _make_graph_fused(model, ids, n_steps):
    """Build the Triton fused decoder and its captured graph.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        n_steps (int): Number of graph replays to size the cache for.

    Returns:
        tuple[list[torch.Tensor], Callable[[], torch.Tensor], dict]: Two prefix
        tokens, the replay step function, and decoder metadata.
    """
    from bench.fused import FusedDecoder

    dec = FusedDecoder(model, ids, n_steps)
    return [dec.first, dec.second], dec.step, {"max_cache_len": dec.maxlen, "fused": dec}


BUILDERS = {
    "eager": _make_eager,
    "static": _make_static,
    "graph": _make_graph,
    "graph_fused": _make_graph_fused,
    "compile": _make_compile,
}


def build_decoder(model, ids, mode, n_steps):
    """Create a decode stepper for ``mode``.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        mode (str): One of ``eager``, ``static``, ``graph``, ``compile``.
        n_steps (int): Number of decode steps to size caches for.

    Returns:
        tuple[list[torch.Tensor], Callable[[], torch.Tensor], dict | None]:
        ``(prefix_tokens, step_fn, info)``.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in BUILDERS:
        raise ValueError(f"unknown mode {mode}; choose {list(BUILDERS)}")
    return BUILDERS[mode](model, ids, n_steps)


def burn(seconds=None):
    """Run matmuls to bring the GPU to steady clocks before timing.

    WSL does not allow clock locking, and short prompts otherwise run at lower
    clocks. Controlled by ``BURN_SECONDS`` (default 0.3; 0 disables).

    Args:
        seconds (float | None): Burn duration. Defaults to ``$BURN_SECONDS``.
    """
    if seconds is None:
        try:
            seconds = float(os.environ.get("BURN_SECONDS", "0.3"))
        except ValueError:
            seconds = 0.3
    if seconds <= 0:
        return
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        torch.mm(a, b)
    torch.cuda.synchronize()


def _timed_loop(step_fn, n_steps):
    """Run ``n_steps`` steps with CUDA-event timing around each step.

    A steady-clock burn runs first. Per-step times are GPU-side elapsed times
    between the events; tokens are cloned for later parity checks.

    Args:
        step_fn (Callable[[], torch.Tensor]): One decode step.
        n_steps (int): Number of steps to run.

    Returns:
        tuple[list[torch.Tensor], list[float]]: Cloned tokens and per-step
        milliseconds.
    """
    burn()
    evs = []
    toks = []
    for _ in range(n_steps):
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        tok = step_fn()
        t1 = torch.cuda.Event(enable_timing=True)
        t1.record()
        evs.append((t0, t1))
        toks.append(tok.detach().clone())
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) for a, b in evs]
    return toks, times


def _tok_list(tensors):
    """Flatten a list of token tensors into a Python list of ints.

    Args:
        tensors (list[torch.Tensor]): Token tensors of shape ``(1, 1)`` or similar.

    Returns:
        list[int]: Concatenated token ids on CPU.
    """
    if not tensors:
        return []
    return torch.cat([t.reshape(-1) for t in tensors]).cpu().tolist()


@torch.inference_mode()
def run_mode(model, ids, mode, n_steps):
    """Decode ``n_steps`` tokens in ``mode`` and return tokens and timings.

    Args:
        model (transformers.PreTrainedModel): Loaded causal LM on CUDA.
        ids (torch.LongTensor): Prompt token ids ``(1, prompt_len)``.
        mode (str): One of ``eager``, ``static``, ``graph``, ``compile``.
        n_steps (int): Number of decode steps to time.

    Returns:
        dict: ``tokens`` (prefix + generated ids), ``times`` (per-step ms), and
        ``info`` (cache/graph metadata from the builder).
    """
    prefix, step, info = build_decoder(model, ids, mode, n_steps)
    toks, times = _timed_loop(step, n_steps)
    return {
        "tokens": _tok_list(prefix + toks),
        "times": times,
        "info": info,
    }


def time_stats(times, warmup):
    """Summarize per-step timings, discarding ``warmup`` samples.

    Args:
        times (list[float]): Per-step milliseconds from ``run_mode``.
        warmup (int): Leading samples to discard.

    Returns:
        dict: ``n``, ``median_ms``, ``mean_ms``, ``p95_ms``, ``min_ms``, ``max_ms``.
    """
    measured = times[warmup:] if len(times) > warmup else times
    s = sorted(measured)
    return {
        "n": len(measured),
        "median_ms": statistics.median(measured),
        "mean_ms": statistics.mean(measured),
        "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))],
        "min_ms": min(measured),
        "max_ms": max(measured),
    }
