import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT_LEN = 128
N_NEW = 32
DTYPE = torch.bfloat16


def p(*a):
    """Print arguments immediately (flushed)."""
    print(*a, flush=True)


def load():
    """Load the experiment model and tokenizer in bf16 on CUDA.

    Returns:
        tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
        ``(tokenizer, model)``.
    """
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda")
    model.eval()
    return tok, model


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
    ids = tok(text, return_tensors="pt").input_ids
    assert ids.shape[1] >= n, (ids.shape, n)
    return ids[:, :n].cuda()


def ref_generate(model, ids, n_new):
    """Greedy-decode reference tokens with HF ``generate``.

    Args:
        model (transformers.PreTrainedModel): Causal LM on CUDA.
        ids (torch.LongTensor): Prompt ids ``(1, prompt_len)``.
        n_new (int): New tokens to generate.

    Returns:
        list[int]: Generated ids (prompt excluded).
    """
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=n_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    return out[0, ids.shape[1] :].tolist()


def mode_eager(model, ids, n_new):
    """Time a manual eager decode loop.

    Args:
        model (transformers.PreTrainedModel): Causal LM on CUDA.
        ids (torch.LongTensor): Prompt ids ``(1, prompt_len)``.
        n_new (int): New tokens to generate.

    Returns:
        tuple[list[int], float]: Generated ids and elapsed seconds.
    """
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = [cur.item()]
        t0 = time.perf_counter()
        for _ in range(n_new - 1):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(cur.item())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return toks, dt


def mode_graph(model, ids, n_new, max_len):
    """Time a StaticCache + CUDA-graph decode loop.

    Args:
        model (transformers.PreTrainedModel): Causal LM on CUDA.
        ids (torch.LongTensor): Prompt ids ``(1, prompt_len)``.
        n_new (int): New tokens to generate.
        max_len (int): Allocated cache length.

    Returns:
        tuple[list[int], float]: Generated ids and elapsed seconds.
    """
    dev = ids.device
    plen = ids.shape[1]
    with torch.inference_mode():
        cache = StaticCache(config=model.config, max_cache_len=max_len)
        pos = torch.arange(plen, device=dev, dtype=torch.long)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=pos)
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)

        static_ids = next_tok.clone()
        static_pos = torch.tensor([plen], device=dev, dtype=torch.long)
        static_mask = torch.ones((1, max_len), device=dev, dtype=torch.long)

        # warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                o = model(
                    input_ids=static_ids,
                    past_key_values=cache,
                    cache_position=static_pos,
                    attention_mask=static_mask,
                    use_cache=True,
                )
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = model(
                input_ids=static_ids,
                past_key_values=cache,
                cache_position=static_pos,
                attention_mask=static_mask,
                use_cache=True,
            )
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)

        toks = [next_tok.item()]
        p_ = plen + 1
        for _ in range(n_new - 1):
            static_ids.copy_(next_tok)
            static_pos.fill_(p_)
            g.replay()
            next_tok = nxt.clone()
            toks.append(next_tok.item())
            p_ += 1
        torch.cuda.synchronize()

        # timed replay loop
        t0 = time.perf_counter()
        for _ in range(n_new - 1):
            static_ids.copy_(next_tok)
            static_pos.fill_(p_)
            g.replay()
            next_tok = nxt.clone()
            p_ += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return toks, dt


def main():
    """Compare eager, compiled, and graph decode latency and token parity."""
    p(torch.__version__, torch.version.cuda)
    tok, model = load()
    p("loaded", model.__class__.__name__, model.config.model_type)
    ids = make_ids(tok, PROMPT_LEN)
    p("ids", ids.shape)

    ref = ref_generate(model, ids, N_NEW)
    p("ref[:8]", ref[:8])

    e_toks, e_dt = mode_eager(model, ids, N_NEW)
    p("eager[:8]", e_toks[:8], "match", e_toks == ref, "ms/tok", round(e_dt / (N_NEW - 1) * 1e3, 3))

    try:
        c_model = torch.compile(model, mode="reduce-overhead")
        c_toks, c_dt = mode_eager(c_model, ids, N_NEW)
        p("compile[:8]", c_toks[:8], "match", c_toks == ref, "ms/tok", round(c_dt / (N_NEW - 1) * 1e3, 3))
    except Exception as exc:
        p("compile FAILED", repr(exc))

    try:
        g_toks, g_dt = mode_graph(model, ids, N_NEW, PROMPT_LEN + N_NEW + 8)
        p("graph[:8]", g_toks[:8], "match", g_toks == ref, "ms/tok", round(g_dt / (N_NEW - 1) * 1e3, 3))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        p("graph FAILED", repr(exc))


if __name__ == "__main__":
    main()
