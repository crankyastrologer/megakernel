import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PLEN = 128


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


def prefill(model, ids, maxlen):
    """Prefill a fresh ``StaticCache`` of the given length.

    Args:
        model (transformers.PreTrainedModel): Causal LM on CUDA.
        ids (torch.LongTensor): Prompt ids ``(1, prompt_len)``.
        maxlen (int): Allocated cache length.

    Returns:
        tuple[transformers.StaticCache, torch.Tensor]: The cache and the first
        generated token ``(1, 1)``.
    """
    cache = StaticCache(config=model.config, max_cache_len=maxlen)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return cache, out.logits[:, -1].argmax(-1, keepdim=True)


def main():
    """Compare eager static-cache steps with a captured graph step and HF generate."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = make_ids(tok, PLEN)
    with torch.inference_mode():
        ref = model.generate(input_ids=ids, max_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=0)[0, PLEN:].tolist()
    print("ref", ref, flush=True)

    for maxlen in [PLEN + 8, PLEN + 64, PLEN + 512]:
        with torch.inference_mode():
            c, t0 = prefill(model, ids, maxlen)
            o = model(input_ids=t0, past_key_values=c, use_cache=True)
            t1 = o.logits[:, -1].argmax(-1, keepdim=True)
            o = model(input_ids=t1, past_key_values=c, use_cache=True)
            t2 = o.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
        print(f"eager static maxlen={maxlen}: {[t0.item(), t1.item(), t2.item()]}", flush=True)

    # eager step vs capture step
    with torch.inference_mode():
        c, t0 = prefill(model, ids, PLEN + 64)
        o = model(input_ids=t0, past_key_values=c, use_cache=True)
        a = o.logits[:, -1].argmax(-1, keepdim=True).clone()
        torch.cuda.synchronize()

        c2, t0b = prefill(model, ids, PLEN + 64)
        static_ids = t0b.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o2 = model(input_ids=static_ids, past_key_values=c2, use_cache=True)
            b = o2.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        print("eager step", a.item(), "capture step", b.item(), "ref1", ref[1], flush=True)


if __name__ == "__main__":
    main()
