import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PLEN = 128
WARM = 10
MEAS = 30


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


def main():
    """Measure a ``torch.compile`` static-cache decode loop and check parity."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = make_ids(tok, PLEN)
    total = 1 + WARM + MEAS
    with torch.inference_mode():
        ref = model.generate(input_ids=ids, max_new_tokens=total, do_sample=False, use_cache=True, pad_token_id=0)[0, PLEN:].tolist()

    compiled = torch.compile(model, mode="reduce-overhead")

    with torch.inference_mode():
        cache = StaticCache(config=model.config, max_cache_len=PLEN + total + 8)
        out = compiled(input_ids=ids, past_key_values=cache, use_cache=True)
        cur = out.logits[:, -1].argmax(-1, keepdim=True).clone()
        toks = [cur.item()]
        evs = []
        for i in range(WARM + MEAS):
            torch.compiler.cudagraph_mark_step_begin()
            t0 = torch.cuda.Event(True)
            t0.record()
            out = compiled(input_ids=cur, past_key_values=cache, use_cache=True)
            cur = out.logits[:, -1].argmax(-1, keepdim=True).clone()
            t1 = torch.cuda.Event(True)
            t1.record()
            evs.append((t0, t1))
            toks.append(cur.item())
        torch.cuda.synchronize()

    per = [a.elapsed_time(b) for a, b in evs][WARM:]
    print("match", toks == ref, "first mismatch",
          next((i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None))
    print("compile: med %.3f ms  p95 %.3f  min %.3f" % (
        statistics.median(per), sorted(per)[int(0.95 * len(per))], min(per)))


if __name__ == "__main__":
    main()
