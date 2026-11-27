import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PLEN = 128
WARM = 10
MEAS = 40


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
    """Measure the manual StaticCache CUDA-graph decode loop and check parity."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    ids = make_ids(tok, PLEN)
    total = 1 + WARM + MEAS
    maxlen = PLEN + total + 8

    with torch.inference_mode():
        ref = model.generate(input_ids=ids, max_new_tokens=total + 1, do_sample=False, use_cache=True, pad_token_id=0)
    ref = ref[0, PLEN:].tolist()

    with torch.inference_mode():
        cache = StaticCache(config=model.config, max_cache_len=maxlen)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        first = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = [first.item()]
        static_ids = first.clone()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = model(input_ids=static_ids, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        g.replay()
        toks.append(nxt.item())
        static_ids.copy_(nxt)

        evs = []
        for i in range(WARM + MEAS):
            t0 = torch.cuda.Event(True)
            t0.record()
            g.replay()
            t1 = torch.cuda.Event(True)
            t1.record()
            evs.append((t0, t1))
            static_ids.copy_(nxt)
            toks.append(nxt.item())
        torch.cuda.synchronize()

    per = [a.elapsed_time(b) for a, b in evs][WARM:]
    print("tokens", len(toks), "ref", len(ref))
    print("match", toks == ref, "first mismatch",
          next((i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None))
    print(
        "graph: med %.3f ms  p95 %.3f  min %.3f  mean %.3f"
        % (statistics.median(per), sorted(per)[int(0.95 * len(per))], min(per), statistics.mean(per))
    )
    print("tok/s", round(1e3 / statistics.median(per), 1))


if __name__ == "__main__":
    main()
