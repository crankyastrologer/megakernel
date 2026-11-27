import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PLEN = 32
NNEW = 8
MAXLEN = PLEN + NNEW + 8


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
    """Check that static-cache decode tokens match HF greedy generate."""
    torch.cuda.init()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    ids = make_ids(tok, PLEN)

    with torch.inference_mode():
        ref = model.generate(input_ids=ids, max_new_tokens=NNEW, do_sample=False, use_cache=True, pad_token_id=0)
    ref = ref[0, PLEN:].tolist()
    print("ref", ref)

    with torch.inference_mode():
        cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        print("prefill done, logits", out.logits.shape, "cum", cache.layers[0].cumulative_length.item())
        toks = [cur.item()]
        for i in range(NNEW - 1):
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(cur.item())
            print(f"step {i} cum", cache.layers[0].cumulative_length.item(), "tok", cur.item())
        torch.cuda.synchronize()
    print("static toks", toks, "match", toks == ref)


if __name__ == "__main__":
    main()
