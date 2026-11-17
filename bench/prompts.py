"""Shared deterministic prompt construction (no heavy imports).

Both the torch harness and the vLLM benchmark use this so prompt token ids are
identical across environments.
"""

PROMPT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a distant galaxy, engineers built small language models that ran on laptops. "
    "Decode speed is limited by memory bandwidth, not arithmetic. "
    "This paragraph exists only to supply deterministic prompt tokens. "
) * 100


def make_prompt_ids(tokenizer, prompt_len):
    """Tokenize the fixed prompt and truncate it to ``prompt_len`` tokens.

    Args:
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer for the model.
        prompt_len (int): Number of prompt tokens to keep.

    Returns:
        torch.LongTensor: Int64 tensor of shape ``(1, prompt_len)``.

    Raises:
        ValueError: If the fixed text tokenizes to fewer than ``prompt_len`` tokens.
    """
    ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids
    if ids.shape[1] < prompt_len:
        raise ValueError(f"prompt text only tokenizes to {ids.shape[1]} < {prompt_len}")
    return ids[:, :prompt_len]
