"""Roofline helpers: weight bytes per decode step, achieved BW, % of peak."""

import json
import os

import torch

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DEFAULT_PEAK_GBS = 187.6


def load_peak_gbs():
    """Return the sustained read bandwidth used as the roofline denominator.

    Reads ``read_gbs`` from ``results/membw.json`` when available.

    Returns:
        float: Peak sustained read bandwidth in GB/s.
    """
    path = os.path.join(RESULTS, "membw.json")
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["read_gbs"])
    return DEFAULT_PEAK_GBS


def dtype_nbytes(dtype):
    """Return the size in bytes of one element of ``dtype``.

    Args:
        dtype (torch.dtype): Tensor dtype, e.g. ``torch.bfloat16``.

    Returns:
        int: Bytes per element.
    """
    return torch.empty((), dtype=dtype).element_size()


def weight_bytes(model):
    """Return the bytes of model weights streamed per decode step.

    ``model.parameters()`` de-duplicates tied weights (e.g. lm_head == embed),
    so each unique parameter is counted once.

    Args:
        model (torch.nn.Module): Loaded language model.

    Returns:
        int: Total parameter bytes.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def kv_bytes_per_token(config, dtype, seq_len=1):
    """Estimate KV-cache bytes read per generated token.

    Computes ``2 * layers * kv_heads * head_dim * seq_len * dtype_bytes``,
    where the leading 2 accounts for both keys and values.

    Args:
        config (transformers.PretrainedConfig): Model config with layer/head fields.
        dtype (torch.dtype): KV cache dtype.
        seq_len (int): Context length to charge per token. Defaults to 1.

    Returns:
        int: Bytes of KV cache traffic for one decode step.
    """
    n_layers = getattr(config, "num_hidden_layers", None)
    n_kv = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads")
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    return 2 * n_layers * n_kv * head_dim * seq_len * dtype_nbytes(dtype)


def metrics(ms_per_token, w_bytes, kv_bytes, peak_gbs):
    """Compute throughput and roofline percentages for one decode configuration.

    Args:
        ms_per_token (float): Median decode latency in milliseconds.
        w_bytes (int): Weight bytes streamed per token.
        kv_bytes (int): KV-cache bytes read per token.
        peak_gbs (float): Roofline denominator in GB/s.

    Returns:
        dict: Keys ``ms_per_token``, ``tok_per_s``, ``weight_gb``,
        ``kv_gb_per_token``, ``weight_gbs``, ``total_gbs``,
        ``pct_weight_roofline``, ``pct_total_roofline`` and ``peak_read_gbs``.
    """
    sec = ms_per_token / 1e3
    weight_gbs = w_bytes / sec / 1e9 if sec > 0 else 0.0
    total_gbs = (w_bytes + kv_bytes) / sec / 1e9 if sec > 0 else 0.0
    return {
        "ms_per_token": ms_per_token,
        "tok_per_s": 1e3 / ms_per_token if ms_per_token > 0 else 0.0,
        "weight_gb": w_bytes / 1e9,
        "kv_gb_per_token": kv_bytes / 1e9,
        "weight_gbs": weight_gbs,
        "total_gbs": total_gbs,
        "pct_weight_roofline": 100.0 * weight_gbs / peak_gbs,
        "pct_total_roofline": 100.0 * total_gbs / peak_gbs,
        "peak_read_gbs": peak_gbs,
    }
