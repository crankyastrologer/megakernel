"""Fast single-token decode attention for Llama + StaticCache (split-K Triton).

Problem: with a 4D causal mask, PyTorch SDPA picks the mem-efficient CUTLASS
kernel, which is slow for q_len=1 and scales poorly with KV length. This patch
uses a split-K online-softmax kernel: SPLITS programs per query head each scan
a slice of the KV cache, then a tiny reduction kernel combines them.

The current cache length comes from the device `cumulative_length` scalar, so
the kernels are CUDA-graph-safe (fixed shapes, runtime length). Prefill keeps
the original masked path.
"""

import torch
import triton
import triton.language as tl
from transformers.cache_utils import StaticCache
from transformers.models.llama import modeling_llama as ml

_PATCHED = False
SPLITS = 16
BLOCK_N = 64


@triton.jit
def _decode_attn_split(
    q_ptr,
    k_ptr,
    v_ptr,
    acc_ptr,
    m_ptr,
    l_ptr,
    seq_ptr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    scale,
    BLOCK_N: tl.constexpr,
):
    """Compute one split's partial online-softmax attention for every query head.

    Programs are indexed by ``(query_head, split)``. Each scans the KV slice
    ``[split*CHUNK, split*CHUNK + CHUNK)`` for its KV head and writes partial
    softmax state to the split buffers. Keys beyond the runtime ``seq_len`` are
    masked out.

    Args:
        q_ptr (tl.pointer): Query ``[H, D]`` (bf16).
        k_ptr (tl.pointer): Key cache ``[KVH, L, D]`` (bf16).
        v_ptr (tl.pointer): Value cache ``[KVH, L, D]`` (bf16).
        acc_ptr (tl.pointer): Partial weighted values ``[H, SPLITS, D]`` (fp32).
        m_ptr (tl.pointer): Partial max logits ``[H, SPLITS]`` (fp32).
        l_ptr (tl.pointer): Partial softmax sums ``[H, SPLITS]`` (fp32).
        seq_ptr (tl.pointer): Device scalar with the current valid KV length.
        H (tl.constexpr): Number of query heads.
        KVH (tl.constexpr): Number of KV heads (grouped-query attention).
        D (tl.constexpr): Head dimension.
        L (tl.constexpr): Allocated KV length.
        SPLITS (tl.constexpr): Number of KV splits.
        CHUNK (tl.constexpr): Keys per split.
        scale (float): Attention scale (``1/sqrt(D)``).
        BLOCK_N (tl.constexpr): Keys loaded per online-softmax iteration.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    kvh = h // (H // KVH)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + h * D + offs_d).to(tl.float32)
    seq_len = tl.load(seq_ptr).to(tl.int32)

    base = s * CHUNK

    m = -float("inf")
    l = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for i in tl.static_range(0, CHUNK, BLOCK_N):
        offs = base + i + tl.arange(0, BLOCK_N)
        mask = (offs < seq_len) & (offs < L)
        k = tl.load(
            k_ptr + kvh * L * D + offs[:, None] * D + offs_d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        qk = tl.sum(q[None, :] * k, axis=1) * scale
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m, tl.max(qk))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        p = tl.exp(qk - m_safe)
        p = tl.where(mask, p, 0.0)
        l = l * alpha + tl.sum(p)
        v = tl.load(
            v_ptr + kvh * L * D + offs[:, None] * D + offs_d[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    tl.store(acc_ptr + (h * SPLITS + s) * D + offs_d, acc)
    tl.store(m_ptr + h * SPLITS + s, m)
    tl.store(l_ptr + h * SPLITS + s, l)


@triton.jit
def _decode_attn_reduce(
    acc_ptr,
    m_ptr,
    l_ptr,
    o_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Combine per-split online-softmax partials into the final attention output.

    One program per query head. Rescales each split by the global max, sums the
    softmax denominators and weighted values, and writes ``out = acc / l``.

    Args:
        acc_ptr (tl.pointer): Partial weighted values ``[H, SPLITS, D]`` (fp32).
        m_ptr (tl.pointer): Partial max logits ``[H, SPLITS]`` (fp32).
        l_ptr (tl.pointer): Partial softmax sums ``[H, SPLITS]`` (fp32).
        o_ptr (tl.pointer): Output attention ``[H, D]`` (fp32).
        H (tl.constexpr): Number of query heads.
        D (tl.constexpr): Head dimension.
        SPLITS (tl.constexpr): Number of KV splits.
    """
    h = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, SPLITS)
    m = tl.load(m_ptr + h * SPLITS + offs_s)
    l = tl.load(l_ptr + h * SPLITS + offs_s)
    acc = tl.load(acc_ptr + (h * SPLITS + offs_s)[:, None] * D + offs_d[None, :])

    m_max = tl.max(m)
    m_max = tl.where(m_max == -float("inf"), 0.0, m_max)
    w = tl.exp(m - m_max)
    l_tot = tl.sum(l * w)
    acc_tot = tl.sum(acc * w[:, None], axis=0)
    out = tl.where(l_tot > 0, acc_tot / l_tot, 0.0)
    tl.store(o_ptr + h * D + offs_d, out)


def patch_llama_fast_decode():
    """Install the fast decode-attention patch on ``LlamaAttention``.

    Idempotent. For single-token bf16 decode with a ``StaticCache``, replaces
    the attention forward with the split-K Triton kernels (no causal mask);
    prefill and other dtypes keep the original path. Also makes
    ``create_causal_mask`` return ``None`` for single-token static decode.
    """
    global _PATCHED
    if _PATCHED:
        return

    orig_forward = ml.LlamaAttention.forward

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        """Llama attention forward with a fast single-token static-cache path."""
        if (
            past_key_values is not None
            and isinstance(past_key_values, StaticCache)
            and hidden_states.shape[0] == 1
            and hidden_states.shape[1] == 1
            and hidden_states.dtype == torch.bfloat16
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = ml.apply_rotary_pos_emb(q, k, cos, sin)
            k, v = past_key_values.update(k, v, self.layer_idx)

            layer = past_key_values.layers[self.layer_idx]
            H = q.shape[1]
            D = q.shape[3]
            KVH = k.shape[1]
            L = k.shape[2]
            qc = q.contiguous().view(H, D)
            part_acc = torch.empty((H, SPLITS, D), device=q.device, dtype=torch.float32)
            part_m = torch.empty((H, SPLITS), device=q.device, dtype=torch.float32)
            part_l = torch.empty((H, SPLITS), device=q.device, dtype=torch.float32)
            out = torch.empty((H, D), device=q.device, dtype=torch.float32)

            _decode_attn_split[(H, SPLITS)](
                qc, k, v, part_acc, part_m, part_l, layer.cumulative_length,
                H=H, KVH=KVH, D=D, L=L, SPLITS=SPLITS, CHUNK=(L + SPLITS - 1) // SPLITS,
                scale=self.scaling, BLOCK_N=BLOCK_N,
            )
            _decode_attn_reduce[(H,)](part_acc, part_m, part_l, out, H=H, D=D, SPLITS=SPLITS)

            out = out.to(q.dtype).view(1, H, 1, D)
            out = out.reshape(*input_shape, -1).contiguous()
            return self.o_proj(out), None

        return orig_forward(
            self,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    ml.LlamaAttention.forward = forward

    orig_ccm = ml.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        """Skip causal-mask construction for single-token static decode."""
        ie = kwargs.get("inputs_embeds")
        pkv = kwargs.get("past_key_values")
        if ie is not None and ie.shape[1] == 1 and isinstance(pkv, StaticCache):
            return None
        return orig_ccm(*args, **kwargs)

    ml.create_causal_mask = create_causal_mask
    _PATCHED = True


def fill_cache_sentinels(cache, value=None):
    """No-op retained for call-site compatibility.

    The split-K attention masks unused cache slots via the runtime ``seq_len``,
    so no sentinel key values are required.

    Args:
        cache (StaticCache): Ignored.
        value (float | None): Ignored.
    """
    return
