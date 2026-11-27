import torch
import triton
import triton.language as tl


@triton.jit
def embed_lookup(tok_ptr, w_ptr, out_ptr,
                 H: tl.constexpr, BLOCK: tl.constexpr):
    """Gather the embedding row for the current token.

    This is a gather, not a matmul: it reads one row (~1.1 KB), so it is
    launch-bound.

    Inputs:
        tok_ptr: int64 scalar (or 1-element tensor) holding the token id.
        w_ptr: bf16 [V, H] embedding matrix (tied with lm_head for SmolLM2).
        H: hidden size (576).
        BLOCK: power of two >= H (1024); lanes >= H are masked.

    Output:
        out_ptr: bf16 [H], where out[j] = w[token, j] for j < H.

    Math:
        h[j] = W[token, j]

    Grid: (1,).
    """
    t = tl.load(tok_ptr).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(w_ptr + t * H + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)

@triton.jit
def rms_qkv(x_ptr, nw_ptr, w_ptr, y_ptr, eps,
            K: tl.constexpr, N: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr):
    """Reference fused RMSNorm + packed QKV projection (per-block reduction).

    Kept as a correctness reference. In the hot path prefer qkv_gemv with
    norm_stats or producer partials: recomputing the reduction in every block
    serializes the weight stream (~0.10 ms/step on 30 layers).

    Inputs:
        x_ptr: bf16 [K] hidden state for one token.
        nw_ptr: bf16 [K] input_layernorm weight (gamma).
        w_ptr: bf16 [N, K] packed weights [Wq; Wk; Wv], HF [out, in] layout.
        eps: RMSNorm epsilon (runtime scalar).
        K: hidden size (576).
        N: packed output size, Hq*D + 2*Hkv*D (960).
        BN: output rows per program; grid = N / BN.
        BK: reduction chunk; K % BK == 0 (576 % 64 == 0).

    Output:
        y_ptr: bf16 [N] = [q; k; v] flattened, with q [Hq*D], k/v [Hkv*D].

    Math, per block of BN output rows:
        ss   = sum_k x[k]^2                       (fp32)
        rstd = rsqrt(ss / K + eps)
        xhat = round_bf16(x * rstd)               (HF rounding point 1)
        xn   = round_bf16(xhat * gamma)           (HF rounding point 2)
        y[n] = sum_k W[n, k] * xn[k]              (fp32 accum, bf16 store)

    No bias for SmolLM2 (attention_bias=False). Add a bias epilogue if porting
    to a model with QKV biases (e.g. Qwen2.5).
    """
    pid = tl.program_id(0)
    offs_n = pid*BN + tl.arange(0,BN)

    ss = 0.0
    for k0 in tl.static_range(0,K,BK):
        xv = tl.load(x_ptr +k0 + tl.arange(0,BK)).to(tl.float32)
        ss += tl.sum(xv*xv)
    rstd = tl.rsqrt(ss/K+eps)

    acc = tl.zeros((BN,),tl.float32)
    for k0 in range(0, K, BK):
        xv = tl.load(x_ptr +k0 + tl.arange(0,BK)).to(tl.float32)
        xh = (xv*rstd).to(tl.bfloat16)
        xn = (xh.to(tl.float32) *
              tl.load(nw_ptr + k0 + tl.arange(0, BK)).to(tl.float32)).to(tl.bfloat16)
        w = tl.load(w_ptr + offs_n[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                    eviction_policy="evict_first")
        acc += tl.sum(w.to(tl.float32) * xn.to(tl.float32)[None, :], axis=1)

    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16))

    

@triton.jit
def rope_kv_append(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr,
                   kcache_ptr, vcache_ptr, seq_ptr,
                   Hq: tl.constexpr, Hkv: tl.constexpr,
                   D: tl.constexpr, L: tl.constexpr):
    """Apply RoPE to q/k and append k/v to the KV cache for one token.

    Programs 0..Hq-1 rotate q in place, the next Hkv rotate k into the cache,
    and the last Hkv copy v into the cache.

    Inputs:
        q_ptr: bf16 [Hq*D] query heads; rotated in place.
        k_ptr: bf16 [Hkv*D] key heads, unrotated.
        v_ptr: bf16 [Hkv*D] value heads.
        cos_ptr, sin_ptr: bf16 [L, D] RoPE tables with duplicated halves.
        kcache_ptr, vcache_ptr: bf16 [Hkv, L, D] cache views (batch 1).
        seq_ptr: int64 device scalar with the current cache length. It is
            incremented once per step before this kernel, so pos = seq - 1.
            This kernel does not write seq_ptr.
        Hq, Hkv, D: head counts and head dimension (9, 3, 64).
        L: max cache length, used as the [Hkv, L, D] stride.

    Outputs:
        q_ptr overwritten with q_rot.
        kcache[:, pos, :] = k_rot.
        vcache[:, pos, :] = v (unchanged; no RoPE on values).

    Math (per head, i = 0..D/2-1, fp32 then bf16 store):
        c_i = cos[pos, i], s_i = sin[pos, i]
        y_i        = x_i        * c_i - x_{i+D/2} * s_i
        y_{i+D/2}  = x_{i+D/2}  * c_i + x_i        * s_i

    This is the rotate_half convention over the two halves of each head, not
    adjacent pairs. cos/sin are duplicated halves, so one c_i/s_i drives both.

    Grid: (Hq + 2*Hkv,), one program per head.
    """
    pid = tl.program_id(0)
    pos = tl.load(seq_ptr) - 1
    half = D // 2
    i = tl.arange(0, D // 2)

    c = tl.load(cos_ptr + pos * D + i).to(tl.float32)
    s = tl.load(sin_ptr + pos * D + i).to(tl.float32)

    if pid < Hq:
        base = q_ptr + pid * D
        x1 = tl.load(base + i).to(tl.float32)
        x2 = tl.load(base + half + i).to(tl.float32)
        tl.store(base + i, (x1 * c - x2 * s).to(tl.bfloat16))
        tl.store(base + half + i, (x2 * c + x1 * s).to(tl.bfloat16))

    elif pid < Hq + Hkv:
        g = pid - Hq
        base = k_ptr + g * D
        x1 = tl.load(base + i).to(tl.float32)
        x2 = tl.load(base + half + i).to(tl.float32)
        out = kcache_ptr + g * L * D + pos * D
        tl.store(out + i, (x1 * c - x2 * s).to(tl.bfloat16))
        tl.store(out + half + i, (x2 * c + x1 * s).to(tl.bfloat16))

    else:
        g = pid - Hq - Hkv
        j = tl.arange(0, D)
        x = tl.load(v_ptr + g * D + j)
        tl.store(vcache_ptr + g * L * D + pos * D + j, x)


def rope_kv_append_launch(qkv, cos, sin, kcache, vcache, seq,
                          Hq=9, Hkv=3, D=64, L=2048, num_warps=2):
    """Launch rope_kv_append on a flat q/k/v buffer.

    Args:
        qkv: contiguous bf16 [Hq*D + 2*Hkv*D], laid out as q, then k, then v.
        cos, sin: bf16 [L, D] RoPE tables (duplicated halves).
        kcache, vcache: bf16 [Hkv, L, D] cache views for this layer.
        seq: int64 device scalar holding the current cache length.
    """
    hq = Hq * D
    hkv = Hkv * D
    rope_kv_append[(Hq + 2 * Hkv,)](
        qkv[:hq], qkv[hq:hq + hkv], qkv[hq + hkv:],
        cos, sin, kcache, vcache, seq,
        Hq=Hq, Hkv=Hkv, D=D, L=L, num_warps=num_warps,
    )


@triton.jit
def norm_stats(x_ptr, ss_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    """Compute sum(x^2) for one token: the RMSNorm reduction only.

    Inputs:
        x_ptr: bf16 [K] hidden state that the next RMSNorm will consume.
        K: number of elements (576).
        BLOCK: power of two >= K (1024); lanes >= K are masked.

    Output:
        ss_ptr: one fp32 scalar = sum_i x_i^2. Epsilon is not applied here;
        the consumer computes rstd = rsqrt(ss/K + eps).

    Math:
        ss = sum_{i < K} x_i^2      (fp32 accumulation)

    Grid: (1,). A producer kernel may instead write one partial per CTA; the
    consumer (qkv_gemv / rms_gate_up_swiglu) sums the P partials.
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ss_ptr, tl.sum(x * x))


@triton.jit
def qkv_gemv(x_ptr, nw_ptr, ss_ptr, w_ptr, y_ptr, eps,
             K: tl.constexpr, N: tl.constexpr,
             P: tl.constexpr, PB: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    """Packed QKV GEMV with the RMSNorm reduction supplied as partial sums.

    Every block sums P small partials (cheap), computes rstd, then immediately
    streams its weight rows. This avoids the redundant 576-element reduction
    per block that stalls the weight stream.

    Inputs:
        x_ptr: bf16 [K] hidden state for one token.
        nw_ptr: bf16 [K] norm weight gamma.
        ss_ptr: fp32 [P] partial sums of x^2, from norm_stats or producer CTAs.
        w_ptr: bf16 [N, K] packed [Wq; Wk; Wv].
        eps: RMSNorm epsilon (runtime scalar).
        K: hidden size (576).
        N: packed output size (960).
        P: number of valid partials (e.g. 1 or the producer grid size).
        PB: power of two >= P, for the masked partial load.
        BN: output rows per program; grid = N / BN.
        BK: reduction chunk; K % BK == 0.

    Output:
        y_ptr: bf16 [N] = [q; k; v] flattened.

    Math:
        ss   = sum_{p < P} ss[p]                  (fp32)
        rstd = rsqrt(ss / K + eps)
        xhat = round_bf16(x * rstd)               (HF rounding point 1)
        xn   = round_bf16(xhat * gamma)           (HF rounding point 2)
        y[n] = sum_k W[n, k] * xn[k]              (fp32 accum, bf16 store)

    No bias for SmolLM2; add a bias epilogue for models that have one.
    """
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)

    p = tl.arange(0, PB)
    parts = tl.load(ss_ptr + p, mask=p < P, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(parts) / K + eps)

    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
        xh = (xv * rstd).to(tl.bfloat16)
        xn = (xh.to(tl.float32) *
              tl.load(nw_ptr + k0 + tl.arange(0, BK)).to(tl.float32)).to(tl.bfloat16)
        w = tl.load(w_ptr + offs_n[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                    eviction_policy="evict_first")
        acc += tl.sum(w.to(tl.float32) * xn.to(tl.float32)[None, :], axis=1)

    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16))

@triton.jit
def gemv_res(x_ptr, w_ptr, res_ptr, y_ptr, ss_ptr,
             K: tl.constexpr, N: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    """GEMV + residual add, plus a sum-of-squares partial for the next norm.

    Used for o_proj + residual (K=576, N=576) and down_proj + residual
    (K=1536, N=576). Each program computes one partial; the next RMSNorm
    consumer sums P = N/BN partials.

    Inputs:
        x_ptr: bf16 [K] activation (attention output or SwiGLU intermediate).
        w_ptr: bf16 [N, K] weight, HF [out, in] layout.
        res_ptr: bf16 [N] residual to add; may alias y_ptr.
        y_ptr: bf16 [N] output buffer.
        ss_ptr: fp32 [N/BN] partial buffer; program pid writes ss_ptr[pid].
        K, N: reduction length and output rows.
        BN: output rows per program; grid = N / BN.
        BK: reduction chunk; K % BK == 0.

    Outputs:
        y[n] = round_bf16(res[n] + sum_k W[n, k] * x[k])
        ss[pid] = sum over this block's rows of y[n]^2 (fp32)

    Math:
        acc[n] = sum_k W[n, k] * x[k]            (fp32 accumulation)
        out[n] = res[n] + acc[n]
        y[n]   = round_bf16(out[n])
        ss_partial = sum_{n in block} y[n]^2

    The sum uses the bf16-rounded y, because the next RMSNorm consumes the
    bf16 tensor, not the fp32 pre-rounding value. In-place use (res_ptr ==
    y_ptr) is safe: each program loads only its own rows before storing them.
    """
    pid = tl.program_id(0)
    offs = pid * BN + tl.arange(0, BN)

    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        xv = tl.load(x_ptr + k0 + tl.arange(0,BK)).to(tl.float32)
        w = tl.load(w_ptr + offs[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                    eviction_policy="evict_first")
        acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)

    out = tl.load(res_ptr + offs).to(tl.float32) + acc
    out_bf = out.to(tl.bfloat16)
    tl.store(y_ptr + offs, out_bf)

    # partial for the *next* RMSNorm; must use the bf16-rounded values
    o = out_bf.to(tl.float32)
    tl.store(ss_ptr + pid, tl.sum(o * o))

@triton.jit
def bump_seq(seq_ptr):
    """Increment the shared KV-cache length by one.

    Input:
        seq_ptr: int64 device scalar holding the number of tokens already in
            the cache.

    Output:
        seq_ptr <- seq + 1.

    Call exactly once per decode step, before the layer loop. Append kernels
    write at pos = seq - 1; attention reads seq as the valid cache length.
    Writing seq + 1 from inside a multi-block append kernel would race, which
    is why this is a separate 1-program kernel.

    Grid: (1,).
    """
    tl.store(seq_ptr, tl.load(seq_ptr) + 1)


@triton.jit
def rms_gate_up_swiglu(x_ptr, nw_ptr, ss_ptr, wg_ptr, wu_ptr, y_ptr, eps,
                       K: tl.constexpr, N: tl.constexpr,
                       P: tl.constexpr, PB: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr):
    """RMSNorm scaling + gate/up GEMVs + SiLU(gate) * up.

    Consumes the partial sums written by the preceding gemv_res (o_proj +
    residual) and produces the SwiGLU intermediate consumed by down_proj.

    Inputs:
        x_ptr: bf16 [K] post-attention hidden state.
        nw_ptr: bf16 [K] post_attention_layernorm weight (gamma).
        ss_ptr: fp32 [P] partial sums of x^2, one per producer CTA.
        wg_ptr: bf16 [N, K] gate_proj weight, HF [out, in] layout.
        wu_ptr: bf16 [N, K] up_proj weight.
        eps: RMSNorm epsilon (runtime scalar).
        K: hidden size (576).
        N: intermediate size (1536).
        P: number of valid partials.
        PB: power of two >= P.
        BN: output rows per program; grid = N / BN.
        BK: reduction chunk; K % BK == 0.

    Output:
        y_ptr: bf16 [N], y[n] = silu(gate[n]) * up[n].

    Math:
        ss   = sum_{p < P} ss[p]                  (fp32)
        rstd = rsqrt(ss / K + eps)
        xhat = round_bf16(x * rstd)               (HF rounding point 1)
        xn   = round_bf16(xhat * gamma)           (HF rounding point 2)
        g[n] = sum_k Wg[n, k] * xn[k]             (fp32 accum)
        u[n] = sum_k Wu[n, k] * xn[k]
        g    = round_bf16(g); u = round_bf16(u)
        y[n] = round_bf16(silu(g) * u)            (HF op order)

    No bias for SmolLM2. The input x is read K times per block, but it is tiny
    and L1/L2-hot; the 3.54 MB of gate/up weights dominate.
    """
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)

    p = tl.arange(0, PB)
    parts = tl.load(ss_ptr + p, mask=p < P, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(parts) / K + eps)

    acc_g = tl.zeros((BN,), tl.float32)
    acc_u = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
        xh = (xv * rstd).to(tl.bfloat16)
        xn = (xh.to(tl.float32) *
              tl.load(nw_ptr + k0 + tl.arange(0, BK)).to(tl.float32)).to(tl.bfloat16)
        xnf = xn.to(tl.float32)
        wg = tl.load(wg_ptr + offs_n[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                     eviction_policy="evict_first")
        wu = tl.load(wu_ptr + offs_n[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                     eviction_policy="evict_first")
        acc_g += tl.sum(wg.to(tl.float32) * xnf[None, :], axis=1)
        acc_u += tl.sum(wu.to(tl.float32) * xnf[None, :], axis=1)

    g = acc_g.to(tl.bfloat16)
    u = acc_u.to(tl.bfloat16)
    gf = g.to(tl.float32)
    silu = (gf * tl.sigmoid(gf)).to(tl.bfloat16)
    out = (silu.to(tl.float32) * u.to(tl.float32)).to(tl.bfloat16)
    tl.store(y_ptr + offs_n, out)
