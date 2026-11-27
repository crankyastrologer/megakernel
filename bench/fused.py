"""Fused single-token decoder: HF prefill + Triton decode + CUDA graph.

The HF model is used only to prefill a StaticCache and to supply the final
RMSNorm + lm_head. Decode runs through the Triton kernels in
kernels/triton/decode.py plus the split-K attention in bench/fastattn.py.

One graph replay performs one full token:
    bump seq -> embed -> qkv(norm) -> rope+kv -> attention
    -> o+res -> gate/up+SwiGLU -> down+res -> norm + lm_head + argmax

The graph is self-feeding: the final argmax writes the same token buffer that
embed_lookup reads on the next replay.
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import StaticCache

from bench.fastattn import BLOCK_N, SPLITS, _decode_attn_reduce, _decode_attn_split
from kernels.triton.decode import (
    bump_seq,
    embed_lookup,
    gemv_res,
    norm_stats,
    qkv_gemv,
    rms_gate_up_swiglu,
    rope_kv_append,
)


class FusedDecoder:
    """Hold packed weights, cache, buffers, and the captured decode graph.

    Attributes:
        first: prefill argmax token, shape (1,).
        second: token produced by the first graph replay, shape (1,).
        maxlen: allocated KV-cache length.
    """

    @torch.inference_mode()
    def __init__(self, model, ids, n_steps):
        """Prefill, pack weights, warm up kernels, and capture the step graph.

        Args:
            model: HF causal LM on CUDA (usually already fast-attn patched).
            ids: prompt token ids, shape (1, prompt_len).
            n_steps: number of graph replays the cache must hold.
        """
        cfg = model.config
        self.model = model
        self.H = cfg.hidden_size
        self.I = cfg.intermediate_size
        self.Hq = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or (self.H // self.Hq)
        self.n_layers = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.scale = self.D ** -0.5
        self.Nqkv = self.Hq * self.D + 2 * self.Hkv * self.D
        self.plen = ids.shape[1]
        self.maxlen = self.plen + n_steps + 16
        self.P = self.H // 32          # partials written by gemv_res (BN=32)
        self.PB = 32                   # next power of two >= P

        # ---- prefill and cache ------------------------------------------------
        self.cache = StaticCache(config=cfg, max_cache_len=self.maxlen)
        out = model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.first = out.logits[:, -1].argmax(-1, keepdim=True).reshape(-1).clone()
        self.seq = self.cache.layers[0].cumulative_length.clone().to(torch.int64)

        self.keys = [self.cache.layers[l].keys[0] for l in range(self.n_layers)]
        self.values = [self.cache.layers[l].values[0] for l in range(self.n_layers)]

        # ---- packed weights ---------------------------------------------------
        layers = model.model.layers
        self.Wqkv = torch.stack(
            [
                torch.cat(
                    [
                        layer.self_attn.q_proj.weight,
                        layer.self_attn.k_proj.weight,
                        layer.self_attn.v_proj.weight,
                    ],
                    dim=0,
                )
                for layer in layers
            ]
        ).contiguous()
        self.Wo = torch.stack([layer.self_attn.o_proj.weight for layer in layers]).contiguous()
        self.Wg = torch.stack([layer.mlp.gate_proj.weight for layer in layers]).contiguous()
        self.Wu = torch.stack([layer.mlp.up_proj.weight for layer in layers]).contiguous()
        self.Wd = torch.stack([layer.mlp.down_proj.weight for layer in layers]).contiguous()
        self.n1 = torch.stack([layer.input_layernorm.weight for layer in layers]).contiguous()
        self.n2 = torch.stack([layer.post_attention_layernorm.weight for layer in layers]).contiguous()
        self.embed_w = model.model.embed_tokens.weight   # tied with lm_head
        self.final_norm = model.model.norm

        # ---- RoPE tables ------------------------------------------------------
        dummy = torch.zeros(1, 1, self.H, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(self.maxlen, device="cuda").unsqueeze(0)
        cos, sin = model.model.rotary_emb(dummy, position_ids=positions)
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()

        # ---- static buffers ---------------------------------------------------
        dev = "cuda"
        self.h = torch.empty(self.H, device=dev, dtype=torch.bfloat16)
        self.qkv = torch.empty(self.Nqkv, device=dev, dtype=torch.bfloat16)
        self.g = torch.empty(self.I, device=dev, dtype=torch.bfloat16)
        self.q = self.qkv[: self.Hq * self.D]
        self.k = self.qkv[self.Hq * self.D : self.Hq * self.D + self.Hkv * self.D]
        self.v = self.qkv[self.Hq * self.D + self.Hkv * self.D :]

        self.ss0 = torch.zeros(1, device=dev, dtype=torch.float32)
        self.ss_in = torch.zeros(self.P, device=dev, dtype=torch.float32)
        self.ss_post = torch.zeros(self.P, device=dev, dtype=torch.float32)

        self.part_acc = torch.empty(self.Hq, SPLITS, self.D, device=dev, dtype=torch.float32)
        self.part_m = torch.empty(self.Hq, SPLITS, device=dev, dtype=torch.float32)
        self.part_l = torch.empty(self.Hq, SPLITS, device=dev, dtype=torch.float32)
        self.attn_f32 = torch.empty(self.Hq * self.D, device=dev, dtype=torch.float32)

        self.static_tok = torch.empty(1, device=dev, dtype=torch.int64)
        self.static_tok.copy_(self.first)

        # ---- warm up kernels, reset state, capture ----------------------------
        for _ in range(2):
            self._step_impl()
        torch.cuda.synchronize()
        self.seq.fill_(self.plen)
        self.static_tok.copy_(self.first)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_impl()

        self.graph.replay()
        self.second = self.static_tok.clone()

    def _step_impl(self):
        """Launch one full decode step (recorded into the graph)."""
        bump_seq[(1,)](self.seq)
        embed_lookup[(1,)](self.static_tok, self.embed_w, self.h,
                           H=self.H, BLOCK=1024, num_warps=4)
        norm_stats[(1,)](self.h, self.ss0, K=self.H, BLOCK=1024, num_warps=4)

        for layer in range(self.n_layers):
            ss = self.ss0 if layer == 0 else self.ss_in
            p, pb = (1, 1) if layer == 0 else (self.P, self.PB)
            block = 16
            qkv_gemv[(self.Nqkv // block,)](
                self.h, self.n1[layer], ss, self.Wqkv[layer], self.qkv,
                self.eps, K=self.H, N=self.Nqkv, P=p, PB=pb,
                BN=block, BK=64, num_warps=8,
            )

            rope_kv_append[(self.Hq + 2 * self.Hkv,)](
                self.q, self.k, self.v, self.cos, self.sin,
                self.keys[layer], self.values[layer], self.seq,
                Hq=self.Hq, Hkv=self.Hkv, D=self.D, L=self.maxlen,
                num_warps=2,
            )

            _decode_attn_split[(self.Hq, SPLITS)](
                self.q, self.keys[layer], self.values[layer],
                self.part_acc, self.part_m, self.part_l, self.seq,
                H=self.Hq, KVH=self.Hkv, D=self.D, L=self.maxlen,
                SPLITS=SPLITS,
                CHUNK=(self.maxlen + SPLITS - 1) // SPLITS,
                scale=self.scale, BLOCK_N=BLOCK_N,
            )
            _decode_attn_reduce[(self.Hq,)](
                self.part_acc, self.part_m, self.part_l, self.attn_f32,
                H=self.Hq, D=self.D, SPLITS=SPLITS,
            )
            attn = self.attn_f32.to(torch.bfloat16)

            gemv_res[(self.H // 32,)](
                attn, self.Wo[layer], self.h, self.h, self.ss_post,
                K=self.H, N=self.H, BN=32, BK=64, num_warps=4,
            )
            rms_gate_up_swiglu[(self.I // 32,)](
                self.h, self.n2[layer], self.ss_post,
                self.Wg[layer], self.Wu[layer], self.g, self.eps,
                K=self.H, N=self.I, P=self.P, PB=self.PB,
                BN=32, BK=64, num_warps=8,
            )
            # Down is K=1536, N=576: only 18 CTAs at BN=32. BK=128 + 8
            # warps is ~30% faster than BK=64 (cold, 30-layer graph).
            gemv_res[(self.H // 32,)](
                self.g, self.Wd[layer], self.h, self.h, self.ss_in,
                K=self.I, N=self.H, BN=32, BK=128, num_warps=8, num_stages=1,
            )

        hn = self.final_norm(self.h.view(1, 1, -1)).view(-1)
        logits = F.linear(hn, self.embed_w)
        self.static_tok.copy_(logits.argmax(-1).reshape(1))

    def step(self):
        """Replay one decode step and return the new token (1,)."""
        self.graph.replay()
        return self.static_tok
