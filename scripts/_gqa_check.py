import torch
import torch.nn.functional as F

torch.manual_seed(0)
q = torch.randn(1, 9, 1, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 3, 20, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 3, 20, 64, device="cuda", dtype=torch.bfloat16)
a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
b = F.scaled_dot_product_attention(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1))
print("enable_gqa vs expanded max diff:", (a - b).abs().max().item())

# add sentinels
k2 = k.clone()
v2 = v.clone()
k2[:, :, 10:, :] = -1e8
c = F.scaled_dot_product_attention(q, k2, v2, enable_gqa=True)
d = F.scaled_dot_product_attention(q, k2.repeat_interleave(3, 1), v2.repeat_interleave(3, 1))
print("with sentinels diff:", (c - d).abs().max().item())

# compare masked valid-only vs sentinel no-mask
mask = torch.zeros(1, 1, 1, 20, device="cuda", dtype=torch.bool)
mask[..., :10] = True
e = F.scaled_dot_product_attention(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=mask)
print("sentinel vs true-mask max diff:", (c - e).abs().max().item())
print("a sample:", a.flatten()[:4].tolist())
print("c sample:", c.flatten()[:4].tolist())
print("e sample:", e.flatten()[:4].tolist())
