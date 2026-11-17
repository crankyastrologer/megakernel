import torch

a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    torch.mm(a, b)
torch.cuda.synchronize()
print("nsys smoke done", torch.cuda.get_device_name(0))
