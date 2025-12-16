import torch

device = "cuda"
dtype = torch.float32

sizes = [
    64, 256, 512,
    1024, 2048, 4096,
    8192, 16384,
    65536,
    1048576,
    1<<22,
    16777216
]

iters = 100

for size in sizes:
    x = torch.randn(1, size, device=device, dtype=dtype)

    # warm-up
    for _ in range(10):
        torch.softmax(x, dim=1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = torch.softmax(x, dim=1)
    end.record()

    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    print(f"size={size:>9}, pytorch softmax = {ms:.6f} ms")
