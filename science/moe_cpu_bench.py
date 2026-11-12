"""
CPU expert kernels (ext.exl3_moe_cpu_forward, the worker that runs the CPU-resident tail experts of a split layer)
speed per trellis rate at a real model's expert shape (default Qwen3.6-35B-A3B: hidden 2048, expert width 512,
top-8). One layer of E CPU experts, a decode step of T tokens routed uniformly at random over them.

    python science/moe_cpu_bench.py [--rates 4,4.5,...] [--experts 104] [--tokens 1] [--threads 6] [--runs 200]
"""
import argparse, os, sys, time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_experts

ap = argparse.ArgumentParser()
ap.add_argument("--rates", default = "4,4.5,5,5.5,6,6.5,7,7.5,8"); ap.add_argument("--experts", type = int, default = 104)
ap.add_argument("--hidden", type = int, default = 2048); ap.add_argument("--inter", type = int, default = 512)
ap.add_argument("--top_k", type = int, default = 8); ap.add_argument("--tokens", type = int, default = 1)
ap.add_argument("--threads", type = int, default = 6); ap.add_argument("--runs", type = int, default = 200)
a = ap.parse_args()

print(f"CPU experts: E={a.experts} H={a.hidden} I={a.inter} tokens={a.tokens} top_k={a.top_k} threads={a.threads}")
print("rate   us/call   GB/s (trellis bytes of the picked experts)")
for K in [float(x) for x in a.rates.split(",")]:
    K = int(K) if K.is_integer() else K
    gen = torch.Generator().manual_seed(0)
    ex = rand_experts(a.experts, a.hidden, a.inter, K, gen)
    lists = []
    for p in ("g", "u", "d"):
        lists += [[e[0].contiguous() for e in ex[p]], [e[1].contiguous() for e in ex[p]], [e[2].contiguous() for e in ex[p]]]
    h = ext.exl3_moe_cpu_make_layer(*lists, [], [], [], 0, 0.0, 0)
    T, k = a.tokens, min(a.top_k, a.experts)
    x = (torch.randn((T, a.hidden), generator = gen) * 0.05).half()
    w = (torch.rand((T, k), generator = gen) * 0.5 + 0.5).half()
    out = torch.empty((T, a.hidden), dtype = torch.float32)
    sels = [torch.stack([torch.randperm(a.experts, generator = gen)[:k] for _ in range(T)]) for _ in range(16)]
    for s in sels[:4]:
        ext.exl3_moe_cpu_forward(h, x, s.contiguous(), w, out, a.threads)
    t0 = time.perf_counter()
    for i in range(a.runs):
        ext.exl3_moe_cpu_forward(h, x, sels[i % 16].contiguous(), w, out, a.threads)
    dt = (time.perf_counter() - t0) / a.runs
    nb = sum(t.numel() * t.element_size() for p in ("g", "u", "d") for t in [ex[p][0][0]]) * len({int(e) for e in sels[0].flatten()})
    print(f"{K:>4}   {dt * 1e6:8.1f}   {nb / dt / 1e9:6.1f}")
    ext.exl3_moe_cpu_free_layer(h)
