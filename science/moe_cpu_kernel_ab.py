"""
A/B harness for the CPU expert kernels (exllamav3_ext/cpu/moe_mul1.cpp) without rebuilding the CUDA extension:
each variant .cpp is JIT-compiled alone (torch cpp_extension, the release build's -Ofast) with bindings for the
worker entry points, run on the same random experts and routing, outputs compared bitwise against the first
variant, and timed interleaved (A B A B ...) so drift on a shared machine hits both alike.

    python science/moe_cpu_kernel_ab.py A.cpp B.cpp [...] [--rates 4,4.5,5] [--tokens 1] [--threads 6] [--runs 300] [--rounds 5]

The trellis layout is the one the loader stores for this CPU's tier (testlib.moe_cpu.swizzle, from the installed
extension: the variants must keep the layout functions unchanged).
"""
import argparse, hashlib, os, shutil, sys, tempfile, time
import torch
from torch.utils.cpp_extension import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from testlib.exl3 import rand_experts
from testlib.moe_cpu import swizzle
from exllamav3.ext import exllamav3_ext as ext

BIND = r'''
#include <torch/extension.h>
#include "moe_mul1.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_layer", &exl3_moe_cpu_make_layer);
    m.def("free_layer", &exl3_moe_cpu_free_layer);
    m.def("forward", &exl3_moe_cpu_forward, py::call_guard<py::gil_scoped_release>());
}
'''

ap = argparse.ArgumentParser()
ap.add_argument("variants", nargs = "+")
ap.add_argument("--rates", default = "4,4.5,5")
ap.add_argument("--experts", type = int, default = 104); ap.add_argument("--hidden", type = int, default = 2048)
ap.add_argument("--inter", type = int, default = 512); ap.add_argument("--top_k", type = int, default = 8)
ap.add_argument("--tokens", type = int, default = 1); ap.add_argument("--threads", type = int, default = 6)
ap.add_argument("--runs", type = int, default = 300); ap.add_argument("--rounds", type = int, default = 5)
a = ap.parse_args()

cpu_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "exllamav3_ext", "cpu")
mods = []
for v in a.variants:
    tag = hashlib.sha1(open(v, "rb").read()).hexdigest()[:10]
    d = tempfile.mkdtemp(prefix = f"m1ab_{tag}_")
    shutil.copy(v, os.path.join(d, "moe_mul1.cpp"))
    shutil.copy(os.path.join(cpu_dir, "moe_mul1.h"), d)
    open(os.path.join(d, "bind.cpp"), "w").write(BIND)
    t0 = time.time()
    mods.append(load(name = f"m1ab_{tag}", sources = [os.path.join(d, "moe_mul1.cpp"), os.path.join(d, "bind.cpp")],
                     extra_cflags = ["-Ofast"], extra_include_paths = [d], verbose = False))
    print(f"built {os.path.basename(v)} ({tag}) in {time.time() - t0:.0f} s", flush = True)

print(f"E={a.experts} H={a.hidden} I={a.inter} tokens={a.tokens} top_k={a.top_k} threads={a.threads}")
for K in [float(x) for x in a.rates.split(",")]:
    K = int(K) if K.is_integer() else K
    gen = torch.Generator().manual_seed(0)
    ex = rand_experts(a.experts, a.hidden, a.inter, K, gen)
    swz = ext.exl3_moe_cpu_swizzle_group(K) > 0
    lists = []
    for p in ("g", "u", "d"):
        lists += [[swizzle(e[0]) if swz else e[0].contiguous() for e in ex[p]], [e[1].contiguous() for e in ex[p]],
                  [e[2].contiguous() for e in ex[p]]]
    T, k = a.tokens, min(a.top_k, a.experts)
    x = (torch.randn((T, a.hidden), generator = gen) * 0.05).half()
    w = (torch.rand((T, k), generator = gen) * 0.5 + 0.5).half()
    sels = [torch.stack([torch.randperm(a.experts, generator = gen)[:k] for _ in range(T)]).contiguous() for _ in range(16)]
    hs = [m.make_layer(*lists, [], [], [], 0, 0.0, 1 if swz else 0) for m in mods]
    outs = []
    for m, h in zip(mods, hs):
        o = torch.empty((T, a.hidden), dtype = torch.float32)
        m.forward(h, x, sels[0], w, o, a.threads); outs.append(o.clone())
    same = ["bit-exact" if torch.equal(o, outs[0]) else f"DIFF max {(o - outs[0]).abs().max().item():.3g}" for o in outs]
    times = [[] for _ in mods]
    out = torch.empty((T, a.hidden), dtype = torch.float32)
    for r in range(a.rounds):
        for j, (m, h) in enumerate(zip(mods, hs)):
            for s in sels[:4]:
                m.forward(h, x, s, w, out, a.threads)
            t0 = time.perf_counter()
            for i in range(a.runs):
                m.forward(h, x, sels[i % 16], w, out, a.threads)
            times[j].append((time.perf_counter() - t0) / a.runs * 1e6)
    base = min(times[0])
    for j, v in enumerate(a.variants):
        t = min(times[j])
        print(f"K {K:>4}  {os.path.basename(v):28s} best {t:7.1f} us  median {sorted(times[j])[len(times[j]) // 2]:7.1f}  "
              f"vs A {base / t:5.3f}x  {same[j]}", flush = True)
    for m, h in zip(mods, hs):
        m.free_layer(h)
