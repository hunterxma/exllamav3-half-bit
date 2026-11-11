"""
Fused MoE kernel speed per trellis rate at a real model's expert shape (default Qwen3.6-35B-A3B: 256 experts,
hidden 2048, expert width 512, top-8), for decode-sized batches (1 token, 2, and 6 = an MTP verify round).

    python science/moe_rate_bench.py [--rates 2,2.5,3,3.5,4,4.5,5,6] [--mixed 3:4,4:5] [--tokens 1,2,6] [--mul1]

Each call routes every token to top_k distinct random experts (fresh picks per call, so the experts' weights
are read from DRAM, not L2) through ext.exl3_moe with 16-row tiles. Reports microseconds per call and the
effective bandwidth (trellis bytes of the picked experts / time) against the device's peak, so a rate whose
kernel instance is slow shows as low GB/s. --mixed g:d runs gate/up at g and down at d (runtime-K instance).
"""
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_trellis, rand_scale
from testlib.moe import expert_ptr_tables, moe_buffers


def experts(E, H, I, Kgu, Kd, device, gen):
    ex = {p: [] for p in "gud"}
    for _ in range(E):
        for p, (k, n, K) in (("g", (H, I, Kgu)), ("u", (H, I, Kgu)), ("d", (I, H, Kd))):
            ex[p].append((rand_trellis(k, n, K, gen, device), rand_scale(k, gen, device), rand_scale(n, gen, device)))
    return ex


@torch.inference_mode()
def bench(E, H, I, top_k, Kgu, Kd, tokens, runs, mul1, device):
    gen = torch.Generator().manual_seed(0)
    ex = experts(E, H, I, Kgu, Kd, device, gen)
    tabs = expert_ptr_tables(ex, device)
    bufs = moe_buffers(H, I, 128, device)
    A = tokens * top_k
    y = (torch.randn((A, H), generator = gen) * 0.05).half().to(device)
    out = torch.zeros((A, H), dtype = torch.float, device = device)
    w = torch.full((A,), 1.0 / top_k, dtype = torch.half, device = device)
    # pre-drawn routing per run: each token picks top_k distinct experts; assignments sorted by expert
    layouts = []
    for _ in range(runs + 10):
        picks = torch.stack([torch.randperm(E, generator = gen)[:top_k] for _ in range(tokens)]).reshape(-1)
        counts = torch.bincount(picks, minlength = E)
        token_sorted = torch.argsort(picks).to(device)
        layouts.append((torch.cat([counts, torch.zeros(1, dtype = torch.long)]).to(device), token_sorted, len(set(picks.tolist()))))
    K_int = lambda K: K   # noqa: E731
    def call(i):
        cnt, ts, _ = layouts[i]
        ext.exl3_moe(y, out, cnt, ts, w, *bufs, 0, K_int(Kgu), K_int(Kgu), K_int(Kd), *tabs["g"], *tabs["u"], *tabs["d"],
                     not mul1, mul1, not mul1, mul1, not mul1, mul1, 0.0, -1, None, None, 1, 128, 16)
    for i in range(10): call(i)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    t0.record()
    for i in range(10, runs + 10): call(i)
    t1.record(); torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1000 / runs
    distinct = sum(l[2] for l in layouts[10:]) / runs
    nbytes = distinct * (2 * H * I * Kgu + I * H * Kd) / 8
    return us, nbytes / (us * 1e-6) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type = int, default = 256); ap.add_argument("--hidden", type = int, default = 2048)
    ap.add_argument("--inter", type = int, default = 512); ap.add_argument("--top_k", type = int, default = 8)
    ap.add_argument("--rates", default = "2,2.5,3,3.5,4,4.5,5,5.5,6"); ap.add_argument("--mixed", default = "3:4,4:5,3.5:4.5")
    ap.add_argument("--tokens", default = "1,2,6"); ap.add_argument("--runs", type = int, default = 200)
    ap.add_argument("--mcg", action = "store_true", help = "mcg codebook (integer rates only); default mul1")
    ap.add_argument("--device", default = "cuda:0")
    a = ap.parse_args()
    mul1 = not a.mcg
    props = torch.cuda.get_device_properties(a.device)
    print(f"{props.name}  E={a.experts} H={a.hidden} I={a.inter} top_k={a.top_k}  codebook {'mul1' if mul1 else 'mcg'}")
    cases = [(float(r), float(r)) for r in a.rates.split(",") if r] + \
            [tuple(float(x) for x in m.split(":")) for m in a.mixed.split(",") if m]
    toks = [int(t) for t in a.tokens.split(",")]
    print("gate/up  down  " + "  ".join(f"{t:>2} tok: us    GB/s" for t in toks))
    for Kgu, Kd in cases:
        if not mul1 and (Kgu % 1 or Kd % 1):
            continue
        row = []
        for t in toks:
            try:
                us, gbs = bench(a.experts, a.hidden, a.inter, a.top_k, Kgu, Kd, t, a.runs, mul1, a.device)
                row.append(f"{us:9.1f} {gbs:7.1f}")
            except Exception as e:   # an unsupported combination reports instead of aborting the table
                row.append(f"{'err':>9} {str(e)[:7]:>7}")
            torch.cuda.empty_cache()
        print(f"{Kgu:7}  {Kd:4}  " + "  ".join(row), flush = True)


if __name__ == "__main__":
    main()
