"""
Fused DECODE MoE kernel (ext.exl3_moe_coop: BlockSparseMLP's bsz <= MAX_BSZN path, used for every decode step and
MTP verify round) speed per trellis rate, at a real model's expert shape (default Qwen3.6-35B-A3B: 256 experts,
hidden 2048, expert width 512, top-8).

    python science/moe_coop_bench.py [--rates ...] [--mixed g:d,...] [--tokens 1,2,6] [--cb mul1|mcg]

Fresh random routing per call over all experts (weights come from DRAM, not L2). Reports microseconds per call and
effective bandwidth (trellis bytes of the distinct picked experts / time) vs the device peak.
"""
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_trellis, rand_scale
from testlib.moe import ptr_table


def proj(E, k, n, K, gen, device):
    tr = [rand_trellis(k, n, K, gen, device) for _ in range(E)]
    su = [rand_scale(k, gen, device) for _ in range(E)]
    sv = [rand_scale(n, gen, device) for _ in range(E)]
    return (tr, su, sv), (ptr_table(tr, device), ptr_table(su, device), ptr_table(sv, device))


@torch.inference_mode()
def bench(E, H, I, top_k, Kgu, Kd, bsz, runs, cb, device):
    gen = torch.Generator().manual_seed(0)
    keep_g, g = proj(E, H, I, Kgu, gen, device)
    keep_u, u = proj(E, H, I, Kgu, gen, device)
    keep_d, d = proj(E, I, H, Kd, gen, device)
    x = (torch.randn((bsz, H), generator = gen) * 0.05).half().to(device)
    slots = bsz * top_k; smax = 4 * slots
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = device); gu_u = torch.empty_like(gu_g)
    act_out = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    d_out = torch.empty((smax, 1, H), dtype = torch.float, device = device)
    out = torch.zeros((8, H), dtype = torch.float, device = device)
    ctr = torch.zeros((smax * (I // 128) + 8 * (H // 128) + 2 * smax + 3,), dtype = torch.int, device = device)
    had_g = torch.empty((slots, H), dtype = torch.half, device = device); had_u = torch.empty_like(had_g)
    sels = [torch.stack([torch.randperm(E, generator = gen)[:top_k] for _ in range(bsz)]).to(device) for _ in range(runs + 10)]
    rw = torch.full((bsz, top_k), 1.0 / top_k, dtype = torch.half, device = device)
    def call(i):
        ext.exl3_moe_coop(x, sels[i], rw, -1, -1, H, *g, *u, *d, None, None, None, Kgu, Kgu, Kd, cb == "mcg", cb == "mul1",
                          0, 0.0, True, had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out[:bsz], None, None)
    for i in range(10): call(i)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    t0.record()
    for i in range(10, runs + 10): call(i)
    t1.record(); torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1000 / runs
    distinct = sum(len(set(s.reshape(-1).tolist())) for s in sels[10:]) / runs
    return us, distinct * (2 * H * I * Kgu + I * H * Kd) / 8 / (us * 1e-6) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type = int, default = 256); ap.add_argument("--hidden", type = int, default = 2048)
    ap.add_argument("--inter", type = int, default = 512); ap.add_argument("--top_k", type = int, default = 8)
    ap.add_argument("--rates", default = "2,2.5,3,3.5,4,4.5,5,5.5,6"); ap.add_argument("--mixed", default = "3:4,4:5,3.5:4.5,2.5:3.5")
    ap.add_argument("--tokens", default = "1,2,6"); ap.add_argument("--runs", type = int, default = 300)
    ap.add_argument("--cb", default = "mul1", choices = ["mul1", "mcg"]); ap.add_argument("--device", default = "cuda:0")
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_properties(a.device).name}  exl3_moe_coop  E={a.experts} H={a.hidden} I={a.inter} top_k={a.top_k} cb={a.cb}")
    cases = [(float(r), float(r)) for r in a.rates.split(",") if r] + [tuple(float(x) for x in m.split(":")) for m in a.mixed.split(",") if m]
    toks = [int(t) for t in a.tokens.split(",")]
    print("gate/up  down  " + "  ".join(f"{t:>2} tok: us    GB/s" for t in toks))
    for Kgu, Kd in cases:
        if a.cb == "mcg" and (Kgu % 1 or Kd % 1):
            continue
        row = []
        for t in toks:
            try:
                us, gbs = bench(a.experts, a.hidden, a.inter, a.top_k, Kgu, Kd, t, a.runs, a.cb, a.device)
                row.append(f"{us:9.1f} {gbs:7.1f}")
            except Exception as e:
                row.append(f"{'err':>9} {str(e)[:7]:>7}")
            torch.cuda.empty_cache()
        print(f"{Kgu:7}  {Kd:4}  " + "  ".join(row), flush = True)


if __name__ == "__main__":
    main()
