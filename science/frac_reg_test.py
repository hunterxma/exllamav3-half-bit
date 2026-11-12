"""
Half-rate trellis quantizer: the register-resident kernel (EXL3_FRAC_REG=1, default) against the generic one
(EXL3_FRAC_REG=0) on the same tiles. Checks the encodings are identical and times both per KA, the
register kernel at 512 and 1024 threads (EXL3_FRAC_REG_NT).

  PYTHONPATH=<ext build dir> python science/frac_reg_test.py [--tiles N] [--ka 2,3,...]
"""
import argparse, os, time
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers_frac

p = argparse.ArgumentParser()
p.add_argument("--tiles", type = int, default = 1024)
p.add_argument("--ka", type = str, default = "2,3,4,5,6,7")
p.add_argument("--reps", type = int, default = 2)
args = p.parse_args()

torch.manual_seed(0)
dev = torch.device("cuda:0")
# Regularized weights are close to unit Gaussian; the drift factor of the scale search is ~1
tiles = torch.randn((args.tiles, 256), device = dev)

def run(ka, reg, nt = None):
    os.environ["EXL3_FRAC_REG"] = "1" if reg else "0"
    if nt: os.environ["EXL3_FRAC_REG_NT"] = str(nt)
    else: os.environ.pop("EXL3_FRAC_REG_NT", None)
    q = torch.zeros_like(tiles)
    idx = torch.zeros_like(tiles, dtype = torch.short)
    tc, te = get_temp_buffers_frac(dev, ka)
    ext.quantize_tiles_frac(tiles, q, idx, tc, te, ka, 0xAAAA)   # warm-up / compile
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.reps):
        ext.quantize_tiles_frac(tiles, q, idx, tc, te, ka, 0xAAAA)
    torch.cuda.synchronize()
    return (time.time() - t0) / args.reps, q, idx

for ka in [int(x) for x in args.ka.split(",")]:
    t_old, q_old, i_old = run(ka, False)
    for nt in (512, 1024):
        t_new, q_new, i_new = run(ka, True, nt)
        same = torch.equal(i_old, i_new) and torch.equal(q_old, q_new)
        mse = (q_new - tiles).square().mean().item()
        print(f"rate {ka + 0.5}: generic {args.tiles / t_old:8.0f} tiles/s  reg/{nt} {args.tiles / t_new:8.0f} tiles/s  "
              f"x{t_old / t_new:.2f}  identical {same}  mse {mse:.5f}", flush = True)
        if not same:
            bad = (i_old != i_new).any(dim = 1).sum().item()
            print(f"   !! {bad} of {args.tiles} tiles differ")
