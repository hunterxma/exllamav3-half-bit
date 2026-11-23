"""Find the tile-count crossover for half-rate GPU fan-out, including pinned PCIe copies.

Run with both GPUs visible and the project extension on PYTHONPATH. Outputs must stay bit-identical;
the timings use CUDA synchronization around the full local or multi-GPU call.
"""

import os
import statistics
import time

import torch

from exllamav3.modules.quant.exl3_lib.quantize import arch_prior_speed, quantize_tiles, quantize_tiles_multigpu


if torch.cuda.device_count() < 2:
    raise SystemExit("requires both GPUs visible")

os.environ["EXL3_FRAC_MGPU_MIN_TILES"] = "1"
torch.manual_seed(20261122)
devices = [0, 1]
ratios = [arch_prior_speed(dev) for dev in devices]
args = {"K": 5.5, "mul1": True, "devices": devices, "device_ratios": ratios}
local_args = {"K": 5.5, "mul1": True, "devices": [devices[0]]}


def elapsed(fn, reps=7):
    values = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return statistics.median(values), out


for count in (128, 256, 512, 1024, 2048, 4096, 8192):
    tiles = torch.randn((count, 256), device="cuda:0")
    quantize_tiles(tiles, local_args)
    quantize_tiles_multigpu(tiles, args)
    local_s, local_out = elapsed(lambda: quantize_tiles(tiles, local_args))
    multi_s, multi_out = elapsed(lambda: quantize_tiles_multigpu(tiles, args))
    exact = torch.equal(local_out[0], multi_out[0]) and torch.equal(local_out[1], multi_out[1])
    print(f"tiles={count:5d} local={local_s*1e3:8.2f}ms multi={multi_s*1e3:8.2f}ms "
          f"speedup={local_s/multi_s:5.2f}x exact={exact}", flush=True)
    if not exact:
        raise SystemExit("multi-GPU half-rate output differed from the single-GPU result")
