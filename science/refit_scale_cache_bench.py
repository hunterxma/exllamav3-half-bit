"""Compare streamed versus retained H@W in refit_scales on a real EXL3/BF16 projection pair.

Example:
  PYTHONPATH=<worktree> python science/refit_scale_cache_bench.py \
      --h500 <dir of an EXL3 Qwen3.8-27B build> \
      --bf16 <dir of the BF16 Qwen3.8-27B checkpoint>

H is generated as a synthetic positive-definite Hessian. Weights and the EXL3 reconstruction are loaded from the
local checkpoints. The benchmark requires one visible CUDA device and checks exact fitted outputs and errors.
"""

import argparse
import importlib
import json
import os
import statistics
import time

import torch
from safetensors import safe_open

from exllamav3.modules.quant.exl3 import LinearEXL3


def load_tensor(root, key):
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))['weight_map']
    filename = index[key]
    with safe_open(os.path.join(root, filename), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def median_ms(fn, reps):
    values, output = [], None
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values), output, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h500", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--key", default="model.language_model.layers.0.linear_attn.in_proj_z")
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    qmod = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")
    device = torch.device("cuda", torch.cuda.current_device())
    trellis = load_tensor(args.h500, f"{args.key}.trellis")
    suh = load_tensor(args.h500, f"{args.key}.suh").to(device)
    svh = load_tensor(args.h500, f"{args.key}.svh").to(device)
    mul1 = load_tensor(args.h500, f"{args.key}.mul1")
    weight = load_tensor(args.bf16, f"{args.key}.weight").T.contiguous()
    k, n = weight.shape
    module = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis.to(device), mul1=mul1, key=args.key)
    weight_q = module.get_weight_tensor().float()

    torch.manual_seed(20261123)
    g = torch.randn((k, k), device=device) / k**0.5
    h = g @ g.T
    del g
    h.diagonal().add_(0.25)
    su, sv = torch.ones(k, device=device), torch.ones(n, device=device)

    print(f"GPU={torch.cuda.get_device_name(device)} key={args.key} shape={k}x{n} rate={module.K} "
          f"workspace={k*n*4/2**20:.1f} MiB free={torch.cuda.mem_get_info(device)[0]/2**30:.2f} GiB")
    print(f"auto_cache_allowed={qmod._refit_hw_cache_allowed(device, k, n)}")
    results = {}
    for cache in (False, True):
        qmod._refit_hw_cache_allowed = lambda *_args, _cache=cache: _cache
        median, output, samples = median_ms(
            lambda: qmod.refit_scales(weight, weight_q.clone(), h, su, sv), args.reps
        )
        results[cache] = output
        print(f"{'cached' if cache else 'streamed'} median={median:.2f} ms runs="
              f"{[round(x, 2) for x in samples]}")
    streamed, cached = results[False], results[True]
    exact = all(torch.equal(a, b) for a, b in zip(streamed[:3], cached[:3]))
    error_exact = streamed[3:] == cached[3:]
    print(f"scales_and_weight_bit_identical={exact} reported_errors_identical={error_exact}")
    if not exact or not error_exact:
        raise SystemExit("cached H@W changed the refit result")


if __name__ == "__main__":
    main()
