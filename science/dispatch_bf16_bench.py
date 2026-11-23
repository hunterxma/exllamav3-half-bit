"""Compare packed and reconstructed H500 outputs to the local BF16 checkpoint at medium batch sizes.

The default layers are Qwen3.8-27B MLP gate projections. Only these tensors are loaded; this is not a model serve.
"""

import argparse
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


def timed(fn, reps):
    values, output = [], None
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values), output


def rel_rms(actual, reference):
    a, b = actual.float(), reference.float()
    return (a.sub(b).square().mean().sqrt() / b.square().mean().sqrt()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h500", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--layers", default="0,1,2")
    p.add_argument("--rows", default="160,256")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    print("GPU", torch.cuda.get_device_name(device), flush=True)

    for layer in (int(x) for x in args.layers.split(',')):
        key = f"model.language_model.layers.{layer}.mlp.gate_proj"
        trellis = load_tensor(args.h500, f"{key}.trellis").to(device)
        suh = load_tensor(args.h500, f"{key}.suh").to(device)
        svh = load_tensor(args.h500, f"{key}.svh").to(device)
        mul1 = load_tensor(args.h500, f"{key}.mul1")
        w_bf16 = load_tensor(args.bf16, f"{key}.weight")
        out_features, in_features = w_bf16.shape
        linear = LinearEXL3(None, in_features, out_features, suh=suh, svh=svh,
                            trellis=trellis, mul1=mul1, key=key)
        for rows in (int(x) for x in args.rows.split(',')):
            torch.manual_seed(1000 + layer * 10 + rows)
            x = (torch.randn((rows, in_features), device=device, dtype=torch.float16) * 0.5).contiguous()
            reference = (x.float() @ w_bf16.to(device).float().T).half()
            packed_ms, packed = timed(lambda: linear.bc.run_alloc(x, out_features, False), args.reps)
            recon_ms, recon = timed(lambda: linear.reconstruct_hgemm(x, None), args.reps)
            auto_ms, auto = timed(lambda: linear.forward(x, {"reconstruct": False}), args.reps)
            expected = packed if torch.equal(auto, packed) else recon if torch.equal(auto, recon) else None
            if expected is None:
                raise SystemExit("auto dispatch output matched neither existing inference path")
            print(
                f"layer={layer} rows={rows} packed={packed_ms:.3f}ms reconstruct={recon_ms:.3f}ms "
                f"auto={auto_ms:.3f}ms auto_path={'packed' if expected is packed else 'reconstruct'} "
                f"bf16_rel(packed/reconstruct)={rel_rms(packed, reference):.6%}/"
                f"{rel_rms(recon, reference):.6%} path_rel={rel_rms(packed, recon):.6%}",
                flush=True,
            )
        del linear, w_bf16, trellis, suh, svh
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
