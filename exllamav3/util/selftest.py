"""
Post-install self-test: python -m exllamav3.util.selftest

Checks that the compiled extension loads on this GPU and that every EXL3 bitrate (integer 2..8 and the half rates
1.5 .. 7.5) quantizes and decodes correctly through the kernels a model actually uses:
  quantize : Viterbi quantizer on unit-Gaussian tiles, MSE within the per-rate bound
  gemm     : exl3_gemm (prefill / batched path) vs the dequantize-then-matmul reference, m = 1 and 32
The full test suite (tests/, pytest) covers GEMV, MoE, caches and modules.
Exit code 0 = all passed. Runs in about 10-30 s.
"""
import sys
import time

import torch

RATES = [2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
# Unit-Gaussian tile MSE bounds (generous: the measured values are ~20-30% lower)
MSE_MAX = {2: 0.09, 3: 0.025, 4: 0.0065, 5: 0.0018, 6: 0.0006, 7: 0.00030, 8: 0.00022,
           1.5: 0.2, 2.5: 0.05, 3.5: 0.013, 4.5: 0.0030, 5.5: 0.0009, 6.5: 0.00034, 7.5: 0.00019}


def main() -> int:
    t0 = time.time()
    if not torch.cuda.is_available():
        print("FAIL: torch reports no CUDA device (driver, or a CPU-only torch build)")
        return 1
    dev = torch.device("cuda", 0)
    cc = torch.cuda.get_device_capability(dev)
    print(f"GPU {torch.cuda.get_device_name(dev)} (sm_{cc[0]}{cc[1]}), torch {torch.__version__}, CUDA {torch.version.cuda}")
    try:
        from exllamav3.ext import exllamav3_ext as ext
    except Exception as e:
        print(f"FAIL: extension did not load: {e}")
        return 1
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
    from exllamav3.version import __version__
    print(f"exllamav3 {__version__}, extension {ext.__file__}")

    fails = []
    gen = torch.Generator().manual_seed(0)

    def rand_trellis(k, n, K):
        return torch.randint(0, 65536, (k // 16, n // 16, int(16 * K)), dtype = torch.int32, generator = gen).to(torch.int16).to(dev)

    def rand_scale(n):
        mag = torch.rand((n,), generator = gen) * 0.2 + 0.9
        return (mag * torch.where(torch.rand((n,), generator = gen) < 0.5, -1.0, 1.0)).half().to(dev)

    def dense(tr, suh, svh, K):
        w = torch.empty((tr.shape[0] * 16, tr.shape[1] * 16), dtype = torch.half, device = dev)
        ext.reconstruct_had_slice(w, tr, suh, svh, K, False, True, 0)
        return w

    with torch.inference_mode():
        for K in RATES:
            # quantize
            x = torch.randn((64, 256), generator = gen).to(dev)
            y, _ = quantize_tiles(x, {"K": K, "mul1": True})
            mse = (y.float() - x).square().mean().item()
            ok_q = mse < MSE_MAX[K]
            # gemm vs dense reference
            k, n = 512, 512
            tr, suh, svh = rand_trellis(k, n, K), rand_scale(k), rand_scale(n)
            w = dense(tr, suh, svh, K)
            worst = 0.0
            for m in (1, 32):
                A = (torch.randn((m, k), generator = gen) * 0.5).half().to(dev)
                C = torch.empty((m, n), dtype = torch.half, device = dev)
                ext.exl3_gemm(A, tr, C, suh, torch.empty_like(A), svh, 0, False, True, 0)
                ref = (A.float() @ w.float())
                rel = ((C.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()).item()
                worst = max(worst, rel)
            ok_g = worst < 0.02 and torch.isfinite(C).all().item()
            print(f"  {K:>4} bpw  quantize mse {mse:.5f} {'ok' if ok_q else 'FAIL'}   gemm rel err {worst:.4f} {'ok' if ok_g else 'FAIL'}")
            if not (ok_q and ok_g): fails.append(K)
        torch.cuda.synchronize()

    dt = time.time() - t0
    if fails:
        print(f"FAIL at {fails} ({dt:.1f} s). Please report with the output above.")
        return 1
    print(f"OK: all {len(RATES)} rates ({dt:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
