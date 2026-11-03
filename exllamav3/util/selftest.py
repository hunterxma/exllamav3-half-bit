"""
Post-install self-test: python -m exllamav3.util.selftest

Checks that the compiled extension loads on this GPU and that every EXL3 bitrate (integer 2..8 and the half rates
1.5 .. 7.5) quantizes and decodes correctly through the kernels a model actually uses:
  quantize : Viterbi quantizer on unit-Gaussian tiles clamped to +-3 (the codebook's range, as regularized weights
             are), MSE within the per-rate bound and strictly falling with the rate (each half rate between its
             integer neighbours)
  gemm     : exl3_gemm (prefill / batched path) vs the dequantize-then-matmul reference, m = 1 and 32
The full test suite (tests/, pytest) covers GEMV, MoE, caches and modules.
Exit code 0 = all passed. Runs in about 10-30 s.
"""
import sys
import time

import torch

RATES = [2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
# MSE bounds on clamped unit-Gaussian tiles: 1.25x the worst of 8 seeds (RTX 5060 Ti). Unclamped
# tiles are a poor test at high rates: the few samples past the codebook's range dominate the error, so the
# MSE of one draw swings ~2x at 7-8 bpw
MSE_MAX = {2: 0.086, 3: 0.0217, 4: 0.00555, 5: 0.00143, 6: 0.00037, 7: 0.000099, 8: 0.000026,
           1.5: 0.173, 2.5: 0.0432, 3.5: 0.011, 4.5: 0.00281, 5.5: 0.00073, 6.5: 0.00019, 7.5: 0.000051}


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

    mses = {}
    with torch.inference_mode():
        x = torch.randn((64, 256), generator = gen).clamp(-3, 3).to(dev)   # one draw for every rate: comparable
        for K in RATES:
            # quantize
            y, _ = quantize_tiles(x.clone(), {"K": K, "mul1": True})
            mse = (y.float() - x).square().mean().item()
            ok_q = mse < MSE_MAX[K]; mses[K] = mse
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
            print(f"  {K:>4} bpw  quantize mse {mse:.2e} {'ok' if ok_q else 'FAIL'}   gemm rel err {worst:.4f} {'ok' if ok_g else 'FAIL'}")
            if not (ok_q and ok_g): fails.append(K)
        torch.cuda.synchronize()
    order = sorted(mses)
    for lo, hi in zip(order, order[1:]):
        if not mses[hi] < mses[lo]:
            print(f"  FAIL: {hi} bpw (mse {mses[hi]:.2e}) is not better than {lo} bpw ({mses[lo]:.2e})")
            fails.append(hi)

    dt = time.time() - t0
    if fails:
        print(f"FAIL at {fails} ({dt:.1f} s). Please report with the output above.")
        return 1
    print(f"OK: all {len(RATES)} rates ({dt:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
