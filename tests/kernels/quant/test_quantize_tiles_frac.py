"""
quantize_tiles_frac, the half-integer rate Viterbi quantizer (K = KA + 0.5: KA- and (KA + 1)-bit steps alternating,
MASK 0xAAAA, mul1 codebook only), mirroring tests/kernels/quant/test_quantize_tiles.py for integer K:

- the encoding is a valid tail-biting trellis path at the alternating step widths (window i's top 16 - D(i) bits are
  window i - 1's low bits, wrapping), and the returned tile is exactly the mul1 decode of the returned encoding
- the MSE for unit Gaussian tiles stays within per-rate bounds and below the integer rate KA's MSE on the same tiles
  (the extra half bit has to buy distortion)
- a tile that is itself a decoded valid codeword is recovered exactly
- results do not depend on batch composition or on scratch-wave batching (batches crossing it)
- rejections: rates without an instance, tile length != 256, wrong dtypes; an empty batch is a no-op

References: testlib.trellis (NumPy codebook decode, ring step widths, bitstream unpack); quantize_tiles (integer K)
for the relative distortion bound only.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers_frac, quantize_tiles
from testlib import trellis as tref

HALF_K = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
# Unit Gaussian tiles. The Gaussian rate-distortion function halves the MSE per half bit; the integer-K quantizer
# sits at ~0.28 / 0.068 / 0.018 for K = 1 / 2 / 3, so the half rates are bounded at roughly the geometric mean of
# their neighbours with headroom
max_mse = {1.5: 0.2, 2.5: 0.05, 3.5: 0.013, 4.5: 0.0030, 5.5: 0.0009, 6.5: 0.00034, 7.5: 0.00019}   # 4.5+ measured 0.0024 / 0.00071 / 0.00027 / 0.00015


def _q(x, K):
    return quantize_tiles(x, {"K": K, "mul1": True})


def _check_encoding(y, idx, K):
    idx_np = idx.cpu().numpy().view(np.uint16)
    assert tref.is_tail_biting(idx_np, K), "not a tail-biting path at the alternating step widths"
    dec = tref.decode(idx_np, "mul1").astype(np.float32)
    assert np.array_equal(y.cpu().numpy(), dec), "returned tile is not the decode of the returned encoding"


@pytest.mark.parametrize("ka", [2, 5])
@torch.inference_mode()
def test_scratch_capacity_matches_selected_kernel(device, ka):
    dev = torch.device(device)
    optimized, wave = ext.quantize_tiles_frac_scratch(dev.index, ka, 0xAAAA)
    get_temp_buffers_frac.cache_clear()
    costs, edges = get_temp_buffers_frac(device, ka)
    if optimized:
        assert costs.numel() == 1
        assert edges.shape[0] == wave
        assert edges.dtype == torch.uint8
    else:
        assert costs.shape[0] == 128
        assert edges.shape[0] == 128


@torch.inference_mode()
def test_fractional_scratch_cache_normalizes_device_aliases(device):
    get_temp_buffers_frac.cache_clear()
    dev = torch.device(device)
    by_string = get_temp_buffers_frac(device, 2)
    by_device = get_temp_buffers_frac(dev, 2)
    by_index = get_temp_buffers_frac(dev.index, 2)
    assert by_string[0].data_ptr() == by_device[0].data_ptr() == by_index[0].data_ptr()
    assert by_string[1].data_ptr() == by_device[1].data_ptr() == by_index[1].data_ptr()


@pytest.mark.parametrize("batch", [1, 17, 128, 131])
@pytest.mark.parametrize("K", HALF_K)
@torch.inference_mode()
def test_encode(device, K, batch):
    torch.manual_seed(batch)
    x = torch.randn((batch, 256), device = device)
    y, idx = _q(x, K)
    _check_encoding(y, idx, K)
    mse = F.mse_loss(y, x).item()
    assert mse < max_mse[K], f"mse {mse:.4f}"


@pytest.mark.parametrize("K", HALF_K)
@torch.inference_mode()
def test_half_bit_buys_distortion(device, K):
    torch.manual_seed(0)
    x = torch.randn((256, 256), device = device)
    mse_half = F.mse_loss(_q(x, K)[0], x).item()
    mse_int = F.mse_loss(_q(x, int(K))[0], x).item()
    # above 6 bits the fp16 cost accumulation flattens the curve (measured 6 -> 6.5 -37%, 7 -> 7.5 -20%)
    assert mse_half < (0.75 if K < 6 else 0.9) * mse_int, f"K = {K}: {mse_half:.4f} vs K = {int(K)}: {mse_int:.4f}"


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("K", HALF_K)
@torch.inference_mode()
def test_encode_ideal(device, K, batch):
    """A random valid codeword (every packed bit pattern is one) decodes to a tile the quantizer reproduces exactly"""
    words = tref.unpack(tref.random_packed(batch, K, np.random.default_rng(batch)), K)
    ideal = torch.from_numpy(tref.decode(words, "mul1").astype(np.float32)).to(device)
    y, idx = _q(ideal, K)
    assert torch.equal(y, ideal)
    _check_encoding(y, idx, K)


@pytest.mark.parametrize("K", HALF_K)
@torch.inference_mode()
def test_batch_independence(device, K):
    """Tiles quantize the same alone, in a batch and across the 128-tile scratch batching"""
    torch.manual_seed(1)
    x = torch.randn((260, 256), device = device) * 1.2
    x[0] = 0
    x[1] = 2
    x[2] = torch.linspace(-4, 4, 256, device = device)
    y, idx = _q(x, K)
    _check_encoding(y, idx, K)
    for lo, hi in ((0, 3), (127, 130), (255, 260)):
        ys, idxs = _q(x[lo:hi], K)
        assert torch.equal(idxs, idx[lo:hi]) and torch.equal(ys, y[lo:hi]), (lo, hi)


@torch.inference_mode()
def test_rejections(device):
    x = torch.randn((2, 256), device = device)
    y = torch.empty_like(x)
    i = torch.empty_like(x, dtype = torch.short)
    costs, edges = get_temp_buffers_frac(device, 2)
    with pytest.raises(RuntimeError, match = "no instance"):
        ext.quantize_tiles_frac(x, y, i, costs, edges, 2, 0x5555)
    costs8 = torch.zeros((1, 2, 65536 >> 8), dtype = torch.half, device = device)   # 8.5 bpw: past the 1.5 .. 7.5 instances
    edges8 = torch.zeros((1, 256, 65536 >> 8), dtype = torch.short, device = device)
    with pytest.raises(RuntimeError, match = "no instance"):
        ext.quantize_tiles_frac(x, y, i, costs8, edges8, 8, 0xAAAA)
    x160 = torch.randn((2, 160), device = device)
    with pytest.raises(RuntimeError, match = "tile length"):
        ext.quantize_tiles_frac(x160, torch.empty_like(x160), torch.empty_like(x160, dtype = torch.short),
                                costs, edges, 2, 0xAAAA)
    with pytest.raises(RuntimeError):
        ext.quantize_tiles_frac(x.half(), y, i, costs, edges, 2, 0xAAAA)
    with pytest.raises(RuntimeError):
        ext.quantize_tiles_frac(x, y, i.int(), costs, edges, 2, 0xAAAA)


@torch.inference_mode()
def test_empty_batch(device):
    x = torch.empty((0, 256), device = device)
    costs, edges = get_temp_buffers_frac(device, 1)
    ext.quantize_tiles_frac(x, torch.empty_like(x), torch.empty_like(x, dtype = torch.short), costs, edges, 1, 0xAAAA)
    torch.cuda.synchronize(device)
