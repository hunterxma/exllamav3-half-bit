"""Exactness and small-batch fallback for half-rate multi-GPU tile quantization."""

import torch
import pytest

from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles, quantize_tiles_multigpu


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@torch.inference_mode()
def test_half_rate_multigpu_matches_single_gpu():
    torch.manual_seed(20261122)
    x = torch.randn((1024, 256), device="cuda:0")
    args = {"K": 5.5, "mul1": True, "devices": [0, 1], "device_ratios": [2, 4]}
    reference = quantize_tiles(x, {"K": 5.5, "mul1": True, "devices": [0]})
    actual = quantize_tiles_multigpu(x, args)
    for dev in args["devices"]:
        torch.cuda.synchronize(dev)
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@torch.inference_mode()
def test_half_rate_small_batch_stays_local(monkeypatch):
    torch.manual_seed(17)
    x = torch.randn((32, 256), device="cuda:0")
    monkeypatch.setenv("EXL3_FRAC_MGPU_MIN_TILES", "512")
    args = {"K": 5.5, "mul1": True, "devices": [0, 1], "device_ratios": [2, 4]}
    reference = quantize_tiles(x, {"K": 5.5, "mul1": True, "devices": [0]})
    actual = quantize_tiles_multigpu(x, args)
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@torch.inference_mode()
def test_multigpu_orders_caller_stream_input_and_immediate_output(device, monkeypatch):
    torch.manual_seed(20261123)
    source = torch.randn((256, 256), pin_memory=True)
    first = torch.device(device).index
    second = 1 - first
    devices = [first, second]
    monkeypatch.setenv("EXL3_FRAC_MGPU_MIN_TILES", "1")
    args = {"K": 5.5, "mul1": True, "devices": devices, "device_ratios": [1, 1]}
    reference_input = source.to(device)
    reference = quantize_tiles(reference_input, {"K": 5.5, "mul1": True, "devices": [first]})
    torch.cuda.synchronize()

    caller = torch.cuda.Stream(device=device)
    blocker = torch.cuda.Stream(device=device)
    gate = torch.cuda.Event(blocking=False)
    tiles = torch.empty((256, 256), device=device)
    with torch.cuda.stream(blocker):
        torch.cuda._sleep(50_000_000)
        gate.record(blocker)
    with torch.cuda.stream(caller):
        caller.wait_event(gate)
        tiles.copy_(source, non_blocking=True)
        result = quantize_tiles_multigpu(tiles, args)
        # Consume immediately on the caller's stream, before any host synchronization.
        indices_now = result[1].clone()
        tiles_now = result[0].clone()
    torch.cuda.synchronize()
    assert torch.equal(indices_now, reference[1])
    assert torch.equal(tiles_now, reference[0])


@pytest.mark.parametrize("devices", [[0], [0, 1]])
def test_multigpu_dispatch_rejects_fractional_non_mul1(devices):
    tiles = torch.empty((1, 256), device="cuda:0")
    with pytest.raises(AssertionError, match="mul1 codebook"):
        quantize_tiles_multigpu(tiles, {"K": 5.5, "devices": devices})
