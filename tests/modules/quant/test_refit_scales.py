"""The retained H@W refit path must match the memory-bounded streamed calculation exactly."""

import importlib

import pytest
import torch


quantize = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")


@torch.inference_mode()
def test_cached_hw_refit_is_bit_identical_to_streamed(device, monkeypatch):
    torch.manual_seed(20261123)
    k, n = 48, 64
    w_cpu = torch.randn((k, n), dtype=torch.float32)
    w_dev = w_cpu.to(device)
    g = torch.randn((k, k), device=device)
    h = (g @ g.T) / k + torch.eye(k, device=device) * 0.25
    q0 = w_dev + torch.randn_like(w_dev) * 0.03
    su0 = torch.ones(k, device=device)
    sv0 = torch.ones(n, device=device)

    monkeypatch.setattr(quantize, "_refit_hw_cache_allowed", lambda *_: False)
    streamed = quantize.refit_scales(w_cpu, q0.clone(), h, su0, sv0, chunk=16)
    monkeypatch.setattr(quantize, "_refit_hw_cache_allowed", lambda *_: True)
    cached = quantize.refit_scales(w_cpu, q0.clone(), h, su0, sv0, chunk=16)

    for actual, expected in zip(cached[:3], streamed[:3]):
        assert torch.equal(actual, expected)
    assert cached[3:] == streamed[3:]


def test_large_refit_product_stays_streamed(device):
    assert not quantize._refit_hw_cache_allowed(torch.device(device), 8193, 8192)
