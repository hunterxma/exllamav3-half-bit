"""Donor rates are checked from safetensors headers before loading GPU-resident weights."""

import json
from types import SimpleNamespace

from exllamav3.conversion import convert_model


class HeaderOnlyHandle:
    def get_slice(self, key):
        assert key == "layer.weight.trellis"
        return SimpleNamespace(get_shape=lambda: (64, 88))  # 5.5 bpw

    def get_tensor(self, key):
        raise AssertionError("mismatched donor trellis must not be loaded or transferred")


class ReusableHandle(HeaderOnlyHandle):
    def __init__(self):
        self.loaded = []

    def get_tensor(self, key):
        self.loaded.append(key)
        return SimpleNamespace(to=lambda device: (key, device))


def _write_donor_index(path):
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "layer.weight.trellis": "model.safetensors",
            "layer.weight.mul1": "model.safetensors",
            "layer.weight.suh": "model.safetensors",
            "layer.weight.svh": "model.safetensors",
        }
    }))


def test_reuse_rejects_different_rate_from_header(tmp_path, monkeypatch):
    _write_donor_index(tmp_path)
    monkeypatch.setenv("EXL3_CONVERT_REUSE", str(tmp_path))
    monkeypatch.delenv("EXL3_CONVERT_REQUANT", raising=False)
    monkeypatch.setattr(convert_model, "_reuse", None)
    monkeypatch.setattr(convert_model, "_st_safe_open", lambda *args, **kwargs: HeaderOnlyHandle())
    linear = SimpleNamespace(key="layer.weight", device="cuda:0")

    assert convert_model.reuse_linears([linear], {"layer.weight": 4.5}) == [linear]


def test_matching_rate_still_reuses_tensor(tmp_path, monkeypatch):
    _write_donor_index(tmp_path)
    handle = ReusableHandle()
    monkeypatch.setenv("EXL3_CONVERT_REUSE", str(tmp_path))
    monkeypatch.delenv("EXL3_CONVERT_REQUANT", raising=False)
    monkeypatch.setattr(convert_model, "_reuse", None)
    monkeypatch.setattr(convert_model, "_st_safe_open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(convert_model, "LinearEXL3", lambda *args, **kwargs: ("reused", args, kwargs))
    inner = SimpleNamespace(get_bias_tensor=lambda: None)
    linear = SimpleNamespace(
        key="layer.weight", device="cuda:0", config=object(), in_features=64,
        out_features=64, out_dtype="fp16", inner=inner,
    )

    assert convert_model.reuse_linears([linear], {"layer.weight": 5.5}) == []
    assert linear.inner[0] == "reused"
    assert handle.loaded == ["layer.weight.trellis", "layer.weight.suh", "layer.weight.svh", "layer.weight.mul1"]
