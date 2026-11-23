import torch
from safetensors.torch import save_file

from exllamav3.loader.safetensors import SafetensorsCollection


def test_list_tensors_includes_active_conversion_source(tmp_path):
    save_file({
        "layer.a.trellis": torch.zeros((2, 3, 4), dtype=torch.int16),
        "layer.disk_only": torch.zeros((5,), dtype=torch.float16),
    }, str(tmp_path / "model.safetensors"))
    stc = SafetensorsCollection(str(tmp_path))
    stc.set_new_tensors({
        "layer.a.trellis": torch.zeros((2, 3, 7), dtype=torch.int16),
        "layer.b.trellis": torch.zeros((4, 6, 8), dtype=torch.int16),
    })

    tensors = stc.list_tensors("layer")
    assert tensors["layer.a.trellis"]["shape"] == [2, 3, 7]
    assert tensors["layer.a.trellis"]["n_bytes"] == 2 * 3 * 7 * 2
    assert tensors["layer.b.trellis"]["shape"] == [4, 6, 8]
    assert tensors["layer.disk_only"]["shape"] == [5]

    serializable = stc.list_tensors("layer", only_serializable=True)
    assert "torch_dtype" not in serializable["layer.b.trellis"]
