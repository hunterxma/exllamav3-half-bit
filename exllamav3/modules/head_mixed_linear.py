"""EXL3 attention projections grouped by independently selected head rates."""

from collections import defaultdict

import torch

from .linear import Linear
from .module import Module


class HeadMixedLinear(Module):
    # Existing fused projections require a single trellis and deliberately fall back here.
    quant_type = "head_mixed"

    def __init__(self, original: Linear, layout: dict):
        super().__init__(original.config, original.key, None)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.out_dtype = original.out_dtype
        self.axis = layout["axis"]
        if self.axis not in (0, 1):
            raise ValueError(f"{self.key}: invalid head split axis")
        dimension = (self.in_features, self.out_features)[self.axis]
        channels = defaultdict(list)
        for _, start, width, rate in sorted(layout["heads"], key=lambda head: head[1]):
            if start < 0 or width <= 0 or start % 128 or width % 128 or rate not in {4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 16}:
                raise ValueError(f"{self.key}: invalid head span or rate")
            channels[rate].extend(range(start, start + width))
        joined = [i for group in channels.values() for i in group]
        if sorted(joined) != list(range(dimension)) or len(channels) < 2:
            raise ValueError(f"{self.key}: head layout must partition all channels into multiple rates")
        self.channel_groups = [torch.tensor(group) for group in channels.values()]
        self.contiguous_groups = [
            (group[0], len(group)) if group[-1] - group[0] + 1 == len(group) else None
            for group in channels.values()
        ]
        self.restore_order = None if joined == list(range(dimension)) else torch.argsort(torch.tensor(joined))
        self.indices = []
        self.inverse = None
        self._source_weight = None

        for rate, group in zip(channels, self.channel_groups):
            in_features = len(group) if self.axis == 0 else self.in_features
            out_features = len(group) if self.axis == 1 else self.out_features
            axis = 1 - self.axis  # checkpoint weight orientation is (out, in)

            def read_weight(stc, key, device, indices=group, gather_axis=axis):
                return self._read_source_group(stc, key, device, gather_axis, indices)

            suffix = str(int(rate * 2))
            key = f"{self.key}.rate_{suffix}"
            self.register_submodule(Linear(
                self.config, key, in_features, out_features,
                # Output groups share their full-input Hessian. Input groups each see a
                # different activation subspace and need their own Hessian.
                qmap=original.qmap if self.axis == 1 else f"{original.qmap}.rate_{suffix}",
                qgroup=key,
                fkey=original.key, frange=(0, out_features), fdequant=read_weight,
                qbits_key=original.qbits_key,
                out_dtype=original.out_dtype,
            ))

    def _read_source_group(self, stc, key, device, axis, indices):
        if self._source_weight is None:
            self._source_weight = stc.get_tensor(
                key + ".weight", "cpu", float2half=True, no_defer=True, arena=False,
            )
        return self._source_weight.index_select(axis, indices).to(device)

    def load(self, device, **kwargs):
        # BF16 conversion reads a parent weight once even when it spans several rate
        # groups. Quantized loads read each group's trellis directly.
        try:
            super().load(device, **kwargs)
        finally:
            self._source_weight = None
        if self.axis == 0:
            self.indices = [indices.to(device) for indices in self.channel_groups]
        elif self.restore_order is not None:
            self.inverse = self.restore_order.to(device)

    def unload(self):
        self.indices = []
        self.inverse = None
        super().unload()

    def optimizer_targets(self):
        return [child.key for child in self.modules]

    def forward(self, x, params, out_dtype=None):
        dtype = out_dtype or self.out_dtype or torch.half
        if self.axis == 1:
            joined = torch.cat([child.forward(x, params, dtype) for child in self.modules], dim=-1)
            return joined if self.inverse is None else joined.index_select(-1, self.inverse)

        # Sum in fp32 even when the caller requests half. Using a separate rounded sum
        # for each input group would add error unrelated to the selected trellis rates.
        total = None
        single_row = x.numel() == self.in_features and x.is_contiguous()
        for child, indices, span in zip(self.modules, self.indices, self.contiguous_groups):
            # A single contiguous row can feed a contiguous head group as a view.
            grouped = x.narrow(-1, *span) if span is not None and single_row else x.index_select(-1, indices)
            partial = child.forward(grouped, params, torch.float)
            total = partial if total is None else total + partial
        return total.to(dtype)


def apply_head_layout(attention, layouts):
    for name in ("q_proj", "k_proj", "v_proj", "qkv_proj", "z_proj", "o_proj"):
        original = getattr(attention, name, None)
        if original is None or original.key not in layouts:
            continue
        mixed = HeadMixedLinear(original, layouts[original.key])
        attention.modules[attention.modules.index(original)] = mixed
        setattr(attention, name, mixed)
