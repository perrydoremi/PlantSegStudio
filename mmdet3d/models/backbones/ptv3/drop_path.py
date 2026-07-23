"""
Stochastic-depth ``DropPath``.

VENDORED replacement for ``timm.layers.DropPath`` (timm is not installed in the
PSS conda env). This is the standard timm implementation — per-sample drop along
dim 0 with ``scale_by_keep`` — so behaviour matches the Pointcept reference.
"""

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (stochastic depth) per sample (the first dim of ``x``)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
