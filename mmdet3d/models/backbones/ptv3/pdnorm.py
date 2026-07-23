# VENDORED from Pointcept
# (pointcept/models/point_prompt_training/prompt_driven_normalization.py).
# Modifications:
#   - dropped `from pointcept.models.builder import MODULES` and the
#     `@MODULES.register_module()` decorator (PDNorm is constructed directly via
#     `functools.partial`, not through a registry).
#   - `from pointcept.models.modules import ...` -> `from .modules import PointModule`
# Class body unmodified. PDNorm is only instantiated when `pdnorm_bn`/`pdnorm_ln`
# are True; the default PTv3 plant config keeps both False.

import torch.nn as nn

from .modules import PointModule


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point
