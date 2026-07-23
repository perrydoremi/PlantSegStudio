# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from sunjiahao1999/SPFormer.
import functools
import spconv.pytorch as spconv
import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn
from typing import List

from mmdet3d.models.backbones.spconv_unet import ResidualBlock, SpConvUNet
from mmdet3d.models.layers.minkowski_engine_block import IS_MINKOWSKI_ENGINE_AVAILABLE
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import OptMultiConfig

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME


@MODELS.register_module()
class SpConvBackbone(BaseModule):
    """SpConv Backbone model with input convolution, UNet, and output layer.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of channels after input convolution.
        num_planes (List[int]): Number of channels in each level for UNet.
        norm_fn (Callable): Normalization function constructor.
        block_reps (int): Number of times to repeat each block.
        block (Callable): Block base class.
        normalize_before (bool): Whether to call norm before conv.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_planes: List[int] = [32, 32, 32, 32, 32],
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2,
                 block=ResidualBlock,
                 normalize_before=True,
                 return_blocks=False,
                 sparseconv_backend='spconv',
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        assert sparseconv_backend in ['spconv'], \
            f'sparseconv backend: {sparseconv_backend} not supported.'
        self.sparseconv_backend = sparseconv_backend
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))

        self.unet = SpConvUNet(
            num_planes=num_planes,
            norm_fn=norm_fn,
            block_reps=block_reps,
            block=block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks)

        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(base_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function. Only support spconv backend currently.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """
        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
        elif self.sparseconv_backend == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape, batch_size)
        elif self.sparseconv_backend == 'minkowski':
            x = ME.SparseTensor(voxel_features, coors)

        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        return x.features
