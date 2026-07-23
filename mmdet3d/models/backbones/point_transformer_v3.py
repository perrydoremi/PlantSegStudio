# Copyright (c) OpenMMLab. All rights reserved.
"""Point Transformer V3 backbone adapter for PlantSegStudio.

This wraps the vendored PTv3 core (``mmdet3d/models/backbones/ptv3/``, see its
``VENDOR.md``) so it plugs into PSS's voxel semantic-segmentation data flow.

The ``MinkUNet`` segmentor calls ``backbone(voxel_features, coors)`` where the
inputs come from ``Det3DDataPreprocessor`` with ``voxel_type='minkunet'``:

- ``voxel_features`` ``(Nv, C)`` -- the point rows of one representative point
  per occupied voxel (``C == load_dim``; the first 3 columns are xyz).
- ``coors`` ``(Nv, 4)`` -- integer voxel indices. With ``batch_first=True`` the
  column order is ``[batch, x, y, z]``.

PTv3 instead consumes a ``Point`` dict (``coord``/``feat``/``grid_coord``/
``batch``). This adapter performs that translation, runs the PTv3 core, and
returns the per-voxel feature tensor ``(Nv, dec_channels[0])`` in the **same row
order** as the input voxels -- so the voxel->point remap in the decode head
(via ``point2voxel_map``) stays valid.
"""

import torch
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from .ptv3 import PointTransformerV3


@MODELS.register_module()
class PointTransformerV3Backbone(BaseModule):
    """Point Transformer V3 backbone for voxel-based semantic segmentation.

    Args:
        in_channels (int): Number of input feature channels fed to PTv3. The
            adapter uses the first ``in_channels`` columns of ``voxel_features``
            as the point feature. Defaults to 3 (xyz).
        init_cfg (dict, optional): mmengine initialization config. Defaults to
            None (PTv3 submodules self-initialize).
        **ptv3_kwargs: Every other Point Transformer V3 hyperparameter (``order``,
            ``stride``, ``enc_depths``, ``enc_channels``, ``enc_num_head``,
            ``enc_patch_size``, ``dec_depths``, ``dec_channels``,
            ``dec_num_head``, ``dec_patch_size``, ``mlp_ratio``, ``drop_path``,
            ``enable_flash``, ``upcast_attention``, ``upcast_softmax``,
            ``pdnorm_*`` ...). Passed straight through to the vendored core.
    """

    def __init__(self, in_channels: int = 3, init_cfg=None, **ptv3_kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        ptv3_kwargs['in_channels'] = in_channels
        self.core = PointTransformerV3(**ptv3_kwargs)
        assert not self.core.enc_mode, (
            'PointTransformerV3Backbone needs the full encoder-decoder '
            '(enc_mode=False) so the output is at input voxel resolution.')

    def forward(self, voxel_features: torch.Tensor,
                coors: torch.Tensor) -> torch.Tensor:
        """Run PTv3 on one batch of voxelized point clouds.

        Args:
            voxel_features (Tensor): ``(Nv, C)`` per-voxel point features; the
                first 3 columns are xyz coordinates.
            coors (Tensor): ``(Nv, 4)`` voxel indices ``[batch, x, y, z]``
                (requires the data preprocessor's ``batch_first=True``).

        Returns:
            Tensor: ``(Nv, dec_channels[0])`` per-voxel features, row-aligned
            with the input voxels.
        """
        assert coors.dim() == 2 and coors.shape[1] == 4, (
            'coors must be (Nv, 4) with batch_first ordering [batch, x, y, z]; '
            'set data_preprocessor.batch_first=True.')
        assert voxel_features.shape[1] >= self.in_channels, (
            f'voxel_features has {voxel_features.shape[1]} channels but '
            f'in_channels={self.in_channels}.')

        data_dict = dict(
            coord=voxel_features[:, :3].float().contiguous(),
            feat=voxel_features[:, :self.in_channels].float().contiguous(),
            grid_coord=coors[:, 1:4].contiguous().int(),
            batch=coors[:, 0].contiguous().long(),
        )
        point = self.core(data_dict)

        # Defensive: unwrap any residual pooling layers. With enc_mode=False the
        # decoder fully unpools, so this loop is a no-op on the happy path.
        while 'pooling_parent' in point.keys():
            parent = point.pop('pooling_parent')
            inverse = point.pop('pooling_inverse')
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        return point.feat
