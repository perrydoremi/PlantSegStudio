# Copyright (c) OpenMMLab. All rights reserved.
"""Decode head for the Point Transformer V3 semantic-segmentation model.

It mirrors :class:`MinkUNetHead` (a per-voxel ``nn.Linear`` classifier with a
voxel->point remap for prediction) but additionally supports **a list of
losses**, so the Pointcept reference combination of CrossEntropy + Lovasz can be
reproduced. ``Base3DDecodeHead`` only allows a single ``loss_decode``.
"""

from typing import List, Union

import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class PTv3SegHead(Base3DDecodeHead):
    """Point Transformer V3 segmentation head.

    Args:
        channels (int): Input channels of the classifier (PTv3 ``dec_channels[0]``).
        num_classes (int): Number of semantic classes.
        loss_decode (dict | list[dict]): One or several loss configs. When a
            list is given, every loss is applied to the same logits and the
            components are reported separately (mmengine sums all ``loss*``
            keys). Defaults to a single CrossEntropy loss.
        kwargs: Other :class:`Base3DDecodeHead` arguments (``dropout_ratio``,
            ``ignore_index`` ...).
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 loss_decode: Union[dict, List[dict]] = dict(
                     type='mmdet.CrossEntropyLoss',
                     avg_non_ignore=True,
                     loss_weight=1.0),
                 **kwargs) -> None:
        if isinstance(loss_decode, dict):
            loss_decode = [loss_decode]
        assert len(loss_decode) > 0, 'loss_decode must not be empty.'
        # Base3DDecodeHead.__init__ builds conv_seg/dropout and a single
        # `self.loss_decode`; pass it the first config so the build succeeds,
        # then replace it with the full ModuleList of losses below.
        super().__init__(
            channels, num_classes, loss_decode=loss_decode[0], **kwargs)
        self.loss_decode = nn.ModuleList(
            [MODELS.build(cfg) for cfg in loss_decode])

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Per-voxel features are dense ``[Nv, C]``, so use a Linear layer."""
        return nn.Linear(channels, num_classes)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concatenate voxel-wise ground truth across the batch."""
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.cat(gt_semantic_segs)

    def forward(self, x: Tensor) -> Tensor:
        """Classify each voxel.

        Args:
            x (Tensor): ``(Nv, channels)`` features from the backbone.

        Returns:
            Tensor: ``(Nv, num_classes)`` segmentation logits.
        """
        return self.cls_seg(x)

    def loss_by_feat(self, seg_logit: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute every configured loss on the voxel logits."""
        seg_label = self._stack_batch_gt(batch_data_samples)
        losses = dict()
        for idx, loss_module in enumerate(self.loss_decode):
            name = loss_module.__class__.__name__.replace('Loss', '').lower()
            key = f'loss_{name}' if name else f'loss_decode_{idx}'
            losses[key] = loss_module(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        return losses

    def predict(self, inputs: Tensor,
                batch_data_samples: SampleList) -> List[Tensor]:
        """Predict per-point logits for each scene in the batch.

        Voxel logits are scattered back to points via ``point2voxel_map`` (set
        by the ``minkunet`` voxelizer in ``Det3DDataPreprocessor``).

        Args:
            inputs (Tensor): ``(Nv, channels)`` features from the backbone.
            batch_data_samples (List[Det3DDataSample]): Seg data samples.

        Returns:
            List[Tensor]: Per-scene ``(Npoints, num_classes)`` logits.
        """
        seg_logits = self.forward(inputs)

        batch_idx = torch.cat(
            [data_samples.batch_idx for data_samples in batch_data_samples])
        seg_logit_list = []
        for i, data_sample in enumerate(batch_data_samples):
            seg_logit = seg_logits[batch_idx == i]
            seg_logit = seg_logit[data_sample.point2voxel_map]
            seg_logit_list.append(seg_logit)

        return seg_logit_list
