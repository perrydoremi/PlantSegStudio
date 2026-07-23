# Copyright (c) OpenMMLab. All rights reserved.
import MinkowskiEngine as ME
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch_scatter import scatter_mean

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType


def mask_matrix_nms(masks,
                    labels,
                    scores,
                    filter_thr=-1,
                    nms_pre=-1,
                    max_num=-1,
                    kernel='gaussian',
                    sigma=2.0,
                    mask_area=None):
    """Matrix NMS for multi-class masks.

    This is a copy from mmdet/models/layers/matrix_nms.py.
    We just change the input shape of `masks` tensor.

    Args:
        masks (Tensor): Has shape (num_instances, m)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, m).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-1:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum(1).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix)).triu(
                      diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
                0, *masks.shape[-1:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds


@MODELS.register_module()
class S3DISOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.backbone = backbone
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
        assert self.backbone.type in \
            ['MinkUNetBackboneV2', 'SPVCNNBackbone', 'SpConvUNet',
             'PointTransformerV3Backbone']

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, num_channels, kernel_size=3, padding=1, bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1), torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        if self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            # (x_idx, y_idx, z_idx, batch_idx) for torchsparse backend
            x = self.unet(x['voxels'], x['coors'])
            out = []
            for i in x.C[:, -1].unique():
                out.append(x.F[x.C[:, -1] == i])
            return out
        elif self.backbone.type == 'PointTransformerV3Backbone':
            # PTv3 takes batch-first coors [batch, x, y, z] and returns a flat
            # (Nv, C) feature tensor row-aligned with the input voxels. Split it
            # per batch in coordinate-row order, matching the GT superpoint order
            # built in loss() (torch.unique of voxel indices).
            feats = self.unet(x['voxels'], x['coors'])
            batch = x['coors'][:, 0]
            return [feats[batch == i] for i in batch.unique()]
        elif self.backbone.type == 'SpConvUNet':
            x = self.input_conv(x)
            x, _ = self.unet(x)
            x = self.output_layer(x)
            out = []
            for i in x.indices[:, 0].unique():
                out.append(x.features[x.indices[:, 0] == i])
            return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate([
                ((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                 torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0)))) for p in points
            ])
        else:
            coordinates, features = ME.utils.batch_sparse_collate([
                ((el_p - el_p.min(0)[0]), torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                for el_p, p in zip(elastic_points, points)
            ])

        spatial_shape = torch.clip(coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'], batch_inputs_dict.get('elastic_coords', None))
        if self.backbone.type == 'SpConvUNet':
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape,
                                        len(batch_data_samples))
        elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            x = dict(voxels=features, coors=coordinates[:, [1, 2, 3, 0]])
        elif self.backbone.type == 'PointTransformerV3Backbone':
            # PTv3 wants batch-first coors [batch, x, y, z]; collate's
            # `coordinates` is already in that order (ME convention).
            x = dict(voxels=features, coors=coordinates)

        x = self.extract_feat(x)
        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                self.get_gt_semantic_masks(sem_mask,
                                           voxel_superpoints,
                                           self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                self.get_gt_inst_masks(inst_mask,
                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        if self.backbone.type == 'SpConvUNet':
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape,
                                        len(batch_data_samples))
        elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            x = dict(voxels=features, coors=coordinates[:, [1, 2, 3, 0]])
        elif self.backbone.type == 'PointTransformerV3Backbone':
            # PTv3 wants batch-first coors [batch, x, y, z]; collate's
            # `coordinates` is already in that order (ME convention).
            x = dict(voxels=features, coors=coordinates)

        x = self.extract_feat(x)
        x = self.decoder(x)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  pred_labels[:-self.test_cfg.num_sem_cls, :], superpoints,
                                  self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :], superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(), pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())
        ]

    def pred_inst(self, pred_masks, pred_scores, pred_labels, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes, device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls, 1).flatten(0, 1)

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
                                                   pred_labels[:-n_cls, :], superpoints, thr)
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)

        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0

        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):
        """Create ground truth semantic masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.

        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).

        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class MinkVoxelOneFormer3D(S3DISOneFormer3D):
    """OneFormer3D variant supporting voxelization from preprocessor."""

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        if self.backbone.type == 'SpConvUNet':
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'], batch_inputs_dict.get('elastic_coords', None))
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape,
                                        len(batch_data_samples))
        elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            x = batch_inputs_dict['voxels']

        x = self.extract_feat(x)
        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            if self.backbone.type == 'SpConvUNet':
                voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
                voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
                voxel_superpoints = batch_data_samples[i].point2voxel_map
                assert torch.equal(voxel_superpoints,
                                   torch.unique(voxel_superpoints, return_inverse=True)[1])

            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                self.get_gt_semantic_masks(sem_mask, voxel_superpoints, self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                self.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        if self.backbone.type == 'SpConvUNet':
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape,
                                        len(batch_data_samples))
        elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            x = batch_inputs_dict['voxels']
            inverse_mapping = batch_data_samples[0].point2voxel_map

        x = self.extract_feat(x)
        x = self.decoder(x)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples


@MODELS.register_module()
class DynamicOneFormer3D(S3DISOneFormer3D):
    """OneFormer3D supporting dynamicVFE."""

    def __init__(self, voxel_encoder: ConfigType, **kwargs):
        super().__init__(**kwargs)

        self.voxel_encoder = MODELS.build(voxel_encoder)

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        if self.backbone.type == 'SpConvUNet':
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'], batch_inputs_dict.get('elastic_coords', None))
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape,
                                        len(batch_data_samples))
        elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            x = batch_inputs_dict['voxels']
            voxel_features, feature_coors = self.voxel_encoder(x['voxels'], x['coors'])
            x = dict(voxels=voxel_features, coors=feature_coors[:, [1, 2, 3, 0]])

        x = self.extract_feat(x)
        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            if self.backbone.type == 'SpConvUNet':
                voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
                voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
                voxel_superpoints = batch_data_samples[i].point2voxel_map
                assert torch.equal(voxel_superpoints,
                                   torch.unique(voxel_superpoints, return_inverse=True)[1])

            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                self.get_gt_semantic_masks(sem_mask, voxel_superpoints, self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                self.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        if self.backbone.type == 'SpConvUNet':
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape,
                                        len(batch_data_samples))
        elif self.backbone.type in ['MinkUNetBackboneV2', 'SPVCNNBackbone']:
            x = batch_inputs_dict['voxels']
            voxel_features, feature_coors = self.voxel_encoder(x['voxels'], x['coors'])
            x = dict(voxels=voxel_features, coors=feature_coors[:, [1, 2, 3, 0]])
            inverse_mapping = batch_data_samples[0].point2voxel_map

        x = self.extract_feat(x)
        x = self.decoder(x)

        #TODO ablation the semantic or instance only training
        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat_semantic_only(self, out, superpoints):
        """Predict only semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queries, 1) or None.
            superpoints (Tensor): of shape (n_raw_points, ).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`.
        """
        pred_masks = out['masks'][0]
        sem_res = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls:, :], superpoints)

        return [PointData(pts_semantic_mask=sem_res.cpu().numpy())]

    def predict_by_feat_instance_only(self, out, superpoints):
        """Predict only instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queries, 1) or None.
            superpoints (Tensor): of shape (n_raw_points, ).

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`,
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        inst_res = self.pred_inst(
            pred_masks[:-self.test_cfg.num_sem_cls, :],
            pred_scores[:-self.test_cfg.num_sem_cls, :],
            pred_labels[:-self.test_cfg.num_sem_cls, :],
            superpoints,
            self.test_cfg.inst_score_thr)

        return [
            PointData(
                pts_instance_mask=inst_res[0].cpu().bool().numpy(),
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())
        ]


@MODELS.register_module()
class InstanceOnlyOneFormer3D(Base3DDetector):
    r"""InstanceOnlyOneFormer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(InstanceOnlyOneFormer3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset
        self.num_classes_2dataset = num_classes_2dataset

        self.prefix_1dataset = prefix_1dataset
        self.prefix_2dataset = prefix_2dataset

        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, num_channels, kernel_size=3, padding=1, bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1), torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate([
                ((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                 torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0)))) for p in points
            ])
        else:
            coordinates, features = ME.utils.batch_sparse_collate([
                ((el_p - el_p.min(0)[0]), torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                for el_p, p in zip(elastic_points, points)
            ])

        spatial_shape = torch.clip(coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'], batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_masks = \
                S3DISOneFormer3D.get_gt_inst_masks(
                    inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`,
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds']
        pred_masks = out['masks']
        pred_scores = out['scores']
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        if self.prefix_1dataset in scene_name:
            labels = torch.arange(
                self.num_classes_1dataset,
                device=scores.device).unsqueeze(0).repeat(self.decoder.num_queries_1dataset,
                                                          1).flatten(0, 1)
        elif self.prefix_2dataset in scene_name:
            labels = torch.arange(
                self.num_classes_2dataset,
                device=scores.device).unsqueeze(0).repeat(self.decoder.num_queries_2dataset,
                                                          1).flatten(0, 1)
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        if self.prefix_1dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_1dataset, rounding_mode='floor')
        elif self.prefix_2dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_2dataset, rounding_mode='floor')
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')

        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(pts_instance_mask=mask_pred, instance_labels=labels, instance_scores=scores)
        ]
