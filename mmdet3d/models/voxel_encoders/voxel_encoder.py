# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
import numpy as np
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from .utils import VFELayer, get_paddings_indicator

import torch_scatter

@MODELS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int, optional): Number of features to use. Default: 4.
    """

    def __init__(self, num_features: int = 4) -> None:
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
        """
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()


@MODELS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

    @torch.no_grad()
    def forward(self, features: Tensor, coors: Tensor, *args,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors


@MODELS.register_module()
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance of
            points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion
            layer used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: list = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_voxel_center: bool = False,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 fusion_layer: dict = None,
                 return_point_feats: bool = False):
        super(DynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    def forward(self,
                features: Tensor,
                coors: Tensor,
                points: Optional[Sequence[Tensor]] = None,
                img_feats: Optional[Sequence[Tensor]] = None,
                img_metas: Optional[dict] = None,
                *args,
                **kwargs) -> tuple:
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image features used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors

@MODELS.register_module()
class DynamicScatterVFE(DynamicVFE):
    """Same with DynamicVFE but use torch_scatter to avoid construct canvas in
    map_voxel_center_to_point.

    The canvas is very memory-consuming when use tiny voxel size (5cm * 5cm * 5cm) in large 3D space.
    """

    def __init__(
        self,
        in_channels=4,
        quantize_channel=32,
        feat_channels=[],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        with_quantize_cat=False,
        init_weights = False,
        # with_point_fpfh=False,
        # fpfh_radius=0.25,
        # fpfh_max_nn=50,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',
        fusion_layer=None,
        return_point_feats=False,
        return_inv=False,
    ):
        # self._with_point_fpfh = with_point_fpfh
        # if self._with_point_fpfh:
        #     in_channels += 33  # FPFH features have 33 dimensions
        super(DynamicScatterVFE, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.mode = mode
        self.with_quantize_cat = with_quantize_cat
        self.return_inv = return_inv
        if with_quantize_cat:
            self.input_layer = nn.Linear(in_channels, quantize_channel)
            # self.voxel_proj = nn.Linear(feat_channels[-1], quantize_channel)
            # self.output_layer = nn.Sequential(
            #     nn.Linear(feat_channels[-1] + quantize_channel, feat_channels[-1] + quantize_channel),
            #     nn.BatchNorm1d(feat_channels[-1] + quantize_channel, eps=1e-4, momentum=0.1),
            #     nn.ReLU(inplace=True)
            # )
            # self.fusion_module = ResidualFusionV2(
            # voxel_dim=feat_channels[-1],  
            # quantize_dim=quantize_channel,  
            # output_dim=quantize_channel  
            # )
                    # With this:
            # self.fusion_module = WeightedFusion()

        # Add LayerNorm for feature normalization
        # self.feature_norm = nn.LayerNorm(self.in_channels)
        if init_weights:
            self.init_weights()
        # self.init_weights()  
        # self.input_layer=nn.Sequential(
        #             nn.Linear(in_channels, quantize_channel),
        #             nn.BatchNorm1d(quantize_channel), nn.ReLU(True))

        # self.fpfh_radius = fpfh_radius
        # self.fpfh_max_nn = fpfh_max_nn
    def _init_layer(self, layer):
        """Initialize a single layer (Linear or BatchNorm1d)."""
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def init_weights(self):
        """Initialize weights for all layers."""
        # Initialize VFE layers
        for vfe in self.vfe_layers:
            for m in vfe.modules():
                self._init_layer(m)
        
        # Initialize quantize layers
        if self.with_quantize_cat:
            self._init_layer(self.input_layer)
            # for m in self.output_layer.modules():
            #     self._init_layer(m)
            # Initialize fusion module
            # for m in self.fusion_module.modules():
            #     self._init_layer(m)

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # def calculate_fpfh(self, points):
    #     """
    #     Calculate FPFH features for a point cloud based on Open3D.
        
    #     Args:
    #         points: Tensor of shape [N, 3] containing xyz coordinates
            
    #     Returns:
    #         fpfh_features: Tensor of shape [N, 33] containing FPFH features
    #     """
    #     # Convert to numpy for Open3D
    #     points_np = points.detach().cpu().numpy()
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(points_np)
        
    #     pc.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #             radius=self.fpfh_radius/2, max_nn=30
    #         )
    #     )

    #     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #         pc, 
    #         o3d.geometry.KDTreeSearchParamHybrid(
    #             radius=self.fpfh_radius, 
    #             max_nn=self.fpfh_max_nn
    #         )
    #     )
    #     fpfh_tensor = torch.from_numpy(fpfh.data.T.copy()).to(
    #         device=points.device, dtype=points.dtype
    #     )
    #     return fpfh_tensor


    # if out_fp16=True, the large numbers of points
    # lead to overflow error in following layers
    def forward(self, features, coors, points=None, img_feats=None, img_metas=None):
        if isinstance(features, (list, tuple)):
            # features: [dynamic, quantize]
            # coors: [dynamic, quantize]
            features_ls = [features[0]]
            feature_dynamic = features[0]
            features_quantize = features[1]
            coors = coors[0]
        else:
            # features: dynamic
            # coors: dynamic
            features_ls = [features]
            feature_dynamic = features
            features_quantize = None
        # origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors, unq_inv = self.scatter_v2(feature_dynamic, coors, 'avg')
            points_mean = self.map_voxel_center_to_point(voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = feature_dynamic[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = feature_dynamic.new_zeros(size=(feature_dynamic.size(0), 3))
            f_center[:, 0] = feature_dynamic[:, 0] - (
                coors[:, 3].type_as(feature_dynamic) * self.vx + self.x_offset)
            f_center[:, 1] = feature_dynamic[:, 1] - (
                coors[:, 2].type_as(feature_dynamic) * self.vy + self.y_offset)
            f_center[:, 2] = feature_dynamic[:, 2] - (
                coors[:, 1].type_as(feature_dynamic) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(feature_dynamic[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Add FPFH features if enabled
        # if self._with_point_fpfh:
        #     # Calculate FPFH features for each point
        #     fpfh_features = self.calculate_fpfh(features[:, :3])
        #     features_ls.append(fpfh_features)        

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # features = self.feature_norm(features)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats, img_metas)
            voxel_feats, voxel_coors, unq_inv = self.scatter_v2(point_feats, coors, self.mode)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.with_quantize_cat:
            features_q = self.input_layer(features_quantize)
            # MLPs output
            voxel_feats = torch.cat([voxel_feats, features_q], dim=1)

            #TODO only for ablation; Sparse quantize only
            # voxel_feats = features_q
            
            # TODO: Addition
            # voxel_feats = self.voxel_proj(voxel_feats)
            # voxel_feats = voxel_feats + features_q
            

            # voxel_feats = self.output_layer(voxel_feats)
            # Residual fusion
            # voxel_feats = self.fusion_module(voxel_feats, features_q)

            # Weighted fusion
            # voxel_feats = self.fusion_module(voxel_feats, features_q)

        if self.return_point_feats:
            return point_feats

        if self.return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors

    def scatter_v2(self, feat, coors, mode):

        new_coors, unq_inv, _ = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

        if mode == 'max':
            new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
        elif mode == 'avg':
            new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='mean')
        else:
            raise NotImplementedError

        return new_feat, new_coors, unq_inv

    def scatter_quantize(self, feat, coors, mode):
        coors_numpy = coors.cpu().numpy()
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coors_numpy), return_index=True, return_inverse=True)
        inverse_indices = torch.from_numpy(inverse_indices).to(feat.device)
        new_coors = coors[indices]
        if mode == 'max':
            new_feat, argmax = torch_scatter.scatter_max(feat, inverse_indices, dim=0)
        elif mode == 'avg':
            new_feat = torch_scatter.scatter(feat, inverse_indices, dim=0, reduce='mean')
        else:
            raise NotImplementedError

        return new_feat, new_coors, inverse_indices.long()

    def ravel_hash(self, x: np.ndarray):
        """Get voxel coordinates hash for np.unique.

        Args:
            x (np.ndarray): The voxel coordinates of points, Nx3.

        Returns:
            np.ndarray: Voxels coordinates hash.
        """
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

# class ResidualFusion(nn.Module):
#     def __init__(self, voxel_dim=32, quantize_dim=64, output_dim=64):
#         super().__init__()
#         # Expand voxel features to match quantized dimension
#         self.voxel_proj = nn.Linear(voxel_dim, quantize_dim)  # 32 -> 64
        
#         self.fusion_layers = nn.Sequential(
#             nn.Linear(quantize_dim * 2, quantize_dim * 2),  # 128 -> 128
#             nn.BatchNorm1d(quantize_dim * 2),
#             nn.ReLU(),
#             nn.Linear(quantize_dim * 2, output_dim),  # 128 -> 64
#             nn.BatchNorm1d(output_dim)
#         )
        
#         self.residual_proj = nn.Linear(voxel_dim, output_dim)  # 32 -> 64
    
#     def forward(self, voxel_feats, features_q):
#         v_proj = self.voxel_proj(voxel_feats)  # [N, 32] -> [N, 64]
#         concat_feats = torch.cat([v_proj, features_q], dim=1)  # [N, 128]
#         fused = self.fusion_layers(concat_feats)  # [N, 128] -> [N, 64]
        
#         residual = self.residual_proj(voxel_feats)  # [N, 32] -> [N, 64]
#         return torch.nn.functional.relu(fused + residual)  # [N, 64]

# class ResidualFusionV2(nn.Module):
#     def __init__(self, voxel_dim=32, quantize_dim=64, output_dim=64):
#         super().__init__()
#         # Expand voxel features to match quantized dimension
#         self.voxel_proj = nn.Linear(voxel_dim, quantize_dim)  # 32 -> 64
        
#         self.fusion_layers = nn.Sequential(
#             nn.Linear(quantize_dim * 2, quantize_dim * 2),  # 128 -> 128
#             nn.BatchNorm1d(quantize_dim * 2),
#             nn.ReLU(),
#             nn.Linear(quantize_dim * 2, quantize_dim),  # 128 -> 64
#             nn.BatchNorm1d(quantize_dim)
#         )
        
#         # Residual processing without addition
#         self.residual_proj = nn.Linear(voxel_dim, quantize_dim)  # 32 -> 64
        
#         # Final fusion layer that combines fused and residual features
#         self.final_fusion = nn.Sequential(
#             nn.Linear(quantize_dim * 2, output_dim),  # 128 -> 64
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU()
#         )
    
#     def forward(self, voxel_feats, features_q):
#         v_proj = self.voxel_proj(voxel_feats)  # [N, 32] -> [N, 64]
#         concat_feats = torch.cat([v_proj, features_q], dim=1)  # [N, 128]
#         fused = self.fusion_layers(concat_feats)  # [N, 128] -> [N, 64]
        
#         residual = self.residual_proj(voxel_feats)  # [N, 32] -> [N, 64]
        
#         # Concatenate instead of adding
#         combined = torch.cat([fused, residual], dim=1)  # [N, 128]
#         return self.final_fusion(combined)  # [N, 128] -> [N, 64]

# class WeightedFusion(nn.Module):
#     """
#     Learnable weighted fusion between voxel and quantized features.
#     """
    
#     def __init__(self):
#         super().__init__()

#         self.fusion_weight = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))  # [w1, w2] for voxel and quantized features
#         # self.output_proj = nn.Linear(voxel_dim + quantize_dim, output_dim)  # e.g., 32 + 64 -> 64

#     def forward(self, voxel_feats, features_q):
#         """
#         Args:
#             voxel_feats: [N, 32] - Features from voxel processing
#             features_q:  [N, 64] - Features from quantization
            
#         Returns:
#             fused: [N, output_dim] - Weighted combination of both features
#         """
        
#         weights = 2.0 * F.softmax(self.fusion_weight, dim=0)  # [2] -> [w1, w2] where w1+w2=1
#         fused = torch.cat([weights[0] * voxel_feats, weights[1] * features_q], dim=1)
#         # fused = self.output_proj(fused)
            
#         return fused

@MODELS.register_module()
class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance
            of points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance to
            center of voxel for each points inside a voxel. Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points inside a
            voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion layer
            used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the
            features of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: list = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_voxel_center: bool = False,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 fusion_layer: dict = None,
                 return_point_feats: bool = False):
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    def forward(self,
                features: Tensor,
                num_points: Tensor,
                coors: Tensor,
                img_feats: Optional[Sequence[Tensor]] = None,
                img_metas: Optional[dict] = None,
                *args,
                **kwargs) -> tuple:
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image features used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if (self.fusion_layer is not None and img_feats is not None):
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
                                                coors, img_feats, img_metas)

        return voxel_feats

    def fusion_with_mask(self, features: Tensor, mask: Tensor,
                         voxel_feats: Tensor, coors: Tensor,
                         img_feats: Sequence[Tensor],
                         img_metas: Sequence[dict]) -> Tensor:
        """Fuse image and point features with mask.

        Args:
            features (torch.Tensor): Features of voxel, usually it is the
                values of points in voxels.
            mask (torch.Tensor): Mask indicates valid features in each voxel.
            voxel_feats (torch.Tensor): Features of voxels.
            coors (torch.Tensor): Coordinates of each single voxel.
            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.
            img_metas (list(dict)): Meta information of image and points.

        Returns:
            torch.Tensor: Fused features of each voxel.
        """
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats,
                                        img_metas)

        voxel_canvas = voxel_feats.new_zeros(
            size=(voxel_feats.size(0), voxel_feats.size(1),
                  point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out


@MODELS.register_module()
class SegVFE(nn.Module):
    """Voxel feature encoder used in segmentation task.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 6.
        feat_channels (list(int)): Channels of features in VFE.
        with_voxel_center (bool): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float]): Size of a single voxel (rho, phi, z).
            Defaults to None.
        grid_shape (tuple[float]): The grid shape of voxelization.
            Defaults to (480, 360, 32).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Defaults to (0, -3.14159265359, -4, 50, 3.14159265359, 2).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        with_pre_norm (bool): Whether to use the norm layer before
            input vfe layer.
        feat_compression (int, optional): The voxel feature compression
            channels, Defaults to None
        return_point_feats (bool): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 6,
                 feat_channels: Sequence[int] = [],
                 with_voxel_center: bool = False,
                 voxel_size: Optional[Sequence[float]] = None,
                 grid_shape: Sequence[float] = (480, 360, 32),
                 point_cloud_range: Sequence[float] = (0, -3.14159265359, -4,
                                                       50, 3.14159265359, 2),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
                 mode: bool = 'max',
                 with_pre_norm: bool = True,
                 feat_compression: Optional[int] = None,
                 return_point_feats: bool = False) -> None:
        super(SegVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        assert not (voxel_size and grid_shape), \
            'voxel_size and grid_shape cannot be setting at the same time'
        if with_voxel_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        self.point_cloud_range = point_cloud_range
        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] -
                          point_cloud_range[:3]) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (
                grid_shape - 1)
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = self.voxel_size[0]
        self.vy = self.voxel_size[1]
        self.vz = self.voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                vfe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                vfe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters), norm_layer,
                        nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.vfe_scatter = DynamicScatter(self.voxel_size,
                                          self.point_cloud_range,
                                          (mode != 'max'))
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression), nn.ReLU())

    def forward(self, features: Tensor, coors: Tensor, *args,
                **kwargs) -> Tuple[Tensor]:
        """Forward functions.

        Args:
            features (Tensor): Features of voxels, shape is NxC.
            coors (Tensor): Coordinates of voxels, shape is  Nx(1+NDim).

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels additionally.
        """
        features_ls = [features]

        # Find distance of x, y, and z from voxel center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 1].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 3].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        # Combine together feature decorations
        features = torch.cat(features_ls[::-1], dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = []
        for vfe in self.vfe_layers:
            features = vfe(features)
            point_feats.append(features)
        voxel_feats, voxel_coors = self.vfe_scatter(features, coors)

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        if self.return_point_feats:
            return voxel_feats, voxel_coors, point_feats
        return voxel_feats, voxel_coors
