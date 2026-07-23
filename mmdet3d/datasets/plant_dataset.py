# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Any, Callable, List, Optional, Tuple, Union

import mmengine
import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.visualization import palette_list
from .det3d_dataset import Det3DDataset
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class PlantDataset(Det3DDataset):
    r"""Plant Dataset for Instance Seg Task.
    This class is the inner dataset for Plant dataset.
    Modified from `S3DISSegDataset` class.
    `mmengine.datasets.dataset_wrappers.ConcatDataset` should be used.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        plant_type (str): Type of plant dataset.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for data. Defaults to
            dict(pts='points',
                 pts_instance_mask='instance_mask',
                 pts_semantic_mask='semantic_mask').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 plant_type: str,
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points',
                     pts_instance_mask='instance_mask',
                     pts_semantic_mask='semantic_mask'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs) -> None:
        if metainfo is not None:
            self.METAINFO = metainfo

        else:
            plant_info = mmengine.load('./tools/plant_info.json')[plant_type]
            num_ins_cls = len(plant_info['instance'])

            self.METAINFO = {
                'classes':
                tuple(plant_info['instance']),
                # the valid ids of segmentation annotations
                'seg_valid_class_ids':
                tuple(plant_info['label'][ins_item]
                      for ins_item in plant_info['instance']),
                'seg_all_class_ids':
                tuple(range(1,
                            len(plant_info['label']) + 1)),
                'palette':
                palette_list(num_colors=num_ins_cls)
            }
        # construct seg_label_mapping for semantic mask
        seg_max_cat_id = len(self.METAINFO['seg_all_class_ids'])
        seg_valid_cat_ids = self.METAINFO['seg_valid_class_ids']
        neg_label = len(seg_valid_cat_ids)
        seg_label_mapping = np.ones(
            seg_max_cat_id + 1, dtype=np.int64) * neg_label
        for cls_idx, cat_id in enumerate(seg_valid_cat_ids):
            seg_label_mapping[cat_id] = cls_idx
        self.seg_label_mapping = seg_label_mapping

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=self.METAINFO,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.metainfo['seg_label_mapping'] = self.seg_label_mapping
        assert 'use_camera' in self.modality and \
               'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['pts_instance_mask_path'] = osp.join(
            self.data_prefix.get('pts_instance_mask', ''),
            info['pts_instance_mask_path'])
        info['pts_semantic_mask_path'] = osp.join(
            self.data_prefix.get('pts_semantic_mask', ''),
            info['pts_semantic_mask_path'])

        info = super().parse_data_info(info)
        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.seg_label_mapping
        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = super().parse_ann_info(info)
        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        # to target box structure

        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return ann_info


@DATASETS.register_module()
class PlantInsSegDataset(PlantDataset):

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 plant_type: str,
                 data_prefix: dict = dict(
                     pts='points',
                     pts_instance_mask='instance_mask',
                     pts_semantic_mask='semantic_mask'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs) -> None:

        plant_info = mmengine.load('./tools/plant_info.json')[plant_type]
        num_sem_cls = len(plant_info['label'])

        self.METAINFO = {
            'classes': tuple(plant_info['label'].keys()),
            'palette': palette_list(num_colors=num_sem_cls),
            'seg_valid_class_ids': tuple(range(num_sem_cls)),
            'seg_all_class_ids': tuple(range(num_sem_cls))
        }

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            plant_type=plant_type,
            metainfo=self.METAINFO,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)


class _PlantSegDataset(Seg3DDataset):
    r"""Plant Dataset for Semantic Segmentation Task.
    This class is the inner dataset for Plant dataset.
    Modified from `S3DISSegDataset` class.

    Args:
        data_root (str, optional): Path of dataset root, Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        plant_type (str): Type of plant dataset.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points', pts_instance_mask='', pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 plant_type: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', pts_instance_mask='', pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[np.ndarray, str]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
        plant_info = mmengine.load('./tools/plant_info.json')[plant_type]
        num_sem_cls = len(plant_info['label'])

        self.METAINFO = {
            'classes': tuple(plant_info['label'].keys()),
            'palette': palette_list(num_colors=num_sem_cls),
            'seg_valid_class_ids': tuple(range(num_sem_cls)),
            'seg_all_class_ids': tuple(range(num_sem_cls))
        }

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)


@DATASETS.register_module()
class PlantSegDataset(_PlantSegDataset):
    r"""Plant Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the S3DIS Dataset.
    It wraps the provided datasets of different areas.
    We don't use `mmdet.datasets.dataset_wrappers.ConcatDataset` because we
    need to concat the `scene_idxs` of different areas.

    Please refer to the `google form <https://docs.google.com/forms/d/e/1FAIpQL
    ScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1>`_ for
    data downloading.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_files (List[str]): Path of several annotation files.
            Defaults to ''.
        plant_type (str): Type of plant dataset.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points', pts_instance_mask='', pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (List[np.ndarray] | List[str], optional): Precomputed index
            to load data. For scenes with many points, we may sample it
            several times. Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_files: List[str] = '',
                 plant_type: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', pts_instance_mask='', pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[List[np.ndarray],
                                            List[str]]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:

        # make sure that ann_files and scene_idxs have same length
        ann_files = self._check_ann_files(ann_files)
        scene_idxs = self._check_scene_idxs(scene_idxs, len(ann_files))

        # initialize some attributes as datasets[0]
        super().__init__(
            data_root=data_root,
            ann_file=ann_files[0],
            plant_type=plant_type,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs[0],
            test_mode=test_mode,
            **kwargs)

        datasets = [
            _PlantSegDataset(
                data_root=data_root,
                ann_file=ann_files[i],
                plant_type=plant_type,
                metainfo=metainfo,
                data_prefix=data_prefix,
                pipeline=pipeline,
                modality=modality,
                ignore_index=ignore_index,
                scene_idxs=scene_idxs[i],
                test_mode=test_mode,
                **kwargs) for i in range(len(ann_files))
        ]

        # data_list and scene_idxs need to be concat
        self.concat_data_list([dst.data_list for dst in datasets])

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def concat_data_list(self, data_lists: List[List[dict]]) -> None:
        """Concat data_list from several datasets to form self.data_list.

        Args:
            data_lists (List[List[dict]]): List of dict containing
                annotation information.
        """
        self.data_list = [
            data for data_list in data_lists for data in data_list
        ]

    @staticmethod
    def _duplicate_to_list(x: Any, num: int) -> list:
        """Repeat x `num` times to form a list."""
        return [x for _ in range(num)]

    def _check_ann_files(
            self, ann_file: Union[List[str], Tuple[str], str]) -> List[str]:
        """Make ann_files as list/tuple."""
        # ann_file could be str
        if not isinstance(ann_file, (list, tuple)):
            ann_file = self._duplicate_to_list(ann_file, 1)
        return ann_file

    def _check_scene_idxs(self, scene_idx: Union[str, List[Union[list, tuple,
                                                                 np.ndarray]],
                                                 List[str], None],
                          num: int) -> List[np.ndarray]:
        """Make scene_idxs as list/tuple."""
        return self._duplicate_to_list(scene_idx, num)
