import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np


class PLANTData(object):
    """Plant data.

    Generate plant infos for plant_converter.

    Args:
        root_path (str): Root path of the raw data.
    """

    def __init__(self, root_path, info_prefix, plant_info):
        self.root_dir = root_path
        self.info_prefix = info_prefix
        plant_infos = mmengine.load(plant_info)
        assert info_prefix in [plant for plant in plant_infos.keys()], \
            f'unsupported plant dataset {info_prefix}'
        self.num_features = plant_infos[info_prefix]['feature']
        self.labels = plant_infos[info_prefix]['label']
        self.instances = plant_infos[info_prefix]['instance']

        # classes for detection
        self.cat_ids = np.array(
            [self.labels[ins_item] for ins_item in self.instances])
        self.cat_ids2class = {
            cat_id: i
            for i, cat_id in enumerate(list(self.cat_ids))
        }

        self.sample_id_list = os.listdir(osp.join(root_path, info_prefix))
        self.plant_npydatadir = f'{info_prefix}_data'

    def __len__(self):
        return len(self.sample_id_list)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_pc(sample_idx):
            print(f'{self.info_prefix} generate info of: {sample_idx}')
            info = dict()
            pc_info = {
                'num_features': self.num_features,
                'lidar_idx': f'{sample_idx}'
            }
            info['label'] = self.labels
            info['meta_instance'] = self.instances
            info['point_cloud'] = pc_info
            info['dataset'] = self.info_prefix
            pts_filename = osp.join(self.root_dir, self.plant_npydatadir,
                                    f'{sample_idx}_point.npy')

            pts_instance_mask_path = osp.join(self.root_dir,
                                              self.plant_npydatadir,
                                              f'{sample_idx}_ins_label.npy')

            pts_semantic_mask_path = osp.join(self.root_dir,
                                              self.plant_npydatadir,
                                              f'{sample_idx}_sem_label.npy')

            points = np.load(pts_filename).astype(np.float32)
            pts_instance_mask = np.load(pts_instance_mask_path).astype(
                np.int64)
            pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                np.int64)

            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask'))
            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask'))

            points.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))

            pts_instance_mask.tofile(
                osp.join(self.root_dir, 'instance_mask', f'{sample_idx}.bin'))

            pts_semantic_mask.tofile(
                osp.join(self.root_dir, 'semantic_mask', f'{sample_idx}.bin'))

            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')
            info['pts_instance_mask_path'] = osp.join('instance_mask',
                                                      f'{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join('semantic_mask',
                                                      f'{sample_idx}.bin')

            info['annos'] = self.get_bboxes(points, pts_instance_mask,
                                            pts_semantic_mask)

            return info

        def splitlist(sample_id):
            sample_id = sample_id.split('.')[0]
            return sample_id

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list

        print('sample_id_list=', sample_id_list)
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_pc,
                                 list(map(splitlist, sample_id_list)))
        return list(infos)

    def get_bboxes(self, points, pts_instance_mask, pts_semantic_mask):
        """Convert instance masks to axis-aligned bounding boxes.

        Args:
            points (np.array): Scene points of shape (n, 6).
            pts_instance_mask (np.ndarray): Instance labels of shape (n,).
            pts_semantic_mask (np.ndarray): Semantic labels of shape (n,).

        Returns:
            dict: A dict containing detection infos with following keys:

                - gt_boxes_upright_depth (np.ndarray): Bounding boxes
                    of shape (n, 6)
                - class (np.ndarray): Box labels of shape (n,)
                - gt_num (int): Number of boxes.      (n)
        """
        bboxes, labels = [], []
        for i in range(1, pts_instance_mask.max() + 1):
            ids = pts_instance_mask == i
            assert pts_semantic_mask[ids].min() == pts_semantic_mask[ids].max()
            label = pts_semantic_mask[ids][0]
            if label in self.cat_ids2class:
                labels.append(self.cat_ids2class[pts_semantic_mask[ids][0]])
                pts = points[:, :3][ids]
                min_pts = pts.min(axis=0)
                max_pts = pts.max(axis=0)
                locations = (min_pts + max_pts) / 2
                dimensions = max_pts - min_pts
                bboxes.append(np.concatenate((locations, dimensions)))
        annotation = dict()
        # follow ScanNet and SUN RGB-D keys
        annotation['gt_boxes_upright_depth'] = np.array(bboxes)
        annotation['class'] = np.array(labels)
        annotation['gt_num'] = len(labels)
        return annotation


class PLANTSegData(object):
    """Plant dataset used to generate infos for semantic segmentation task.

    Args:
        data_root (str): Root path of the raw data.
        ann_file (str): The generated scannet infos.
        split (str): Set split type of the data. Default: 'train'.
        num_points (int): Number of points in each data input. Default: 8192.
        label_weight_func (function): Function to compute the label weight.
            Default: None.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 num_points=200000,
                 label_weight_func=None):
        self.data_root = data_root
        self.data_infos = mmengine.load(ann_file)
        # num_points setting influencing get_name_label_weight
        self.num_points = num_points
        self.label = self.data_infos[0]['label']

        self.all_ids = np.arange(len(self.label))
        self.cat_ids = np.array(list(self.label.values()))
        self.ignore_index = len(self.cat_ids)
        self.cat_id2class = np.ones(
            (self.all_ids.shape[0], ), dtype=np.int64) * self.ignore_index
        for i, cat_id in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i

        # label weighting function is taken from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        self.label_weight_func = (lambda x: 1.0 / np.log(1.2 + x)) if \
            label_weight_func is None else label_weight_func

    def get_seg_infos(self):
        pcname, scene_idxs, label_weight = self.get_name_label_weight()
        save_folder = osp.join(self.data_root, 'seg_info')
        mmengine.mkdir_or_exist(save_folder)
        np.save(
            osp.join(save_folder, f'{pcname}_resampled_scene_idxs.npy'),
            scene_idxs)
        np.save(
            osp.join(save_folder, f'{pcname}_label_weight.npy'), label_weight)
        print(f'{pcname} resampled scene index and label weight saved')

    def _convert_to_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_name_label_weight(self):
        """Compute scene_idxs for data sampling and label weight for loss \
        calculation.

        We sample more times for scenes with more points. Label_weight is
        inversely proportional to number of class points.
        """
        num_classes = len(self.cat_ids)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1, ))

        for data_info in self.data_infos:
            label = self._convert_to_label(
                osp.join(self.data_root, data_info['pts_semantic_mask_path']))
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, range(num_classes + 2))
            label_weight += class_count
            pcname = data_info['point_cloud']['lidar_idx']

        # repeat scene_idx for num_scene_point // num_sample_point times

        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)

        # calculate label weight, adopted from PointNet++
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)

        return pcname, scene_idxs, label_weight
