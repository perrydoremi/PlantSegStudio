# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
from mmengine.fileio import get
from mmengine.utils import mkdir_or_exist
from mmengine.visualization.utils import check_type, tensor2ndarray

from mmdet3d.models.utils.softgroup_utils import rle_decode
from mmdet3d.structures import Det3DDataSample


def local_inference(data_sample: Det3DDataSample, vis_task: str, save_dir: str) -> None:
    """Inference and save prediction results without trigger visualization.

    Args:
        data_sample (Det3DDataSample): A data sample that contains predictions.
        vis_task (str): The visualization task name.
        save_dir (str): The directory to save the prediction results.
    """
    lidar_path = data_sample.lidar_path
    file_name = lidar_path.split('/')[-1].replace('.bin', '.txt')
    num_pts_feats = data_sample.num_pts_feats
    pts_bytes = get(lidar_path, backend_args=None)
    points = np.frombuffer(pts_bytes, dtype=np.float32)
    points = tensor2ndarray(points.reshape(-1, num_pts_feats))

    if 'pred_pts_seg' in data_sample and vis_task == 'lidar_inst_seg':
        inst_pred_dir = osp.join(save_dir, 'inst_pred')
        mkdir_or_exist(inst_pred_dir)
        if 'point_wise_results' in data_sample.pred_pts_seg:  # softgroup results
            point_wise_results = data_sample.pred_pts_seg['point_wise_results']
            # save instance segmentation results
            pred_insts = data_sample.pred_pts_seg['pred_instances']
            sem_pred = point_wise_results['semantic_preds']
            instance_ids = np.zeros(points.shape[0], dtype=int)
            for idx, inst in enumerate(pred_insts, start=1):
                inst_mask = rle_decode(inst['pred_mask'])
                instance_ids[inst_mask == 1] = idx
            inst_concat = np.concatenate([points, sem_pred.reshape(-1, 1), instance_ids.reshape(-1, 1)], axis=1)
            # save offset points
            offset_pred = point_wise_results['offset_preds'] + \
                point_wise_results['coords_float']
            offset_gt = point_wise_results['offset_labels'] + \
                point_wise_results['coords_float']
            offset_pred_clr = np.concatenate(
                [offset_pred, sem_pred.reshape(-1, 1)], axis=1)

            offset_pred_dir = osp.join(save_dir, 'offset_pred')
            offset_gt_dir = osp.join(save_dir, 'offset_gt')
            offset_pred_clr_dir = osp.join(save_dir, 'offset_pred_color')
            mkdir_or_exist(offset_pred_dir)
            mkdir_or_exist(offset_gt_dir)
            mkdir_or_exist(offset_pred_clr_dir)

            # Save the files
            np.savetxt(osp.join(offset_pred_dir, file_name), offset_pred, fmt='%.6f', delimiter=' ')
            np.savetxt(osp.join(offset_gt_dir, file_name), offset_gt, fmt='%.6f', delimiter=' ')
            np.savetxt(osp.join(inst_pred_dir, file_name), inst_concat, fmt='%.6f', delimiter=' ')
            np.savetxt(osp.join(offset_pred_clr_dir, file_name), offset_pred_clr, fmt='%.6f', delimiter=' ')
        else:
            pts_ins_seg = tensor2ndarray(data_sample.pred_pts_seg.pts_instance_mask)
            pts_sem_seg = tensor2ndarray(data_sample.pred_pts_seg.pts_semantic_mask)
            check_type('instance predictions', pts_ins_seg, (np.ndarray, list))
            if isinstance(pts_ins_seg, list):
                inst_pred = pts_ins_seg[1].reshape(-1, 1)
                sem_pred = pts_sem_seg[0].reshape(-1, 1)
                inst_concat = np.concatenate([points, sem_pred, inst_pred], axis=1)
                np.savetxt(
                    osp.join(inst_pred_dir, file_name), inst_concat, fmt='%.6f', delimiter=' ')
            if isinstance(pts_ins_seg, np.ndarray):
                instance_scores = tensor2ndarray(data_sample.pred_pts_seg.instance_scores)
                instance_points_list = []
                instance_instance_id_list = []
                instance_counter = 1
                for i in range(len(instance_scores)):
                    if instance_scores[i] >= 0.1:
                        instance_mask = pts_ins_seg[i].astype(bool)
                        instance_points = points[instance_mask]
                        instance_ids = np.full((instance_points.shape[0], 1), instance_counter)

                        instance_instance_id_list.append(instance_ids)
                        instance_points_list.append(instance_points)

                        instance_counter += 1

                instance_points = np.vstack(instance_points_list)
                instance_ids = np.vstack(instance_instance_id_list)
                inst_concat = np.concatenate([instance_points, instance_ids], axis=1)
                np.savetxt(
                    osp.join(inst_pred_dir, file_name), inst_concat, fmt='%.6f', delimiter=' ')
    if 'pred_pts_seg' in data_sample and vis_task == 'lidar_seg':
        sem_pred_dir = osp.join(save_dir, 'semantic_pred')
        mkdir_or_exist(sem_pred_dir)
        if 'point_wise_results' in data_sample.pred_pts_seg:  # softgroup results
            point_wise_results = data_sample.pred_pts_seg['point_wise_results']
            sem_pred = point_wise_results['semantic_preds']
            sem_concat = np.concatenate([points, sem_pred.reshape(-1, 1)], axis=1)
            np.savetxt(osp.join(sem_pred_dir, file_name), sem_concat, fmt='%.6f', delimiter=' ')
        else:
            pts_sem_seg = tensor2ndarray(data_sample.pred_pts_seg.pts_semantic_mask)
            if isinstance(pts_sem_seg, list):
                sem_pred = pts_sem_seg[0].reshape(-1, 1)
                sem_concat = np.concatenate([points, sem_pred], axis=1)
                np.savetxt(osp.join(sem_pred_dir, file_name), sem_concat, fmt='%.6f', delimiter=' ')
            if isinstance(pts_sem_seg, np.ndarray):
                sem_pred = pts_sem_seg.reshape(-1, 1)
                sem_concat = np.concatenate([points, sem_pred], axis=1)
                np.savetxt(osp.join(sem_pred_dir, file_name), sem_concat, fmt='%.6f', delimiter=' ')
