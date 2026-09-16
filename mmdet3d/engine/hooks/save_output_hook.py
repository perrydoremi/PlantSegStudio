# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch
import open3d as o3d
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SaveOutputHook(Hook):
    """Hook to save test outputs as point clouds.
    
    Args:
        save_dir (str): Directory to save outputs. Default: './work_dirs/test_outputs'
        score_threshold (float): Confidence threshold for instances. Default: 0.5
        save_semantic_seg (bool): Whether to save semantic segmentation. Default: True
        save_panoptic_seg (bool): Whether to save panoptic segmentation. Default: True
        save_instances (bool): Whether to save individual instances. Default: True
    """
    
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -np.inf, 'less': np.inf}
    
    def __init__(self,
                 save_dir='./work_dirs/test_outputs',
                 score_threshold=0.5,
                 save_semantic_seg=True,
                 save_panoptic_seg=True,
                 save_instances=True):
        self.save_dir = save_dir
        self.score_threshold = score_threshold
        self.save_semantic_seg = save_semantic_seg
        self.save_panoptic_seg = save_panoptic_seg
        self.save_instances = save_instances
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    def _get_color_map(self, num_classes):
        """Generate a color map for semantic classes."""
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab20', num_classes)
        return np.array([cmap(i)[:3] for i in range(num_classes)])
    
    def _save_point_cloud(self, points, file_path):
        """Save points as PLY file."""
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        points = np.asarray(points, dtype=np.float32)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Set colors if available
        if points.shape[1] >= 6:
            colors = points[:, 3:6] / 255.0 if points[:, 3:6].max() > 1 else points[:, 3:6]
            pc.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(file_path, pc)
    
    def _process_output(self, batch_data_samples, batch_inputs_dict):
        """Process and save outputs from test."""
        for i, data_sample in enumerate(batch_data_samples):
            if not hasattr(data_sample, 'pred_pts_seg'):
                continue
            
            pred_pts_seg = data_sample.pred_pts_seg
            instance_labels = pred_pts_seg.instance_labels  # (num_instances,)
            instance_scores = pred_pts_seg.instance_scores  # (num_instances,)
            pts_instance_mask = pred_pts_seg.pts_instance_mask  # (num_instances, num_points)
            
            # Get input points from data_batch
            if isinstance(batch_inputs_dict["inputs"], dict) and 'points' in batch_inputs_dict["inputs"]:
                input_points = batch_inputs_dict['inputs']["points"][i] if isinstance(batch_inputs_dict['inputs']["points"], list) else batch_inputs_dict['inputs']['points']
                # input_points= input_points[:,:3] #3 only xyz
            else:
                continue
            
            # Extract name from data_sample
            if hasattr(data_sample, 'lidar_path'):
                point_name = data_sample.lidar_path.split('/')[-1].split('.')[0]
            elif hasattr(data_sample, 'img_path'):
                point_name = data_sample.img_path.split('/')[-1].split('.')[0]
            else:
                point_name = f'sample_{i}'
            
            # Create directory for this sample
            sample_dir = os.path.join(self.save_dir, point_name)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir, exist_ok=True)
            
            # Save input point cloud
            # input_pc_path = os.path.join(sample_dir, f'{point_name}_input.ply')
            # self._save_point_cloud(input_points, input_pc_path)
            
            # Prepare semantic segmentation visualization if available
            if self.save_semantic_seg and hasattr(pred_pts_seg, 'pts_semantic_mask'):
                seg_colour = input_points.clone() if isinstance(input_points, torch.Tensor) else torch.from_numpy(input_points.copy())
                pts_semantic_mask = pred_pts_seg.pts_semantic_mask[0]  # (num_points,)
                
                num_classes = int(pts_semantic_mask.max().item()) + 1
                color_map = self._get_color_map(num_classes)
                
                if isinstance(pts_semantic_mask, torch.Tensor):
                    pts_semantic_mask = pts_semantic_mask.cpu().numpy()
                
                seg_colour[:, 3:6] = torch.from_numpy(color_map[pts_semantic_mask] * 255).float()
                
                sem_seg_path = os.path.join(sample_dir, f'{point_name}_semantic.ply')
                self._save_point_cloud(seg_colour, sem_seg_path)
            
            # Save panoptic segmentation visualization if available
            if self.save_panoptic_seg and hasattr(pred_pts_seg, 'pts_semantic_mask'):
                seg_colour = input_points.clone() if isinstance(input_points, torch.Tensor) else torch.from_numpy(input_points.copy())
                pts_semantic_mask = pred_pts_seg.pts_semantic_mask[1]  #
                
                num_classes = int(pts_semantic_mask.max().item()) + 1
                color_map = self._get_color_map(num_classes)
                
                if isinstance(pts_semantic_mask, torch.Tensor):
                    pts_semantic_mask = pts_semantic_mask.cpu().numpy()
                
                seg_colour[:, 3:6] = torch.from_numpy(color_map[pts_semantic_mask] * 255).float()
                
                sem_seg_path = os.path.join(sample_dir, f'{point_name}_panoptic.ply')
                self._save_point_cloud(seg_colour, sem_seg_path)
            
            
            # Save individual instances
            if self.save_instances:
                instance_count = {}
                for inst_idx in range(len(instance_scores)):
                    if instance_scores[inst_idx] >= self.score_threshold:
                        seg_colour = input_points.clone() if isinstance(input_points, torch.Tensor) else torch.from_numpy(input_points.copy())
                        # seg_colour[:,3:]=[0,0,0]

                        label = instance_labels[inst_idx].item()
                        
                        if label not in instance_count:
                            instance_count[label] = 0
                        instance_count[label] += 1
                        
                        instance_mask = pts_instance_mask[0][inst_idx].astype(bool)
                        instance_points = seg_colour[instance_mask]

                        instance_path = os.path.join(
                            sample_dir, 
                            f'{point_name}_class_{label}_inst_{instance_count[label]}.ply'
                        )
                        self._save_point_cloud(instance_points, instance_path)
        
    
    def after_test_epoch(self, runner) -> None:
        """Called after every testing epoch."""
        pass
    
    def after_test_iter(self, runner, batch_idx: int, data_batch, outputs) -> None:
        """Called after every testing iteration."""
        # Extract batch data - outputs is a list of data samples
        if isinstance(outputs, list):
            batch_data_samples = outputs
        else:
            batch_data_samples = outputs if hasattr(outputs, '__iter__') else [outputs]
        
        batch_inputs_dict = data_batch
        
        # Process and save outputs
        self._process_output(batch_data_samples, batch_inputs_dict)
