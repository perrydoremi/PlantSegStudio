# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from typing import Optional, Sequence

from mmdet3d.apis.inferencers.inference_utils import local_inference
from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample


@HOOKS.register_module()
class Det3DInferenceHook(Hook):
    """Detection Inference Hook. Used to inference and save prediction results
    without trigger visualization.

    After the testing iteration, the hook will save the
    prediction results to the specified directory.

    Args:
        vis_task (str): The visualization task name. Default: 'lidar_inst_seg'.
        save_dir (str, optional): The directory to save the prediction results.
    """

    def __init__(self, vis_task: str = 'lidar_inst_seg', save_dir: Optional[str] = None):
        self.vis_task = vis_task
        self.save_dir = save_dir

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.save_dir is not None:
            mkdir_or_exist(self.save_dir)

        for data_sample in outputs:

            assert 'img_path' in data_sample or 'lidar_path' in data_sample, \
                "'data_sample' must contain 'img_path' or 'lidar_path'"

            if self.vis_task in [
                    'lidar_det',
                    'lidar_seg',
                    'lidar_inst_seg',
            ]:
                assert 'lidar_path' in data_sample, \
                    'lidar_path is not in data_sample'
            # if self.save_dir is not None:
            #     file_save_path = osp.basename(
            #         data_sample.lidar_path).split('.')[0] + '.txt'
            #     file_save_path = osp.join(self.save_dir, file_save_path)
            local_inference(data_sample, self.vis_task, self.save_dir)
