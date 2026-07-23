# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmengine
import numpy as np

from tools.dataset_converters.plant_data_utils import PLANTData, PLANTSegData


def create_plant_info_file(data_path,
                           pkl_prefix='sunrgbd',
                           save_path=None,
                           workers=4,
                           plant_info=None):
    """Create plant information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        workers (int, optional): Number of threads to be used. Default: 4.
        plant_info (str): Plant information json path. Default: None.
    """
    assert os.path.exists(data_path)
    assert plant_info, 'plant dict is required'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    dataset = PLANTData(
        root_path=data_path, info_prefix=pkl_prefix, plant_info=plant_info)
    info = dataset.get_infos(
        num_workers=workers,
        has_label=True)  # info:list[dict] list include all plant data
    for info_one in info:  # info_one is a dict
        list_one = []
        lidar_idx = info_one['point_cloud']['lidar_idx']
        filename = os.path.join(save_path,
                                f'{pkl_prefix}_infos_{lidar_idx}.pkl')
        list_one.append(info_one)
        mmengine.dump(list_one, filename, 'pkl')
        print(f'{pkl_prefix} info {lidar_idx} file is saved to {filename}'
              )  # list[dict] for pkl files
        seg_dataset = PLANTSegData(
            data_root=data_path,
            ann_file=filename,
            label_weight_func=lambda x: 1.0 / np.log(1.2 + x))  #
        seg_dataset.get_seg_infos()
