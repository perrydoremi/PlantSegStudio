# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.dataset_converters import plant_converter as converter
from tools.dataset_converters.update_infos_to_v2 import update_plant_pkl_infos


def plant_data_prep(root_path, info_prefix, out_dir, plant_info, workers,
                    num_data):
    """Prepare the info file for rapeseed plant dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        plant_info (str): Path of plant info json.
        workers (int): Number of threads to be used.
        num_data (int): Number of data.
    """
    converter.create_plant_info_file(
        root_path,
        info_prefix,
        out_dir,
        plant_info=plant_info,
        workers=workers)
    for id_ in range(1, num_data + 1):
        filename = osp.join(out_dir,
                            f'{info_prefix}_infos_{info_prefix}{id_}.pkl')
        update_plant_pkl_infos(out_dir=out_dir, pkl_path=filename)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--plant-info',
    type=str,
    default='./tools/plant_info.json',
    help='plant info path')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument('--num-data', type=int, default=10, help='number of data')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
parser.add_argument(
    '--skip-cam_instances-infos',
    action='store_true',
    help='''Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--skip-saving-sensor-data',
    action='store_true',
    help='''Whether to skip saving image and lidar.
        Only used when dataset is Waymo!''')
args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    plant_data_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        out_dir=args.out_dir,
        plant_info=args.plant_info,
        workers=args.workers,
        num_data=args.num_data,
    )
