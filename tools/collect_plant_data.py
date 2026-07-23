import argparse
import glob
from functools import partial
from multiprocessing import Pool
from os import path as osp

import mmengine
import numpy as np
from mmengine.utils import mkdir_or_exist

BASE_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Collect plant data.')
parser.add_argument('--name', help='provide name of the dataset.')
parser.add_argument(
    '--data-dir',
    type=str,
    help='provide plant data directory.',
    default='./data/plant/COS')
args = parser.parse_args()

plant_info = mmengine.load(osp.join(BASE_DIR, 'plant_info.json'))

assert args.name in plant_info, f'{args.name} Not Found in plant_info.json'
plant_type = plant_info[args.name]
features = plant_type['feature']


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def process_file(file_path, output_folder):
    print(f'Processing {file_path}')
    out_filename = osp.basename(file_path).split('.')[0]
    out_filename = osp.join(output_folder, out_filename)
    # TODO handle different point cloud file formats
    if file_path.endswith('.txt'):
        pc = np.loadtxt(file_path)
    # normalize point cloud (not necessary for real size point cloud)
    # pc[:, 0:3] = pc_normalize(pc[:, 0:3])

    # shift point cloud
    xyz_min = np.amin(pc, axis=0)[0:3]
    pc[:, 0:3] -= xyz_min

    np.save(f'{out_filename}_point.npy', pc[:, :features].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', pc[:, -2].astype(np.int64))
    np.save(f'{out_filename}_ins_label.npy', pc[:, -1].astype(np.int64))
    print(f'Exporting data to file:{out_filename} is done')


if __name__ == '__main__':
    data_folder = args.data_dir
    assert osp.exists(data_folder), f'{data_folder} Not Found'
    folder_name = osp.basename(data_folder)

    txt_folder = osp.join(data_folder, folder_name)
    assert osp.exists(txt_folder), f'{txt_folder} Not Found'
    output_folder = osp.join(data_folder, f'{folder_name}_data')
    mkdir_or_exist(output_folder)

    file_paths = glob.glob(osp.join(txt_folder, '*.txt'))

    with Pool() as pool:
        process_func = partial(process_file, output_folder=output_folder)
        pool.map(process_func, file_paths)
