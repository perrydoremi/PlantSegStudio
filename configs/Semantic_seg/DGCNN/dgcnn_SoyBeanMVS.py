_base_ = [
    '../../_base_/models/dgcnn.py',
    '../../_base_/schedules/seg-cosine-200e.py',
    '../../_base_/default_runtime.py'
]
# Dataset settings
plant_type = 'SoyBeanMVS'
num_points = 8192
block_size = 350
sample_rate = 1

use_color = False
load_dim = 6
use_dim = list(range(6)) if use_color else list(range(3))
num_classes = 3

train_area = list(range(1, 45)) + list(range(65, 86))
test_area = list(range(45, 65)) + list(range(86, 103))
backend_args = None
# Model settings
bs = 32
max_epoch = 250
val_interval = 4

# Point-based dataset config
class_names = ('mainstem', 'stem', 'leaf')
metainfo = dict(classes=class_names)
dataset_type = 'PlantSegDataset'
data_root = f'data/plant/{plant_type}/'

input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=use_color,
        load_dim=load_dim,
        use_dim=use_dim,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=block_size,
        ignore_index=len(class_names),
        use_normalized_coord=True,
        enlarge_size=0.2,
        min_unique_num=None),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=use_color,
        load_dim=load_dim,
        use_dim=use_dim,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=use_color,
        load_dim=load_dim,
        use_dim=use_dim,
        backend_args=backend_args),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=use_color,
        load_dim=load_dim,
        use_dim=use_dim,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.)
        ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=bs,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[
            f'{plant_type}_infos_{plant_type}{i}.pkl' for i in train_area
        ],
        plant_type=plant_type,
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=None,
        test_mode=False,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=bs,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[
            f'{plant_type}_infos_{plant_type}{i}.pkl' for i in test_area
        ],
        plant_type=plant_type,
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=None,
        test_mode=True,
        backend_args=backend_args))
val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

tta_model = dict(type='Seg3DTTAModel')

# model settings
model = dict(
    backbone=dict(in_channels=len(use_dim) + 3),  # [xyz, normalized_xyz]
    decode_head=dict(
        num_classes=num_classes,
        ignore_index=num_classes,
        loss_decode=dict(class_weight=None)),
    test_cfg=dict(
        num_points=num_points,
        block_size=block_size,
        sample_rate=sample_rate,
        use_normalized_coord=True,
        batch_size=bs))

# runtime settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,
        max_keep_ckpts=1,
        save_best=['miou'],
        rule='greater'))

train_cfg = dict(
    by_epoch=True, max_epochs=max_epoch, val_interval=val_interval)

# Wandb settings
vis_backends = [
    dict(type='LocalVisBackend')
]

visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Randomness
randomness = dict(seed=3407)
