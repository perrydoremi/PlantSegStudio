_base_ = [
    '../../_base_/default_runtime.py',
]

# Dataset and work dir config:
dataset_type = 'PlantInsSegDataset'
plant_type = 'SoyBeanMVSRS'
class_names = ['mainstem', 'stem', 'leaf', 'unlabeled']
data_root = f'data/plant/{plant_type}'
data_prefix = dict(
    pts='points', pts_instance_mask='instance_mask', pts_semantic_mask='semantic_mask')

train_area = list(range(1, 45)) + list(range(65, 86))
test_area = list(range(45, 65)) + list(range(86, 103))

# Model config:
use_color = False
load_dim = 6
use_dim = list(range(3))
voxel_scale = 2
num_points = 180000
# num_points = 250000

channels = 32
num_blocks = 7
semantic_classes = 3
instance_classes = 3
sem2ins_classes = []
semantic_only = False
ignore_label = -100

grouping_radius = 0.3
grouping_cfg = dict(
    score_thr=0.2,
    radius=grouping_radius,
    mean_active=100,
    class_numpoint_mean=[-1., 9652, 8992],
    npoint_thr=0.05,
    ignore_classes=[0])

instance_voxel_cfg = dict(scale=voxel_scale, spatial_shape=20)

model_train_cfg = dict(max_proposal_num=250, pos_iou_thr=0.5)

model_test_cfg = dict(
    cls_score_thr=0.001, mask_score_thr=-0.5, min_npoint=100, eval_tasks=['semantic', 'instance'])

fixed_modules = []

label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat, ignore_index=[semantic_classes], classes=class_names, dataset_name='Plant')
sem_mapping = [0, 1, 2]

# Train/Eval/Test and optimization config
batch_size = 4
max_epoch = 512
val_interval = 16
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=val_interval)  # oringin 16
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=max_epoch, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=val_interval, max_keep_ckpts=1, save_best=['all_ap', 'miou'], rule='greater'))

# load_from = ''

# Details
model = dict(
    type='SoftGroup',
    data_preprocessor=dict(
        type='SoftGroupDataPreprocessor',
        voxel_cfg=dict(
            scale=voxel_scale, spatial_shape=[128, 512], max_npoint=num_points, min_point=5000)),
    in_channels=len(use_dim),
    channels=channels,
    num_blocks=num_blocks,
    semantic_classes=semantic_classes,
    instance_classes=instance_classes,
    semantic_only=semantic_only,
    sem2ins_classes=sem2ins_classes,
    ignore_label=ignore_label,
    grouping_cfg=grouping_cfg,
    instance_voxel_cfg=instance_voxel_cfg,
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg,
    fixed_modules=fixed_modules,
    backbone=dict(type='SoftGroupUNet'))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=use_color,
        load_dim=load_dim,
        use_dim=use_dim),
    dict(
        type='LoadAnnotations3D',
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='PointSample_', num_points=num_points),
    dict(type='PointInstClassMapping_', num_classes=instance_classes),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='InstanceInfoSoftGroup'),
    dict(
        type='Pack3DSoftGroupInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'pt_offset_labels',
            'instance_num', 'instance_pointnum', 'instance_cls'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=use_color,
        load_dim=load_dim,
        use_dim=use_dim),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0)
        ]),
    dict(type='InstanceInfoSoftGroup'),
    dict(
        type='Pack3DSoftGroupInputs_',
        keys=['points', 'pt_offset_labels', 'instance_num', 'instance_pointnum', 'instance_cls'])
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=([
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=f'{plant_type}_infos_{plant_type}{i}.pkl',
                plant_type=plant_type,
                pipeline=train_pipeline,
                filter_empty_gt=True,
                data_prefix=data_prefix,
                box_type_3d='Depth',
                backend_args=None) for i in train_area
        ])))

val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=([
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=f'{plant_type}_infos_{plant_type}{i}.pkl',
                plant_type=plant_type,
                pipeline=test_pipeline,
                test_mode=True,
                data_prefix=data_prefix,
                box_type_3d='Depth',
                backend_args=None) for i in test_area
        ])))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='SoftGroupSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=[1, 2],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

# Wandb settings
vis_backends = [
    dict(type='LocalVisBackend')
]

visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Randomness
randomness = dict(seed=3407)
