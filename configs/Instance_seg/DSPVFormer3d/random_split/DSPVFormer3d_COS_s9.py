_base_ = [
    '../../../_base_/default_runtime.py',
]

# Dataset and work dir config:
dataset = 'Plant'
dataset_type = 'PlantInsSegDataset'
plant_type = 'COS'
class_names = ['trunk', 'branch', 'unlabeled']
data_root = f'data/plant/{plant_type}'
data_prefix = dict(
    pts='points', pts_instance_mask='instance_mask', pts_semantic_mask='semantic_mask')

split = 's9'
train_area=[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 32, 33, 34, 35, 36, 39, 41, 42, 43, 44, 45, 49, 50, 52, 53, 56, 58, 59, 61, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 98]
test_area=[5, 6, 13, 16, 21, 26, 28, 31, 37, 38, 40, 46, 47, 48, 51, 54, 55, 57, 60, 62, 64, 70, 78, 92, 96, 97]

# Model config:
use_color = False
load_dim = 3
use_dim = list(range(3))
point_cloud_range = [0, 0, 0, 1.89, 1.89, 3.31]
voxel_size = 0.005
num_points = 180000

num_channels = 64  # NOT used for MinkUNetBackboneV2 or SPVCNNBackbone
num_instance_classes = 2
num_semantic_classes = 2
num_instance_queries = 150

model_train_cfg = dict()
model_test_cfg = dict(
    topk_insts=200,
    inst_score_thr=0.0,
    pan_score_thr=0.4,
    npoint_thr=50,
    obj_normalization=True,
    obj_normalization_thr=0.01,
    sp_score_thr=0.15,
    nms=True,
    matrix_nms_kernel='linear',
    num_sem_cls=num_semantic_classes,
    stuff_cls=[0],
    thing_cls=[1])

label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name=dataset)
sem_mapping = [0, 1]

# Train/Eval/Test and optimization config
batch_size = 4
max_epochs = 512
val_interval = 16
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=512, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=val_interval, max_keep_ckpts=1, save_best=['all_ap', 'miou'], rule='greater'))

# load_from = 'work_dirs/DSPVFormer3d_b4_voxel-0005_np180k_instqry150-200_Helios006G4H_vfe64_64_scatter_v2_dynamic_quantize_D32_Q64_cat_1e-5_m01_quantize_linear_w_ini/DSPVFormer_BN_wInit_best_all_ap_epoch_400.pth'

wandb_run_name = (f'{plant_type}-DSPVFormer_scatter_v2_DQcat_'
                  f'VoxelSize-{voxel_size}_NP-{num_points/1000}k'
                  f'InstQry-{num_instance_queries}_{num_instance_queries+50}_'
                  f'NumPointThr-{50}_'
                  f'Color_-{use_color}_LoadDim-{load_dim}_'
                  f'Batch-{batch_size}_Ep-{max_epochs}_D32_32_Q64_96_1e-5_m01_'
                  f'QuantizeLinearV1_noInit_full_{split}_5')

work_dir = 'work_dirs/DSPVFormer3d/random_split/COS_s9'

# Details
model = dict(
    type='DynamicOneFormer3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic_quantize',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=[voxel_size, voxel_size, voxel_size],
            max_voxels=(-1, -1))),
    in_channels=len(use_dim),
    num_channels=num_channels,
    voxel_size=voxel_size,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=len(use_dim),
        quantize_channel=64,  # qc
        feat_channels=[32, 32],
        with_distance=False,  #p
        voxel_size=[voxel_size, voxel_size, voxel_size],
        with_cluster_center=True,  #cc
        with_voxel_center=True,  #vc
        with_quantize_cat=True,  
        init_weights=False,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)   
    ),
    backbone=dict(
        type='SPVCNNBackbone',
        in_channels=96,
        num_stages=4,
        base_channels=96,
        encoder_channels=[96, 96, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='torchsparse',
        drop_ratio=0.3,
        return_sparse=True),
    decoder=dict(
        type='QueryDecoder',
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=num_instance_queries,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=96,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='S3DISUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(type='S3DISSemanticCriterion', loss_weight=5.0),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)
                ]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg)

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
    dict(type='PointInstClassMapping_', num_classes=num_instance_classes),
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
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
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
    dict(type='Pack3DDetInputs_', keys=['points'])
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
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=[1],
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
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': f'DSPVFormer-{plant_type}_full_formal_balance_datasplit',
            'group': 'DSPVFormer',
            'name': wandb_run_name
        })
]

visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Randomness
# randomness = dict(seed=3407)
randomness = dict(seed=3407)
