# _base_ = [
#     'mmdet3d::_base_/default_runtime.py',
# ]
_base_ = [
    '../../_base_/default_runtime.py',
]
# custom_imports = dict(imports=['oneformer3d'])

# load_from = 'work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth'
load_from = 'work_dirs/tmp/oneformer3d_1xb2_s3dis-area-5.pth'
# load_from = 'work_dirs/tmp/oneformer3d_1xb4_scannet.pth'



# dataset settings
# dataset_type = 'S3DISSegDataset_'
dataset_type = 'TomatoWURDataset' ## ADDED BART
# data_root = 'data/s3dis/'
version = '0test-paper4-oneformer_instances'
version = '0-paper-2Dto3D_improved_updated_nodes'
data_root = f'/mnt/GARdata/TomatoWUR/ann_versions/{version}/json/' ## ADDED BART
ann_file_train = "train_oneformer.json"
ann_file_val = "test_oneformer.json"
# ann_file_val = "test_oneformer.json"


# Semantic class mapping: Dict for flexible class remapping and merging
# Keys = original class index, Values = target class index after mapping
# This allows reordering, merging multiple classes, or collapsing classes
# 
# Examples:
#   Identity (no change): {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
#   Merge pole into main stem: {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
#   Swap leaves and side stem: {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}
#   Merge all stems: {0: 0, 1: 0, 2: 1, 3: 0, 4: 2, 5: 3}
#
# Original classes: 0=main stem, 1=pole, 2=leaves, 3=side stem, 4=nodes, 5=unlabelled


orig_class_names = [
     "leaves", "main stem", "pole", "side stem", "unlabelled"]

class_mapping = {
    0: 2,  # leaves -> 
    1: 0,  # main stem
    2: 1,  # pole
    3: 3,  # side stem -> 3
    # 4: 4,  # nodes -> 4
    254: 4,  # unlabelled -> 5
}
class_names = [
     "main stem", "pole", "leaves", "side stem","unlabelled"]

# class_mapping = {
#     0: 0,  # leaves -> 
#     1: 1,  # main stem
#     2: 2,  # pole
#     3: 3,  # side stem -> 3
#     4: 4,  # nodes -> 4
#     254: 5,  # unlabelled -> 5
# }

# Build sem_mapping from class_mapping (maps original indices to target indices)
sem_mapping = [class_mapping.get(i, i) for i in range(len(class_mapping.keys()))]

##
# python3 tools/train.py configs/Instance_seg/Oneformer3d/oneformer3d_1xb2_tomatowur_class_mapping.py
##################################################### with nodes

ignore_index=[4] ## ignore index starts with counting from zero!!!!, so [0] means class_names[0] is ignored
# stuff = [0,1] ## wall or floor, objects without instances
# thing = [2,3,4,5]
stuff = [0,1,2] ## wall or floor, objects without instances
thing = [3,4]

pts_semantic_mask_column_name = "semantic"
pts_instance_mask_column_name = "stem_instances"


##################################################### plant parts
# class_names = [
#      "main stem", "pole", "leaves"] ## please ignore class names
# pts_semantic_mask_column_name = "semantic_oneformer_plant_parts"
# pts_instance_mask_column_name = "instances_oneformer_plant_parts"
# stuff = [0] ## wall or floor, objects without instances
# thing = [1] 
# ignore_index=[2] ## ignore index starts with counting from zero!!!!, so [0] means class_names[0] is ignored

#####################################################

max_epochs = 512
max_epochs = 100
num_workers= 16

# model settings
num_channels = 32 # same as scanmmet
num_channels = 64 # same as scs3didammet
num_instance_classes = len(class_names)
num_semantic_classes = len(class_names)

model = dict(
    type='S3DISOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,
    num_channels=num_channels,
    # voxel_size=0.05,
    voxel_size=0.002, ## added code BART
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='QueryDecoder',
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=100, ## ADDED CODE WAS 400
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
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
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=5.0),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=100,
        inst_score_thr=0.0,
        pan_score_thr=0.4,
        npoint_thr=300,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=stuff,
        thing_cls=thing,
        # stuff_cls=[0, 1, 2, 3, 4, 5, 6, 12], ## ADDED BART
        # thing_cls=[7, 8, 9, 10, 11]) ## ADDED BART
        )
        )
# model = dict(
#     type='ScanNetOneFormer3D',
#     data_preprocessor=dict(type='Det3DDataPreprocessor_'),
#     in_channels=6,
#     num_channels=num_channels,
#     voxel_size=0.02,
#     num_classes=num_instance_classes,
#     min_spatial_shape=128,
#     query_thr=0.5,
#     backbone=dict(
#         type='SpConvUNet',
#         num_planes=[num_channels * (i + 1) for i in range(5)],
#         return_blocks=True),
#     decoder=dict(
#         type='ScanNetQueryDecoder',
#         num_layers=6,
#         num_instance_queries=0,
#         num_semantic_queries=0,
#         num_instance_classes=num_instance_classes,
#         num_semantic_classes=num_semantic_classes,
#         num_semantic_linears=1,
#         in_channels=32,
#         d_model=256,
#         num_heads=8,
#         hidden_dim=1024,
#         dropout=0.0,
#         activation_fn='gelu',
#         iter_pred=True,
#         attn_mask=True,
#         fix_attention=True,
#         objectness_flag=False),
#     criterion=dict(
#         type='ScanNetUnifiedCriterion',
#         num_semantic_classes=num_semantic_classes,
#         sem_criterion=dict(
#             type='ScanNetSemanticCriterion',
#             ignore_index=num_semantic_classes,
#             loss_weight=0.2),
#         inst_criterion=dict(
#             type='InstanceCriterion',
#             matcher=dict(
#                 type='SparseMatcher',
#                 costs=[
#                     dict(type='QueryClassificationCost', weight=0.5),
#                     dict(type='MaskBCECost', weight=1.0),
#                     dict(type='MaskDiceCost', weight=1.0)],
#                 topk=1),
#             loss_weight=[0.5, 1.0, 1.0, 0.5],
#             num_classes=num_instance_classes,
#             non_object_weight=0.1,
#             fix_dice_loss_weight=True,
#             iter_matcher=True,
#             fix_mean_loss=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         topk_insts=600,
#         inst_score_thr=0.0,
#         pan_score_thr=0.5,
#         npoint_thr=100,
#         obj_normalization=True,
#         sp_score_thr=0.4,
#         nms=True,
#         matrix_nms_kernel='linear',
#         stuff_classes=stuff))


data_prefix = dict(
    pts='',
    pts_instance_mask='',
    pts_semantic_mask='')

# train_area = [1, 2, 3, 4, 6]
# test_area = 5

train_pipeline = [
    dict(
        type='LoadPointsFromFileCSV',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=9,
        use_dim=[0, 1, 2, 3, 4, 5],
        ),
    dict(
        type='LoadAnnotations3D_CSV',
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
        pts_semantic_mask_column_name=pts_semantic_mask_column_name,
        pts_instance_mask_column_name=pts_instance_mask_column_name,
        dataset_type="tomatowur"
        ),
    dict(
        type='RemapClassLabels',
        class_mapping=class_mapping,
        ),
    dict(
        type='PointSample_',
        num_points=100000),
    dict(type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    # dict(type='PointSegClassMapping'),
    # dict(
    #     type='PointSample_',
    #     num_points=100000),
    # dict(type='PointInstClassMapping_',
    #     num_classes=num_instance_classes),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    # dict(
    #     type='CustomAddSuperPointAnnotations',
    #     num_classes=num_semantic_classes,
    #     stuff_classes=stuff,
    #     merge_non_stuff_cls=False),
    # dict(
    #     type='ElasticTransfrom',
    #     gran=[6, 20],
    #     mag=[40, 160],
    #     voxel_size=0.02,
    #     p=0.5),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d',
            'pts_semantic_mask', 'pts_instance_mask'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileCSV',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=9,
        use_dim=[0, 1, 2, 3, 4, 5],
        ),
    dict(
        type='LoadAnnotations3D_CSV',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
        pts_semantic_mask_column_name=pts_semantic_mask_column_name,
        pts_instance_mask_column_name=pts_instance_mask_column_name,
        dataset_type="tomatowur"
        ),
    dict(
        type='RemapClassLabels',
        class_mapping=class_mapping,
        ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

# run settings
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=3,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#             type='ConcatDataset',
#             datasets=([
#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file=ann_file_train,
#                     pipeline=train_pipeline,
#                     filter_empty_gt=True,
#                     data_prefix=data_prefix,
#                     box_type_3d='Depth',
#                     backend_args=None) for i in train_area])))

train_dataloader = dict(
    batch_size=4,
    num_workers=num_workers,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='scannet_oneformer3d_infos_train.pkl',
        # ann_file=ann_file_folder+'scannet_infos_train.pkl',
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        ignore_index=ignore_index,
        metainfo={"classes": tuple(class_names)},
        scene_idxs=None,
        test_mode=False)
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    # persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True,
        ignore_index=ignore_index,
        metainfo={"classes": tuple(class_names)},
        # box_type_3d='Depth',
        # backend_args=None
        ))
test_dataloader = val_dataloader


label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    # ignore_index=[num_semantic_classes],
    ignore_index=ignore_index, ## ADDED BART
    classes=class_names,
    dataset_name='TomatoWUR')

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=stuff,
    thing_class_inds=thing,
    # stuff_class_inds=[0, 1, 2, 3, 4, 5, 6, 12], ## ADDED BART
    # thing_class_inds=[7, 8, 9, 10, 11], ## ADDED BART
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=max_epochs, power=0.9)

custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    dict(
        type='SaveOutputHook',
        save_dir='./work_dirs/test_outputs',
        score_threshold=0.5,
        save_semantic_seg=True,
        save_panoptic_seg=True,
        save_instances=True)
]
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
