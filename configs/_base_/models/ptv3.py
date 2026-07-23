# Point Transformer V3 (PTv3) semantic-segmentation model.
#
# Architecture hyperparameters follow the Pointcept reference
# config configs/plant/semseg-pt-v3m1-0-base-cos.py. They are intended as a
# starting point and will be tuned per dataset.
#
# PTv3 reuses PSS's voxel data flow: `Det3DDataPreprocessor` (voxel_type
# 'minkunet') voxelizes points, the `MinkUNet` segmentor calls
# `backbone(voxel_features, coors)`, and `PTv3SegHead` classifies each voxel and
# maps predictions back to points. `PointTransformerV3Backbone` adapts the
# vendored PTv3 core to the `(voxel_features, coors)` signature.
model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        # PTv3 adapter expects coors ordered [batch, x, y, z].
        batch_first=True,
        max_voxels=None,
        voxel_layer=dict(
            max_num_points=-1,
            # point_cloud_range / voxel_size are overridden per dataset.
            point_cloud_range=[0, 0, 0, 100, 100, 100],
            voxel_size=[0.05, 0.05, 0.05],
            max_voxels=(-1, -1))),
    backbone=dict(
        type='PointTransformerV3Backbone',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        # flash_attn is not installed in the pss_softgroup env. The non-flash
        # attention path is functionally complete; upcast_attention/softmax keep
        # it numerically stable under fp16 AMP. Set enable_flash=True (and
        # upcast_*=False) if flash_attn is installed.
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False),
    decode_head=dict(
        type='PTv3SegHead',
        channels=64,  # = dec_channels[0]
        num_classes=2,  # overridden per dataset
        dropout_ratio=0,
        # Pointcept reference loss: CrossEntropy + Lovasz.
        loss_decode=[
            dict(
                type='mmdet.CrossEntropyLoss',
                avg_non_ignore=True,
                loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
        ],
        ignore_index=2),  # overridden per dataset
    train_cfg=dict(),
    test_cfg=dict())
