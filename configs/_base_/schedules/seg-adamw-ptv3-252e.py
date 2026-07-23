# Optimizer + LR schedule for Point Transformer V3, translated from the
# Pointcept (pointcept_cair) PTv3 config semseg-pt-v3m1-0-base-cos.py:
#   optimizer   AdamW(lr=0.006, weight_decay=0.05)
#   scheduler   OneCycleLR(max_lr=0.006, pct_start=0.05, div_factor=10,
#                          final_div_factor=1000)
#   param_dicts 0.1x LR for `block.*` parameters
#
# mmengine's `OneCycleLR` is bound to an explicit total iteration count and is
# awkward to wire epoch-based, so the equivalent linear-warmup + cosine-anneal
# shape is used instead (same effective curve):
#   warmup  ~5% of 252 epochs (13) ramping lr/10 -> lr
#   anneal  cosine from lr -> lr / div_factor / final_div_factor (6e-7)
# These are tuning starting points; adjust `max_epochs` together with the
# scheduler `end`/`T_max` below.

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=0.006, weight_decay=0.05),
    # 0.1x LR on PTv3 transformer-block parameters (Pointcept `param_dicts`).
    paramwise_cfg=dict(custom_keys={'block': dict(lr_mult=0.1)}),
    clip_grad=None)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=13,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=239,
        eta_min=6e-7,
        by_epoch=True,
        begin=13,
        end=252,
        convert_to_iter_based=True),
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=252, val_interval=4)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
auto_scale_lr = dict(enable=False, base_batch_size=32)
