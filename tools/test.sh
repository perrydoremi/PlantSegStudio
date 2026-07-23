#!/bin/bash
# Convenience wrapper around tools/test.py for PlantSegStudio_clean.
#
# Usage (run from the repo root):
#   bash tools/test.sh                         # default: PTv3 on COS (reproduces mIoU 0.9069)
#   bash tools/test.sh <config> <checkpoint>   # any config + checkpoint
#   bash tools/test.sh <config> <checkpoint> <work-dir> [extra tools/test.py args...]
#
# Examples:
#   # PTFormer instance on COS (AP checkpoint):
#   bash tools/test.sh \
#     configs/Instance_seg/PTFormer3d/best_flash/PTFormer3d_COS_best_flash.py \
#     $ARCHIVE/instance/COS/PTFormer-PTv3ps/best_all_ap_epoch_448.pth
#
#   # DSPVFormer standard split on COS:
#   bash tools/test.sh \
#     configs/Instance_seg/DSPVFormer3d/standard_split/DSPVFormer3d_COS.py \
#     /data/rd579/DSPVFormer_weights/standard_split_10folds/DSPVFormer/DSPVFormer_f1/best_all_ap_epoch_512.pth
#
#   # Pick a GPU or dump per-scene predictions:
#   CUDA_VISIBLE_DEVICES=3 bash tools/test.sh <cfg> <ckpt> work_dirs/foo --save-local --task='lidar_inst_seg'
set -euo pipefail

# --- environment ---------------------------------------------------------
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pss_softgroup            # flash-attn env required by PTv3/PTFormer
export PYTHONPATH="$REPO"               # resolve mmdet3d to this checkout, not the dev tree
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled              # no experiment logging for eval runs

# Archive root holding the manuscript configs + checkpoints (Tab06 / Tab07).
ARCHIVE="${ARCHIVE:-/home/rd579/PlantSegStudio/manuscript_results_archive}"

# --- defaults: PTv3 on COS ----------------------------------------------
DEF_CONFIG="configs/Semantic_seg/PTv3/best_flash/ptv3_COS_best_flash.py"
DEF_CKPT="$ARCHIVE/semantic/COS/PTv3/best_miou_epoch_112.pth"

CONFIG="${1:-$DEF_CONFIG}"
CKPT="${2:-$DEF_CKPT}"
WORK_DIR="${3:-work_dirs/test/$(basename "${CONFIG%.py}")}"
shift $(( $# < 3 ? $# : 3 )) || true    # remaining args pass through to tools/test.py

cd "$REPO"
mkdir -p "$WORK_DIR"

echo "config     : $CONFIG"
echo "checkpoint : $CKPT"
echo "work-dir   : $WORK_DIR"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "--------------------------------------------------------------"

python tools/test.py "$CONFIG" "$CKPT" --work-dir "$WORK_DIR" "$@"
