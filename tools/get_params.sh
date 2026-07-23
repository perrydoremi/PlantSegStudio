export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/get_flops.py configs/Semantic_seg/DGCNN/dgcnn_COS.py --shape 8192 6 --modality point
