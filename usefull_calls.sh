#!/bin/bash

networks=(
    "DSPVFormer3d"
    # "Mink-SG"
    # "Oneformer3d"
    # "PTFormer3d"
    # "SoftGroup"
    # "SPVCNN-SG"
    # "SPVFormer3d"
)

# Training
# for network in "${networks[@]}"; do
#     config_file="configs/Instance_seg/"$network/$network"_TomatoWUR_nodes.py"
    
#     # Check if config file exists
#     if [ -f "$config_file" ]; then
#         echo "Training $network with $config_file..."
#         python3 tools/train.py "$config_file"
#     else
#         echo "WARNING: Config file not found: $config_file"
#     fi
# done

# Inference
for network in "${networks[@]}"; do
    config_file="configs/Instance_seg/"$network/$network"_TomatoWUR_nodes.py"
    model_path=$(find "work_dirs/${network}_TomatoWUR_nodes" -name "best_all_ap_epoch_*.pth" | head -1)
    
    # Check if config file and model exist
    if [ -f "$config_file" ] && [ -f "$model_path" ]; then
        echo "Running inference for $network with $config_file and $model_path..."
        python3 tools/test.py "$config_file" "$model_path" --save-local --task lidar_inst_seg
    else
        echo "WARNING: Config file or model not found for $config_file"
    fi
done

