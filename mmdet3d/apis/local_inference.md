# Local Inference in PSS

Add inference hook to inference the test data in configs and save the results after one testing iteration to the local directory without triggering visualization.

To only inference locally, please specify the `--save-local` and `--task` flag when testing. The `--task` flag is used to specify the segmentation task of the model, which is used to determine the output format. Currently we support instance seg inference `--task='lidar_inst_seg'` and semantic seg inference `--task='lidar_seg'`.

```python
python tools/test.py configs/oneformer3d/oneformer3d_LLC_instance_only_allsim.py \
    work_dirs/epoch_94.pth --save-local --task='lidar_inst_seg'
```
