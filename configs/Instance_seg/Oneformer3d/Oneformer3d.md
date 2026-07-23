# Oneformer3D Integration
Add [oneformer3d](https://github.com/oneformer3d/oneformer3d) to PSS, including panoptic segmentation and instance-only segmentation. Particularly, to integrate the heavy-editted metric evaluation scripts, a `oneformer3d_utils` folder was created under `mmdet3d/evaluation/functional` to prevent messing up other metric evaluation.

**Notice:** We revise the `oneformer3d_utils/evaluate_semantic_instance` into `scannet_utils/evauate_semantic_instance` format to make metrics on stuff classes as `0` or `nan`.

To train the oneformer3d model, use config files under `configs/oneformer3d`. Make sure you change `load_from` and `work_dir`.

```python
python tools/train.py configs/{}
```