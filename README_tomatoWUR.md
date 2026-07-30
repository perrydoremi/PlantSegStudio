## installation:
```bash
docker compose build
```

## currently tested on stem instances
Make sure to update tomatowur path to:
``/mnt/GARdata/TomatoWUR/``
```bash
python3 tools/train.py configs/Instance_seg/Oneformer3d/oneformer3d_1xb2_tomatowur_class_mapping_stem_instances.py 
```