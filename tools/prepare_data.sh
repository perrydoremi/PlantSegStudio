# python tools/collect_plant_data.py --name HR3DAllThree --data-dir ./data/plant/HR3DAllThree
# python tools/collect_plant_data.py --name HR3DSorghum --data-dir ./data/plant/HR3DSorghum
# python tools/collect_plant_data.py --name HR3DTobacco --data-dir ./data/plant/HR3DTobacco
# python tools/collect_plant_data.py --name HR3DTomato --data-dir ./data/plant/HR3DTomato
# python tools/collect_plant_data.py --name Pheno4DMaize --data-dir ./data/plant/Pheno4DMaize
# python tools/collect_plant_data.py --name Pheno4DTomato --data-dir ./data/plant/Pheno4DTomato

# python tools/collect_plant_data.py --name SYAUMaize --data-dir ./data/plant/SYAUMaize
# python tools/collect_plant_data.py --name ZJURapeSeed --data-dir ./data/plant/ZJURapeSeed
# python tools/collect_plant_data.py --name COS --data-dir ./data/plant/COS
# python tools/collect_plant_data.py --name Pheno4DAll2C --data-dir ./data/plant/Pheno4DAll2C

# python tools/create_plant_data.py --root-path ./data/plant/HR3DAllThree \
# --out-dir ./data/plant/HR3DAllThree --extra-tag HR3DAllThree --num-data 546

# python tools/create_plant_data.py --root-path ./data/plant/HR3DSorghum \
# --out-dir ./data/plant/HR3DSorghum --extra-tag HR3DSorghum --num-data 129

# python tools/create_plant_data.py --root-path ./data/plant/HR3DTobacco \
# --out-dir ./data/plant/HR3DTobacco --extra-tag HR3DTobacco --num-data 105

# python tools/create_plant_data.py --root-path ./data/plant/HR3DTomato \
# --out-dir ./data/plant/HR3DTomato --extra-tag HR3DTomato --num-data 312

# python tools/create_plant_data.py --root-path ./data/plant/Pheno4DMaize \
# --out-dir ./data/plant/Pheno4DMaize --extra-tag Pheno4DMaize --num-data 49



python tools/collect_plant_data.py --name pvllt --data-dir ./data/plant/pvllt

python tools/create_plant_data.py --root-path ./data/plant/pvllt \
--out-dir ./data/plant/pvllt --extra-tag pvllt --num-data 6


# python tools/create_plant_data.py --root-path ./data/plant/SYAUMaize \
# --out-dir ./data/plant/SYAUMaize --extra-tag SYAUMaize --num-data 428

# python tools/create_plant_data.py --root-path ./data/plant/ZJURapeSeed \
# --out-dir ./data/plant/ZJURapeSeed --extra-tag ZJURapeSeed --num-data 50

# python tools/create_plant_data.py --root-path ./data/plant/COS \
# --out-dir ./data/plant/COS --extra-tag COS --num-data 98

# python tools/create_plant_data.py --root-path ./data/plant/Pheno4DAll2C \
# --out-dir ./data/plant/Pheno4DAll2C --extra-tag Pheno4DAll2C --num-data 126
