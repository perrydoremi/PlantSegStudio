Paper Link: https://doi.org/10.3390/agriculture13071321
# Introduction
The dataset used in the paper titled "Soybean-MVS: Annotated Three-Dimensional Model Dataset of Whole Growth Period Soybeans for 3D Plant Organ Segmentation" includes 102 three-dimensional models of soybean plants, covering five different soybean varieties (DN251, DN252, DN253, HN48, and HN51). These models span the entire growth period of soybeans, including 13 stages from the first trifoliolate stage (V1) to full maturity (R8).

## Labeling
The authors manually labeled the dataset using the open-source software CloudCompare v2.6.3. The labels were assigned to three specific plant organs: leaves, main stems, and stems. Each labeled model contains xyzRGB information stored in a .txt format, and each organ was uniformly sampled with 50,000 points.

# Train-test split
89 labeled models were divided into a training set, and 13 labeled models were
divided into a test set.

# PSS settings
x, y, z, R, G, B, sem, ins \
sem: 0: mainstem; 1: stem; 2: leaf \
ins: 0: mainstem; 1,2,3,...: stem or leaf \
Min Bounding Box: [43.171514, 38.978358, 36.144367] \
Max Bounding Box: [639.419983, 625.140808, 888.390155]

## Train-test split (SoyBeanMVS)
Train: 1-44, 65-85 (DN251, DN252, HN48) \
Test: 45-64, 86-102 (DN253, HN51) 


# Notice
There are some label issues in the dataset. 
For example, 20180612_DN253, the main stem and three sub stems are overlapped.