Paper Link: https://doi.org/10.1371/journal.pone.0256340
# Point cloud labeling
To use this dataset for plant analysis tasks, we provide labels for each 3D point in the dataset.
We labeled each point as ‘soil’, ‘stem’, or ‘leaf’ point. Furthermore, each leaf receives its unique
label, making it distinctive from the other leaves on the same plant. The label of a particular
leaf is the same for consecutive point clouds of this plant and is consequently consistent for the
whole series of scans. In the following, we explain the procedures we used for labeling, and
some issues, which emerged in the process.
### Labeling tomato plants.
Fig 5 shows a detailed view of a raw tomato plant point cloud as well as its segmentation into individual leaves and the stem. It can be seen, that the
transition from stem to leaf can be detected quite easily at the spot where the leaf starts to
spread out.
### Labeling maize plants.
In contrast to the tomato plants, the separation of the maize
plants into stem and leaves is not as obvious because maize plants do not show a clearly identifiable stem. Leaves are emerging from the whorl of the plant without showing a distinctive
region that separates the respective leaf from the rest of the plant. Therefore, we labeled the
maize point clouds using the two following approaches, which we derived from two commonly
used methods for staging maize plants. The approaches are shown in Fig 6. Subfigure Fig 6A
shows a point cloud of a maize plant as it has been captured in the measurement series. The
first way of labeling is derived from the ‘Leaf Collar Method’ [60]. The resulting segmentation
is shown in Fig 6B. The second way of labeling is derived from the ‘Leaf Tip Method’, see Fig
6C. We explain both methods below.

In this paper they seems to use the 'Leaf Collar Method' for segmentation tasks.

# Train-test split
The author trained for maize and tomato separately and used 5 plants for training and 2 for
testing.

# PSS settings
x, y, z, sem, ins \
sem: 0: ground; 1: stem, 2: leaf \
ins: 0: ground; 1,2,3,...: stem or leaf \
Min Bounding Box: [102.02, 99.15020000000004, 24.506999999999998] \
Max Bounding Box: [413.24670000000003, 381.4666000000001, 469.3276]

## Train-test split (Pheno4DAll (Maize+Tomato))
Train: 1-35 (Maize 1-5) 50-104 (Tomato 1-5) \
Test: 36-49 (Maize 6-7) 105-126 (Tomato 6-7)