SYNTHSEG_LABEL_NAMES = {
    0: "background",
    2: "left cerebral white matter",
    3: "left cerebral cortex",
    4: "left lateral ventricle",
    5: "left inferior lateral ventricle",
    7: "left cerebellum white matter",
    8: "left cerebellum cortex",
    10: "left thalamus",
    11: "left caudate",
    12: "left putamen",
    13: "left pallidum",
    14: "3rd ventricle",
    15: "4th ventricle",
    16: "brain-stem",
    17: "left hippocampus",
    18: "left amygdala",
    24: "CSF",
    26: "left accumbens area",
    28: "left ventral DC",
    41: "right cerebral white matter",
    42: "right cerebral cortex",
    43: "right lateral ventricle",
    44: "right inferior lateral ventricle",
    46: "right cerebellum white matter",
    47: "right cerebellum cortex",
    49: "right thalamus",
    50: "right caudate",
    51: "right putamen",
    52: "right pallidum",
    53: "right hippocampus",
    54: "right amygdala",
    58: "right accumbens area",
    60: "right ventral DC",
}

EVAL_METRICS = [
    "mse",
    "harddice",
    "hausd",
    "jdstd",
    "jdlessthan0",
]

EVAL_UNI_NAMES = {
    "id": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5083_IXIT1",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5084_IXIT2",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5084_IXIT2",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5085_IXIPD",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5085_IXIPD",
        ),
    ],
    "ood": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset7000_openneuro-ds004791",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset7000_openneuro-ds004791",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset7001_openneuro-ds004848",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset7001_openneuro-ds004848",
        ),
    ],
}
EVAL_MULTI_NAMES = {
    "id": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5084_IXIT2",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5085_IXIPD",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5084_IXIT2",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5085_IXIPD",
        ),
    ],
    "ood": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5084_IXIT2",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5085_IXIPD",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5084_IXIT2",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5085_IXIPD",
        ),
    ],
}
EVAL_LESION_NAMES = {
    "id": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5000_BraTS-GLI_2023",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5000_BraTS-GLI_2023",
        ),
    ],
    "ood": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5000_BraTS-GLI_2023",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset5000_BraTS-GLI_2023",
        ),
    ],
}
EVAL_GROUP_NAMES = {
    "id": [
        "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5083_IXIT1",
    ],
    "ood": [
        "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base_preprocessed/Dataset7000_openneuro-ds004791",
        "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base_preprocessed/Dataset7001_openneuro-ds004848",
    ],
}
EVAL_LONG_NAMES = {
    "id": [
        "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset6000_PPMI-T1-3T-PreProc",
        "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset6001_ADNI-group-T1-3T-PreProc",
        "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset6002_OASIS3",
    ],
    "ood": [
        "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset1006_OASIS3",
    ],
}

EVAL_AUGS = [
    "rot0",
    "rot45",
    "rot90",
    "rot135",
    "rot180",
]

EVAL_KP_ALIGNS = [
    "rigid",
    "affine",
    "tps_10",
    "tps_1",
    "tps_0.1",
    "tps_0.01",
    "tps_0",
]
EVAL_GROUP_KP_ALIGNS = [
    "tps_0",
]
EVAL_LONG_KP_ALIGNS = [
    "rigid",
]

EVAL_ITKELASTIX_ALIGNS = [
    "rigid",
    "affine",
    "bspline",
]
EVAL_GROUP_ITKELASTIX_ALIGNS = [
    "bspline",
]
EVAL_LONG_ITKELASTIX_ALIGNS = [
    "rigid",
]
