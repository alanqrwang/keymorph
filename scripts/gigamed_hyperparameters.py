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
    "softdice",
    "harddice",
    "harddiceroi",
    "hausd",
    "jdstd",
    "jdlessthan0",
]

EVAL_UNI_NAMES = {
    "ss": [
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
    "nss": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5083_IXIT1",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5084_IXIT2",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5084_IXIT2",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5085_IXIPD",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5085_IXIPD",
        ),
    ],
}
EVAL_MULTI_NAMES = {
    "ss": [
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
    "nss": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5084_IXIT2",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5083_IXIT1",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5085_IXIPD",
        ),
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5084_IXIT2",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5085_IXIPD",
        ),
    ],
}
EVAL_LESION_NAMES = {
    "ss": [
        (
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5000_BraTS-GLI_2023",
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5000_BraTS-GLI_2023",
        ),
    ],
    "nss": [],
}
EVAL_GROUP_NAMES = {
    "ss": [
        "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset5083_IXIT1",
    ],
    "nss": [
        "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset5083_IXIT1",
    ],
}
EVAL_LONG_NAMES = {
    "ss": [
        "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/Dataset1009_OASIS3deepsurfercorrected",
    ],
    "nss": [
        "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_preprocessed/Dataset1009_OASIS3deepsurfercorrected",
    ],
}

EVAL_AUGS = [
    "rot0",
    "rot45",
    "rot90",
    "rot135",
    # "rot180",
]

MODEL_HPS = {
    "keymorph": {
        "aligns": [
            "rigid",
            "affine",
            "tps_10",
            "tps_1",
            "tps_0.1",
            "tps_0.01",
            "tps_0",
        ],
        "group_aligns": [
            "tps_0",
        ],
        "long_aligns": [
            "rigid",
        ],
        "perform_groupwise_experiments": True,
        "group_sizes": [4, 8, 16, 32, 64, 128],
        "num_iters_groupwise": 5,
    },
    "itkelastix": {
        "aligns": [
            "rigid",
            "affine",
            "bspline",
        ],
        "group_aligns": [
            "bspline",
        ],
        "long_aligns": [
            "rigid",
        ],
        "perform_groupwise_experiments": True,
        "group_sizes": [4, 8, 16, 32, 64, 128],
    },
    "synthmorph": {
        "aligns": [
            "dense",
        ],
        "group_aligns": [
            "dense",
        ],
        "long_aligns": [
            "dense",
        ],
        "perform_groupwise_experiments": False,
    },
    "ants": {
        "aligns": [
            "rigid",
            "affine",
            "syn",
        ],
        "group_aligns": [
            "syn",
        ],
        "long_aligns": [
            "rigid",
        ],
        "perform_groupwise_experiments": False,
    },
}

GIGAMED_FAMILY_TRAIN_PARAMS = {
    "same_sub_same_mod": {
        "transform_type": "rigid",
        "loss_fn": "mse",
        "max_random_params": (0, 0.15, 3.1416, 0),
    },
    "diff_sub_same_mod": {
        "transform_type": "tps_loguniform",
        "loss_fn": "mse",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "synthbrain": {
        "transform_type": "tps_loguniform",
        "loss_fn": "dice",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "normal_skullstripped": {
        "transform_type": "tps_loguniform",
        "loss_fn": "dice",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "normal_nonskullstripped": {
        "transform_type": "tps_loguniform",
        "loss_fn": "dice",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
}

GIGAMED_FAMILY_TRAIN_PARAMS_MSE_ONLY = {
    "same_sub_same_mod": {
        "transform_type": "rigid",
        "loss_fn": "mse",
        "max_random_params": (0, 0.15, 3.1416, 0),
    },
    "diff_sub_same_mod": {
        "transform_type": "tps_loguniform",
        "loss_fn": "mse",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "synthbrain": {
        "transform_type": "tps_loguniform",
        "loss_fn": "mse",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "normal_skullstripped": {
        "transform_type": "tps_loguniform",
        "loss_fn": "mse",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "normal_nonskullstripped": {
        "transform_type": "tps_loguniform",
        "loss_fn": "mse",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
}

GIGAMED_FAMILY_TRAIN_PARAMS_TPS0_ONLY = {
    "same_sub_same_mod": {
        "transform_type": "tps_0",
        "loss_fn": "mse",
        "max_random_params": (0, 0.15, 3.1416, 0),
    },
    "diff_sub_same_mod": {
        "transform_type": "tps_0",
        "loss_fn": "mse",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "synthbrain": {
        "transform_type": "tps_0",
        "loss_fn": "dice",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "normal_skullstripped": {
        "transform_type": "tps_0",
        "loss_fn": "dice",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
    "normal_nonskullstripped": {
        "transform_type": "tps_0",
        "loss_fn": "dice",
        "max_random_params": (0.2, 0.15, 3.1416, 0.1),
    },
}
