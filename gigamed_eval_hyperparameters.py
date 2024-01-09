EVAL_METRICS = [
    "mse",
    "softdice",
    "harddice",
    "hausd",
    "jdstd",
    "jdlessthan0",
]

EVAL_NAMES = {
    "id": [
        ("Dataset5083_IXIT1", "Dataset5083_IXIT1"),
        ("Dataset5084_IXIT2", "Dataset5084_IXIT2"),
        ("Dataset5085_IXIPD", "Dataset5085_IXIPD"),
    ],
    "ood": [
        (
            "Dataset7000_openneuro-ds004791",
            "Dataset7000_openneuro-ds004791",
        ),
        (
            "Dataset7001_openneuro-ds004848",
            "Dataset7001_openneuro-ds004848",
        ),
    ],
    "raw": [
        "Dataset7000_openneuro-ds004791",
        "Dataset7001_openneuro-ds004848",
    ],
}
EVAL_LESION_NAMES = {
    "id": [
        ("Dataset5000_BraTS-GLI_2023", "Dataset5000_BraTS-GLI_2023"),
    ],
    "ood": [
        #
    ],
    "raw": [],
}
EVAL_GROUP_NAMES = {
    "id": [
        "Dataset5083_IXIT1",
    ],
    "ood": [
        "Dataset7000_openneuro-ds004791",
        "Dataset7001_openneuro-ds004848",
    ],
    "raw": [
        "Dataset7000_openneuro-ds004791",
        "Dataset7001_openneuro-ds004848",
    ],
}
EVAL_LONG_NAMES = {
    "id": [
        "Dataset6000_PPMI-T1-3T-PreProc",
        "Dataset6001_ADNI-group-T1-3T-PreProc",
        "Dataset6002_OASIS3",
    ],
    "ood": [
        "Dataset6003_AIBL",
    ],
    "raw": [],
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
