import os
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import random
from .synthbrain import SynthBrain

data_dir = "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed"

datasets_with_longitudinal = [
    "Dataset5114_UCSF-ALPTDG",
    "Dataset6000_PPMI-T1-3T-PreProc",
    "Dataset6001_ADNI-group-T1-3T-PreProc",
    "Dataset6002_OASIS3",
]
datasets_with_multiple_modalities = [
    "Dataset4999_IXIAllModalities",
    "Dataset5000_BraTS-GLI_2023",
    "Dataset5001_BraTS-SSA_2023",
    "Dataset5002_BraTS-MEN_2023",
    "Dataset5003_BraTS-MET_2023",
    "Dataset5004_BraTS-MET-NYU_2023",
    "Dataset5005_BraTS-PED_2023",
    "Dataset5006_BraTS-MET-UCSF_2023",
    "Dataset5007_UCSF-BMSR",
    "Dataset5012_ShiftsBest",
    "Dataset5013_ShiftsLjubljana",
    "Dataset5038_BrainTumour",
    "Dataset5090_ISLES2022",
    "Dataset5095_MSSEG",
    "Dataset5096_MSSEG2",
    "Dataset5111_UCSF-ALPTDG-time1",
    "Dataset5112_UCSF-ALPTDG-time2",
    "Dataset5113_StanfordMETShare",
]
datasets_with_one_modality = [
    "Dataset5010_ATLASR2",
    "Dataset5041_BRATS",
    "Dataset5042_BRATS2016",
    "Dataset5043_BrainDevelopment",
    "Dataset5044_EPISURG",
    "Dataset5066_WMH",
]

list_of_id_test_datasets = [
    # "Dataset4999_IXIAllModalities",
    "Dataset5083_IXIT1",
    "Dataset5084_IXIT2",
    "Dataset5085_IXIPD",
]

list_of_ood_test_datasets = [
    "Dataset6003_AIBL",
]

list_of_test_datasets = list_of_id_test_datasets + list_of_ood_test_datasets
# unused_datasets = [
# "Dataset5046_FeTA",
# ]


def read_subjects_from_disk(root_dir: str, train: bool, load_seg=True):
    """Creates dictionary of TorchIO subjects.
    Keys are each unique modality in the dataset, values are list of all paths with that modality.
    Ignores timepoints.
    """
    if train:
        split_folder_img, split_folder_seg = "imagesTr", "labelsTr"
    else:
        split_folder_img, split_folder_seg = "imagesTs", "labelsTs"
    img_data_folder = os.path.join(root_dir, split_folder_img)
    seg_data_folder = os.path.join(root_dir, split_folder_seg)

    img_data_paths = sorted(
        [
            os.path.join(img_data_folder, name)
            for name in os.listdir(img_data_folder)
            if ("mask" not in name and name.endswith(".nii.gz"))
        ]
    )

    # First, run through and get all unique modalities
    all_modalities = set()
    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        _, modality_id = (
            basename.split(".")[0][:-5],
            basename.split(".")[0][-5:],
        )
        all_modalities.add(modality_id)

    # Now, load all subjects, separated by modality
    subject_dict = {mod: [] for mod in all_modalities}
    for img_path in img_data_paths:
        subject_kwargs = {"img": tio.ScalarImage(img_path)}
        basename = os.path.basename(img_path)
        _, modality_id = (
            basename.split(".")[0][:-5],
            basename.split(".")[0][-5:],
        )
        if load_seg:
            seg_path = os.path.join(seg_data_folder, basename)
            subject_kwargs["seg"] = tio.LabelMap(seg_path)
        subject = tio.Subject(**subject_kwargs)
        subject_dict[modality_id].append(subject)

    first_list_length = len(list(subject_dict.values())[0])
    for list_value in list(subject_dict.values()):
        assert (
            len(list_value) == first_list_length
        ), "All lists in the dictionary must be of the same length"
    return subject_dict


def read_longitudinal_subjects_from_disk(root_dir: str, train: bool, load_seg=True):
    """Creates dictionary of TorchIO subjects.
    Keys are each unique subject and modality, values are list of all paths with that subject and modality.
    Should be 3 or 4 different timepoints.

    Assumes pathnames are of the form: "<dataset>-time<time_id>_<sub_id>_<mod_id>.nii.gz."
    """
    if train:
        split_folder_img, split_folder_seg = "imagesTr", "labelsTr"
    else:
        split_folder_img, split_folder_seg = "imagesTs", "labelsTs"
    img_data_folder = os.path.join(root_dir, split_folder_img)
    seg_data_folder = os.path.join(root_dir, split_folder_seg)

    img_data_paths = sorted(
        [
            os.path.join(img_data_folder, name)
            for name in os.listdir(img_data_folder)
            if ("mask" not in name and name.endswith(".nii.gz"))
        ]
    )

    # First, run through and get all unique modalities
    all_keys = set()
    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        name = basename.split(".")[0]
        dataset_name, sub_id, modality_id = name.split("_")
        timepoint = dataset_name.split("-")[-1]
        all_keys.add(f"{sub_id}_{modality_id}")

    # Now, load all subjects, separated by modality
    subject_dict = {mod: [] for mod in all_keys}
    for img_path in img_data_paths:
        subject_kwargs = {"img": tio.ScalarImage(img_path)}
        basename = os.path.basename(img_path)
        name = basename.split(".")[0]
        dataset_name, sub_id, modality_id = name.split("_")
        if load_seg:
            seg_path = os.path.join(seg_data_folder, basename)
            subject_kwargs["seg"] = tio.LabelMap(seg_path)
        subject = tio.Subject(**subject_kwargs)
        subject_dict[f"{sub_id}_{modality_id}"].append(subject)

    assert all(
        len(v) > 1 for _, v in subject_dict.items()
    ), "All subjects must have at least 2 timepoints"
    return subject_dict


class SingleSubjectDataset(Dataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_dict = read_subjects_from_disk(root_dir, train, load_seg=load_seg)
        self.transform = transform

    def __len__(self):
        return len(self.subject_dict[list(self.subject_dict.keys())[0]])

    def __getitem__(self, x):
        mult_mod_list = list(self.subject_dict.values())
        single_mod_list1 = random.sample(mult_mod_list, 1)[0]
        sub1 = random.sample(single_mod_list1, 1)[0]
        if self.transform:
            sub1 = self.transform(sub1)
        return sub1


class GroupDataset(SingleSubjectDataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True, group_size=3):
        super().__init__(root_dir, train, transform, load_seg)
        self.group_size = group_size

    def __getitem__(self, x):
        return [
            super(GroupDataset, self).__getitem__(x) for _ in range(self.group_size)
        ]


class SameSubjectSameModalityDataset(Dataset):
    """Longitudinal."""

    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_dict = read_longitudinal_subjects_from_disk(
            root_dir, train, load_seg=load_seg
        )
        self.transform = transform

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        all_sub_mod_list = list(self.subject_dict.values())
        single_sub_mod_list = random.sample(all_sub_mod_list, 1)[0]
        sub1, sub2 = random.sample(single_sub_mod_list, 2)
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2, "same_sub_same_mod"


class SameSubjectDiffModalityDataset(Dataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_dict = read_subjects_from_disk(root_dir, train, load_seg=load_seg)
        assert (
            len(self.subject_dict) > 1
        ), f"Must have at least 2 modalities: {root_dir}"
        self.transform = transform

    def __len__(self):
        return len(self.subject_dict[list(self.subject_dict.keys())[0]])

    def __getitem__(self, i):
        mult_mod_list = list(self.subject_dict.values())
        single_mod_list1, single_mod_list2 = random.sample(mult_mod_list, 2)
        sub_id = random.choice(range(len(single_mod_list1)))
        sub1, sub2 = single_mod_list1[sub_id], single_mod_list2[sub_id]
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2, "same_sub_diff_mod"


class DiffSubjectSameModalityDataset(Dataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_dict = read_subjects_from_disk(root_dir, train, load_seg=load_seg)
        self.transform = transform

    def __len__(self):
        return len(self.subject_dict[list(self.subject_dict.keys())[0]])

    def __getitem__(self, x):
        mult_mod_list = list(self.subject_dict.values())
        single_mod_list = random.sample(mult_mod_list, 1)[0]
        sub1, sub2 = random.sample(single_mod_list, 2)
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2, "diff_sub_same_mod"


class DiffSubjectDiffModalityDataset(Dataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_dict = read_subjects_from_disk(root_dir, train, load_seg=load_seg)
        assert (
            len(self.subject_dict) > 1
        ), f"Must have at least 2 modalities: {root_dir}"
        self.transform = transform

    def __len__(self):
        return len(self.subject_dict[list(self.subject_dict.keys())[0]])

    def __getitem__(self, x):
        mult_mod_list = list(self.subject_dict.values())
        single_mod_list1, single_mod_list2 = random.sample(mult_mod_list, 2)
        sub1 = random.sample(single_mod_list1, 1)[0]
        sub2 = random.sample(single_mod_list2, 1)[0]
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2, "diff_sub_diff_mod"


class ConcatDataset(Dataset):
    """Samples uniformly over list of datasets.
    Then, samples uniformly within that list."""

    def __init__(self, *datasets):
        self.all_datasets = [
            *datasets,
        ]

    def __getitem__(self, i):
        dataset_list = random.choice(self.all_datasets)
        dataset = random.choice(dataset_list)
        return dataset[i]

    def __len__(self):
        l = 0
        for d in self.all_datasets:
            for sub_ds in d:
                l += len(sub_ds)
        return l


class GigaMed:
    def __init__(
        self,
        batch_size,
        num_workers,
        load_seg=True,
        sample_same_mod_only=True,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_seg = load_seg
        self.sample_same_mod_only = sample_same_mod_only
        self.transform = tio.Compose(
            [
                tio.OneHot(num_classes=15, include=("seg")),
            ]
        )

    def get_sssm_datasets(self):
        """Longitudinal"""
        sssm_dataset_names = datasets_with_longitudinal

        sssm_datasets = []
        for ds_name in sssm_dataset_names:
            sssm_datasets.append(
                SameSubjectSameModalityDataset(
                    os.path.join(data_dir, ds_name),
                    True,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )
        self.print_dataset_stats(sssm_datasets, "SSSM")
        return sssm_datasets

    def get_ssdm_datasets(self):
        """Single subject, different modality (SSDM) must have multiple modalities for a single subject"""
        ssdm_dataset_names = datasets_with_multiple_modalities
        ssdm_datasets = []
        for ds_name in ssdm_dataset_names:
            ssdm_datasets.append(
                SameSubjectDiffModalityDataset(
                    os.path.join(data_dir, ds_name),
                    True,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )
        self.print_dataset_stats(ssdm_datasets, "SSDM")
        return ssdm_datasets

    def get_dssm_datasets(self):
        """Different subject, same modality (DSSM) can have one or multiple modalities for a single subject"""
        dssm_dataset_names = (
            datasets_with_one_modality + datasets_with_multiple_modalities
        )
        dssm_datasets = []
        for ds_name in dssm_dataset_names:
            dssm_datasets.append(
                DiffSubjectSameModalityDataset(
                    os.path.join(data_dir, ds_name),
                    True,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )
        self.print_dataset_stats(dssm_datasets, "DSSM")
        return dssm_datasets

    def get_dsdm_datasets(self):
        """Different subject, different modality (DSDM) must have multiple modalities for a single subject"""
        dsdm_dataset_names = datasets_with_multiple_modalities
        dsdm_datasets = []
        for ds_name in dsdm_dataset_names:
            dsdm_datasets.append(
                DiffSubjectDiffModalityDataset(
                    os.path.join(data_dir, ds_name),
                    True,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )

        self.print_dataset_stats(dsdm_datasets, "DSDM")
        return dsdm_datasets

    def get_single_datasets(self):
        dataset_names = set(
            datasets_with_longitudinal
            + datasets_with_multiple_modalities
            + datasets_with_one_modality
        )

        datasets = []
        for ds_name in dataset_names:
            datasets.append(
                SingleSubjectDataset(
                    os.path.join(data_dir, ds_name),
                    True,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )

        self.print_dataset_stats(datasets, "Pretraining")
        return datasets

    def get_train_loader(self):
        sssm_datasets = self.get_sssm_datasets()
        dssm_datasets = self.get_dssm_datasets()
        ssdm_datasets = self.get_ssdm_datasets()
        dsdm_datasets = self.get_dsdm_datasets()
        if self.sample_same_mod_only:
            final_dataset = ConcatDataset(sssm_datasets, dssm_datasets)
        else:
            final_dataset = (
                ConcatDataset(
                    sssm_datasets, ssdm_datasets, dssm_datasets, dsdm_datasets
                ),
            )

        train_loader = DataLoader(
            final_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_test_loaders(self):
        test_datasets = []
        for ds_name in list_of_test_datasets:
            test_datasets.append(
                SingleSubjectDataset(
                    os.path.join(data_dir, ds_name),
                    False,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )
        self.print_dataset_stats(test_datasets, "Test")

        loaders = {}
        for ds_name, ds in zip(list_of_test_datasets, test_datasets):
            loaders[ds_name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return loaders

    def get_group_loaders(self):
        test_datasets = []
        for ds_name in list_of_test_datasets:
            test_datasets.append(
                GroupDataset(
                    os.path.join(data_dir, ds_name),
                    False,
                    self.transform,
                    load_seg=self.load_seg,
                )
            )
        self.print_dataset_stats(test_datasets, "Test")

        loaders = {}
        for ds_name, ds in zip(list_of_test_datasets, test_datasets):
            loaders[ds_name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return loaders

    def get_pretrain_loader(self):
        datasets = self.get_single_datasets()
        train_loader = DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def get_reference_subject(self):
        ds_name = datasets_with_multiple_modalities[0]
        root_dir = os.path.join(data_dir, ds_name)
        subject_dict = read_subjects_from_disk(root_dir, True, load_seg=False)
        return subject_dict[list(subject_dict.keys())[0]][0]

    def print_dataset_stats(self, datasets, prefix=""):
        print(f"\n{prefix} dataset has {len(datasets)} datasets.")
        tot = 0
        for i, ds in enumerate(datasets):
            tot += len(ds)
            print(
                "-> Dataset {} has {} subjects".format(
                    i,
                    len(ds),
                )
            )
        print("Total: ", tot)


class GigaMedSynthBrain(GigaMed):
    def get_train_loader(self):
        sssm_datasets = self.get_sssm_datasets()
        dssm_datasets = self.get_dssm_datasets()
        ssdm_datasets = self.get_ssdm_datasets()
        dsdm_datasets = self.get_dsdm_datasets()
        sb_dataset = SynthBrain(
            self.batch_size, self.num_workers, load_seg=True
        ).get_paired_dataset()
        if self.sample_same_mod_only:
            final_dataset = ConcatDataset(sssm_datasets, dssm_datasets, [sb_dataset])
        else:
            final_dataset = (
                ConcatDataset(
                    sssm_datasets,
                    ssdm_datasets,
                    dssm_datasets,
                    dsdm_datasets,
                    [sb_dataset],
                ),
            )
        train_loader = DataLoader(
            final_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_pretrain_loader(self):
        datasets = self.get_single_datasets()
        sb_dataset = SynthBrain(
            self.batch_size, self.num_workers, load_seg=True
        ).get_single_dataset()
        train_loader = DataLoader(
            ConcatDataset(datasets, [sb_dataset]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader
