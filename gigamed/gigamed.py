import os
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import random
from gigamed.synthbrain import SynthBrain, one_hot
import pandas as pd

id_csv_file = "/home/alw4013/keymorph/gigamed/gigamed_id.csv"
ood_csv_file = "/home/alw4013/keymorph/gigamed/gigamed_ood.csv"


def read_subjects_from_disk(root_dir: str, train: bool, load_seg=True):
    """Creates dictionary of TorchIO subjects.
    Keys are each unique modality in the dataset, values are list of all paths with that modality.
    Ignores timepoints.
    """
    if train:
        split_folder_img, split_folder_seg = "imagesTr", "synthSeglabelsTr"
    else:
        split_folder_img, split_folder_seg = "imagesTs", "synthSeglabelsTs"
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
        ), f"All lists in the dictionary must be of the same length, {img_data_folder}"
    return subject_dict


def read_longitudinal_subjects_from_disk(root_dir: str, train: bool, load_seg=True):
    """Creates dictionary of TorchIO subjects.
    Keys are each unique subject and modality, values are list of all paths with that subject and modality.
    Should be 3 or 4 different timepoints.

    Assumes pathnames are of the form: "<dataset>-time<time_id>_<sub_id>_<mod_id>.nii.gz."
    """
    if train:
        split_folder_img, split_folder_seg = "imagesTr", "synthSeglabelsTr"
    else:
        split_folder_img, split_folder_seg = "imagesTs", "synthSeglabelsTs"
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


class SingleSubjectPathDataset(Dataset):
    """Same as SingleSubjectDataset, but only returns paths.
    Relies on TorchIO's lazy loading. If no transform is performed, then TorchIO
    won't load the image data into memory."""

    def __init__(self, root_dir, train, load_seg=True):
        self.subject_dict = read_subjects_from_disk(root_dir, train, load_seg=load_seg)

    def __len__(self):
        return len(self.subject_dict[list(self.subject_dict.keys())[0]])

    def __getitem__(self, x):
        mult_mod_list = list(self.subject_dict.values())
        single_mod_list1 = random.sample(mult_mod_list, 1)[0]
        sub1 = random.sample(single_mod_list1, 1)[0]
        return sub1


class GroupDataset(SingleSubjectDataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True, group_size=4):
        super().__init__(root_dir, train, transform, load_seg)
        self.group_size = group_size

    def __getitem__(self, x):
        return [
            super(GroupDataset, self).__getitem__(x) for _ in range(self.group_size)
        ]


class GroupPathDataset(SingleSubjectPathDataset):
    """Same as GroupDataset, but only returns paths.
    Relies on TorchIO's lazy loading. If no transform is performed, then TorchIO
    won't load the image data into memory."""

    def __init__(self, root_dir, train, load_seg=True, group_size=4):
        super().__init__(root_dir, train, load_seg)
        self.group_size = group_size

    def __getitem__(self, x):
        return [
            super(GroupPathDataset, self).__getitem__(x) for _ in range(self.group_size)
        ]


class LongPathDataset(SingleSubjectPathDataset):
    """Same as GroupDataset, but only returns paths.
    Samples longitudinal; i.e. only within the same subject.
    Relies on TorchIO's lazy loading. If no transform is performed, then TorchIO
    won't load the image data into memory."""

    def __init__(self, root_dir, train, load_seg=True, group_size=4):
        super().__init__(root_dir, train, load_seg)
        self.subject_dict = read_longitudinal_subjects_from_disk(
            root_dir, train, load_seg=load_seg
        )
        self.group_size = group_size

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        all_sub_mod_list = list(self.subject_dict.values())
        single_sub_mod_list = random.sample(all_sub_mod_list, 1)[0]
        return random.sample(
            single_sub_mod_list, min(len(single_sub_mod_list), self.group_size)
        )


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


class GigaMedNames:
    """Convenience class. Handles all dataset names in GigaMed."""

    def __init__(self):
        self.gigamed_id_df = pd.read_csv(id_csv_file, header=0)
        self.gigamed_ood_df = pd.read_csv(ood_csv_file, header=0)

    def all(self, id=True, train=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        if train:
            mask = df["has_train"]
        else:
            mask = df["has_test"]
        return df.loc[mask]["name"].tolist()

    def with_longitudinal(self, id=True, train=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        if train:
            mask = df["has_longitudinal"] & df["has_train"]
        else:
            mask = df["has_longitudinal"] & df["has_test"]
        return df.loc[mask]["name"].tolist()

    def with_multiple_modalities(self, id=True, train=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        if train:
            mask = df["has_multiple_modalities"] & df["has_train"]
        else:
            mask = df["has_multiple_modalities"] & df["has_test"]
        return df.loc[mask]["name"].tolist()

    def with_one_modality(self, id=True, train=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        if train:
            mask = ~df["has_multiple_modalities"] & df["has_train"]
        else:
            mask = ~df["has_multiple_modalities"] & df["has_test"]
        return df.loc[mask]["name"].tolist()

    def with_lesion_seg(self, id=True, train=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        if train:
            mask = df["has_lesion_seg"] & df["has_train"]
        else:
            mask = df["has_lesion_seg"] & df["has_test"]
        return df.loc[mask]["name"].tolist()


class GigaMedDataset:
    """Convenience class. Handles creating Pytorch Datasets."""

    def __init__(
        self,
        data_dir,
        load_seg=True,
        transform=None,
        group_size=4,
    ):
        self.data_dir = data_dir
        self.load_seg = load_seg
        if transform is None:
            self.transform = tio.Compose(
                [
                    tio.Lambda(one_hot, include=("seg",)),
                ]
            )
        else:
            self.transform = transform

        self.gigamed_names = GigaMedNames()
        self.group_size = group_size

    def get_sssm_datasets(self, id=True, train=True):
        """Single subject, single modality (Longitudinal)"""
        names = self.gigamed_names.with_longitudinal(id=id)
        datasets = {}
        for name in names:
            datasets[name] = SameSubjectSameModalityDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
            )
        self.print_dataset_stats(datasets, f"SSSM, ID={id}, Train={train}")
        return datasets

    def get_ssdm_datasets(self, id=True, train=True):
        """Single subject, different modality (SSDM) must have multiple modalities for a single subject"""
        names = self.gigamed_names.with_multiple_modalities(id=id)
        datasets = {}
        for name in names:
            datasets[name] = SameSubjectDiffModalityDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
            )
        self.print_dataset_stats(datasets, f"SSDM, ID={id}, Train={train}")
        return datasets

    def get_dssm_datasets(self, id=True, train=True):
        """Different subject, same modality (DSSM) can have one or multiple modalities for a single subject"""
        names = self.gigamed_names.with_one_modality(
            id=id
        ) + self.gigamed_names.with_multiple_modalities(id=id)
        datasets = {}
        for name in names:
            datasets[name] = DiffSubjectSameModalityDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
            )
        self.print_dataset_stats(datasets, f"DSSM, ID={id}, Train={train}")
        return datasets

    def get_dsdm_datasets(self, id=True, train=True):
        """Different subject, different modality (DSDM) must have multiple modalities for a single subject"""
        names = self.gigamed_names.with_multiple_modalities(id=id)
        datasets = {}
        for name in names:
            datasets[name] = DiffSubjectDiffModalityDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
            )

        self.print_dataset_stats(datasets, f"DSDM, ID={id}, Train={train}")
        return datasets

    def get_datasets(self, id=True, train=True):
        names = self.gigamed_names.all(id=id, train=train)
        datasets = {}
        for name in names:
            datasets[name] = SingleSubjectDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
            )

        self.print_dataset_stats(datasets, f"All datasets, ID={id}, Train={train}")
        return datasets

    def get_group_datasets(self, id=True, train=True):
        """Group datasets can be from any dataset"""
        names = self.gigamed_names.all(id=id, train=train)
        datasets = {}
        for name in names:
            datasets[name] = GroupDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
                group_size=self.group_size,
            )
        self.print_dataset_stats(datasets, f"Group, ID={id}, Train={train}")
        return datasets

    def get_group_path_datasets(self, id=True, train=True):
        """Group datasets can be from any dataset"""
        names = self.gigamed_names.all(id=id, train=train)
        datasets = {}
        for name in names:
            datasets[name] = GroupPathDataset(
                os.path.join(self.data_dir, name),
                train,
                load_seg=self.load_seg,
                group_size=self.group_size,
            )
        self.print_dataset_stats(datasets, f"Group, ID={id}, Train={train}")
        return datasets

    def get_longitudinal_path_datasets(self, id=True, train=True):
        names = self.gigamed_names.with_longitudinal(id=id, train=train)
        datasets = {}
        for name in names:
            datasets[name] = LongPathDataset(
                os.path.join(self.data_dir, name),
                train,
                load_seg=self.load_seg,
                group_size=self.group_size,
            )
        self.print_dataset_stats(datasets, f"Longitudinal, ID={id}, Train={train}")
        return datasets

    def get_lesion_datasets(self, id=True, train=True):
        names = self.gigamed_names.with_lesion_seg(id=id, train=train)
        datasets = {}
        for name in names:
            datasets[name] = SingleSubjectDataset(
                os.path.join(self.data_dir, name),
                False,
                self.transform,
                load_seg=self.load_seg,
            )
        self.print_dataset_stats(datasets, f"Lesion, ID={id}, Train={train}")
        return datasets

    def print_dataset_stats(self, datasets, prefix=""):
        print(f"\n{prefix} dataset has {len(datasets)} datasets.")
        tot = 0
        for name, ds in datasets.items():
            tot += len(ds)
            print(f"-> {name} has {len(ds)} subjects")
        print("Total: ", tot)

    def get_reference_subject(self):
        ds_name = self.gigamed_names.all(id=True, train=True)[0]
        root_dir = os.path.join(self.data_dir, ds_name)
        subject_dict = read_subjects_from_disk(root_dir, True, load_seg=False)
        return subject_dict[list(subject_dict.keys())[0]][0]


class GigaMed:
    """Top-level class. Handles creating Pytorch dataloaders."""

    def __init__(
        self,
        batch_size,
        num_workers,
        load_seg=True,
        sample_same_mod_only=True,
        transform=None,
        use_raw_data=False,
        group_size=4,
    ):
        proc_data_dir = "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed"
        raw_data_dir = "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_same_mod_only = sample_same_mod_only
        if use_raw_data:
            root_data_dir = raw_data_dir
        else:
            root_data_dir = proc_data_dir
        self.gigamed_dataset = GigaMedDataset(
            root_data_dir, load_seg=load_seg, transform=transform, group_size=group_size
        )

    def get_train_loader(self):
        sssm_datasets = list(
            self.gigamed_dataset.get_sssm_datasets(id=True, train=True).values()
        )
        dssm_datasets = list(
            self.gigamed_dataset.get_dssm_datasets(id=True, train=True).values()
        )
        ssdm_datasets = list(
            self.gigamed_dataset.get_ssdm_datasets(id=True, train=True).values()
        )
        dsdm_datasets = list(
            self.gigamed_dataset.get_dsdm_datasets(id=True, train=True).values()
        )
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

    def get_eval_loaders(self, id):
        datasets = self.gigamed_dataset.get_datasets(id=id, train=False)
        loaders = {}
        for name, ds in datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return loaders

    def get_eval_group_loaders(self, id):
        datasets = self.gigamed_dataset.get_group_path_datasets(id=id, train=False)
        loaders = {}
        for name, ds in datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return loaders

    def get_eval_longitudinal_loaders(self, id):
        datasets = self.gigamed_dataset.get_longitudinal_path_datasets(
            id=id, train=False
        )
        loaders = {}
        for name, ds in datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return loaders

    def get_eval_lesion_loaders(self, id):
        datasets = self.gigamed_dataset.get_lesion_datasets(id=id, train=False)
        loaders = {}
        for name, ds in datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return loaders

    def get_pretrain_loader(self):
        datasets = list(self.gigamed_dataset.get_datasets(id=True, train=True).values())
        train_loader = DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def get_reference_subject(self):
        return self.gigamed_dataset.get_reference_subject()


class GigaMedSynthBrain(GigaMed):
    """Combination of GigaMed + SynthBrain."""

    def get_train_loader(self):
        sssm_datasets = list(self.gigamed_dataset.get_sssm_datasets().values())
        dssm_datasets = list(self.gigamed_dataset.get_dssm_datasets().values())
        ssdm_datasets = list(self.gigamed_dataset.get_ssdm_datasets().values())
        dsdm_datasets = list(self.gigamed_dataset.get_dsdm_datasets().values())
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
        datasets = self.gigamed_dataset.get_datasets()
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


if __name__ == "__main__":
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
        "Dataset5114_UCSF-ALPTDG",
    ]
    datasets_with_one_modality = [
        "Dataset5010_ATLASR2",
        "Dataset5041_BRATS",
        "Dataset5042_BRATS2016",
        "Dataset5043_BrainDevelopment",
        "Dataset5044_EPISURG",
        "Dataset5066_WMH",
        "Dataset5083_IXIT1",
        "Dataset5084_IXIT2",
        "Dataset5085_IXIPD",
        "Dataset6000_PPMI-T1-3T-PreProc",
        "Dataset6001_ADNI-group-T1-3T-PreProc",
        "Dataset6002_OASIS3",
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

    gigamed = GigaMedDataset()
    print(gigamed.get_dataset_names_with_longitudinal(id=True))
    print(gigamed.get_dataset_names_with_multiple_modalities(id=True))
    print(gigamed.get_dataset_names_with_one_modality(id=True))
    assert (
        gigamed.get_dataset_names_with_longitudinal(id=True)
        == datasets_with_longitudinal
    )
    assert (
        gigamed.get_dataset_names_with_multiple_modalities(id=True)
        == datasets_with_multiple_modalities
    )
    assert (
        gigamed.get_dataset_names_with_one_modality(id=True)
        == datasets_with_one_modality
    )
