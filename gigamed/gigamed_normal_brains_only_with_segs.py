import os
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import random
from gigamed.synthbrain import SynthBrain, one_hot
import pandas as pd

id_csv_file = "/home/alw4013/keymorph/gigamed/gigamed_normal_brains_id.csv"
ood_csv_file = "/home/alw4013/keymorph/gigamed/gigamed_ood.csv"


def read_subjects_from_disk_with_seg(root_dir: str, train: bool, load_seg=True):
    """Creates list of TorchIO subjects.
    All subjects must have corresponding segmentation files in order to be included.
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

    # Now, load all subjects, separated by modality
    subject_list = []
    for img_path in img_data_paths:
        subject_kwargs = {"img": tio.ScalarImage(img_path)}
        basename = os.path.basename(img_path)
        seg_path = os.path.join(seg_data_folder, basename)
        # Only include subjects that have a corresponding segmentation file
        if not os.path.exists(seg_path):
            continue
        if load_seg:
            subject_kwargs["seg"] = tio.LabelMap(seg_path)
        subject = tio.Subject(**subject_kwargs)
        subject_list.append(subject)

    return subject_list


class SingleSubjectDataset(Dataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_list = read_subjects_from_disk_with_seg(
            root_dir, train, load_seg=load_seg
        )
        self.transform = transform

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, x):
        sub1 = random.sample(self.subject_list, 1)[0]
        if self.transform:
            sub1 = self.transform(sub1)
        return sub1


class SingleSubjectPathDataset(Dataset):
    """Same as SingleSubjectDataset, but only returns paths.
    Relies on TorchIO's lazy loading. If no transform is performed, then TorchIO
    won't load the image data into memory."""

    def __init__(self, root_dir, train, load_seg=True):
        self.subject_list = read_subjects_from_disk_with_seg(
            root_dir, train, load_seg=load_seg
        )

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, x):
        sub1 = random.sample(self.subject_list, 1)[0]
        return sub1


class GroupDataset(SingleSubjectDataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True, group_size=3):
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

    def __init__(self, root_dir, train, load_seg=True, group_size=3):
        super().__init__(root_dir, train, load_seg)
        self.group_size = group_size

    def __getitem__(self, x):
        return [
            super(GroupPathDataset, self).__getitem__(x) for _ in range(self.group_size)
        ]


class PairedSubjectDataset(Dataset):
    def __init__(self, root_dir, train, transform=None, load_seg=True):
        self.subject_list = read_subjects_from_disk_with_seg(
            root_dir, train, load_seg=load_seg
        )
        self.transform = transform

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, x):
        sub1 = random.sample(self.subject_list, 1)[0]
        sub2 = random.sample(self.subject_list, 1)[0]
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2, "synthbrain"


class ConcatDataset(Dataset):
    """Samples uniformly over list of datasets.
    Then, samples uniformly within that list."""

    def __init__(self, *datasets):
        self.all_datasets = [
            *datasets,
        ]
        print("all dataset", self.all_datasets)
        assert len(self.all_datasets) > 0

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

    def get_paired_datasets(self, id=True, train=True):
        names = self.gigamed_names.all(id=id)
        datasets = {}
        for name in names:
            datasets[name] = PairedSubjectDataset(
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
            )
        self.print_dataset_stats(datasets, f"Group, ID={id}, Train={train}")
        return datasets

    def get_longitudinal_datasets(self, id=True, train=True):
        names = self.gigamed_names.with_longitudinal(id=id, train=train)
        datasets = {}
        for name in names:
            datasets[name] = GroupDataset(
                os.path.join(self.data_dir, name),
                train,
                self.transform,
                load_seg=self.load_seg,
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
        subject_list = read_subjects_from_disk_with_seg(root_dir, True, load_seg=False)
        return subject_list[0]


class GigaMed:
    """Top-level class. Handles creating Pytorch dataloaders.
    Reads data from:
      1) /midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_preprocessed/
      2) /midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/

    Data from both directories are nnUNet-like in structure.
    Data from 1) is not skull stripped. Data from 2) is skull stripped using HD-BET.
    Only samples data if SynthSeg labels are present.
    """

    def __init__(self, batch_size, num_workers, load_seg=True, transform=None):
        noskullstrip_data_dir = "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_preprocessed/"
        skullstrip_data_dir = "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gigamed_dataset_noskullstrip = GigaMedDataset(
            noskullstrip_data_dir,
            load_seg=load_seg,
            transform=transform,
        )
        self.gigamed_dataset_skullstrip = GigaMedDataset(
            skullstrip_data_dir,
            load_seg=load_seg,
            transform=transform,
        )

    def get_train_loader(self):
        noskullstrip_paired_datasets = list(
            self.gigamed_dataset_noskullstrip.get_paired_datasets(
                id=True, train=True
            ).values()
        )
        skullstrip_paired_datasets = list(
            self.gigamed_dataset_skullstrip.get_paired_datasets(
                id=True, train=True
            ).values()
        )

        final_dataset = ConcatDataset(
            noskullstrip_paired_datasets, skullstrip_paired_datasets
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
        datasets = self.gigamed_dataset.get_longitudinal_datasets(id=id, train=False)
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
        noskullstrip_paired_datasets = list(
            self.gigamed_dataset_noskullstrip.get_datasets(id=True, train=True).values()
        )
        skullstrip_paired_datasets = list(
            self.gigamed_dataset_skullstrip.get_datasets(id=True, train=True).values()
        )

        final_dataset = ConcatDataset(
            noskullstrip_paired_datasets, skullstrip_paired_datasets
        )
        train_loader = DataLoader(
            final_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def get_reference_subject(self):
        return self.gigamed_dataset_skullstrip.get_reference_subject()


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
        datasets = self.gigamed_dataset.get_single_datasets()
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
    train_dataset_names = [
        "Dataset4999_IXIAllModalities",
        "Dataset1000_PPMI",
        "Dataset1001_PACS2019",
        "Dataset1002_AIBL",
        "Dataset1004_OASIS2",
        "Dataset1005_OASIS1",
        "Dataset1006_OASIS3",
        "Dataset1007_ADNI",
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
    # print(gigamed.get_dataset_names_with_longitudinal(id=True))
    # print(gigamed.get_dataset_names_with_multiple_modalities(id=True))
    # print(gigamed.get_dataset_names_with_one_modality(id=True))
    # assert (
    #     gigamed.get_dataset_names_with_longitudinal(id=True)
    #     == datasets_with_longitudinal
    # )
    # assert (
    #     gigamed.get_dataset_names_with_multiple_modalities(id=True)
    #     == datasets_with_multiple_modalities
    # )
    # assert (
    #     gigamed.get_dataset_names_with_one_modality(id=True)
    #     == datasets_with_one_modality
    # )
