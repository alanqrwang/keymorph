import os
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from gigamed.synthbrain import SynthBrain
import pandas as pd
from pprint import pprint

from .utils import (
    read_subjects_from_disk,
    SSSMPairedDataset,
    DSSMPairedDataset,
    PairedDataset,
    SingleSubjectDataset,
    GroupDataset,
    LongitudinalGroupDataset,
    AggregatedFamilyDataset,
)

id_csv_file = "/home/alw4013/keymorph/gigamed/gigamed_id.csv"
ood_csv_file = "/home/alw4013/keymorph/gigamed/gigamed_ood.csv"


class GigaMedPaths:
    """Convenience class. Handles all dataset names in GigaMed."""

    def __init__(self):
        self.gigamed_id_df = pd.read_csv(id_csv_file, header=0)
        self.gigamed_ood_df = pd.read_csv(ood_csv_file, header=0)

    @staticmethod
    def get_filtered_ds_paths(df, conditions):
        """
        Returns a list of ds_path values that satisfy the given conditions.

        Parameters:
        - file_path: The path to the CSV file.
        - conditions: A dictionary where keys are column names and values are the conditions those columns must satisfy.

        Returns:
        - A list of ds_path values that meet the conditions.
        """
        # Apply each condition in the conditions dictionary
        for column, value in conditions.items():
            df = df[df[column] == value]
        # Extract the list of ds_path that satisfy the conditions
        return df["ds_path"].tolist()

    def get_ds_dirs(self, conditions, id=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        return self.get_filtered_ds_paths(df, conditions)


class GigaMedDataset:
    """Convenience class. Handles creating Pytorch Datasets."""

    def __init__(
        self,
        include_seg=True,
        transform=None,
        group_size=4,
    ):
        self.include_seg = include_seg
        self.transform = transform
        self.group_size = group_size
        self.gigamed_paths = GigaMedPaths()

    def get_dataset_family(self, conditions, DatasetType, id=True, **dataset_kwargs):
        ds_dirs = self.gigamed_paths.get_ds_dirs(conditions, id=id)
        train = conditions.get("has_train", False)
        datasets = {}
        for ds_dir in ds_dirs:
            datasets[ds_dir] = DatasetType(
                ds_dir,
                train,
                self.transform,
                include_seg=self.include_seg,
                **dataset_kwargs,
            )
        return datasets

    def get_reference_subject(self):
        conditions = {}
        ds_name = self.gigamed_paths.get_ds_dirs(conditions, id=True)[0]
        root_dir = os.path.join(ds_name)
        _, subject_list = read_subjects_from_disk(root_dir, True, include_seg=False)
        return subject_list[0]


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


class GigaMed:
    """Top-level class. Handles creating Pytorch dataloaders.

    Reads data from:
      1) /midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/
      2) /midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_preprocessed/
      3) /midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/

    Data from all directories are nnUNet-like in structure.
    Data from 2) are not skull-stripped. Data from 1) and 3) are skull-stripped using HD-BET.
    Data from 1) have (extreme) lesions. Data from 3) do not have lesions.

    Our data is split along 2 dimensions: skullstripped vs. non-skullstripped, and normal brains vs. brains with (extreme) lesions.
    ---------------------------------------------------------
    | Skullstripped, normal     | Skullstripped, lesion     |
    | Dice Loss or MSE Loss     | MSE Loss                  |
    | Any transform             | Rigid or Affine           |
    ---------------------------------------------------------
    | Non-skullstripped, normal | Non-skullstripped, lesion |
    | Dice Loss                 | -----                     |
    | Any transform             | -----                     |
    ---------------------------------------------------------

    Rules:
        1) If Dice loss, sample pairs without restriction.
        2) If MSE loss, sample same-modality pairs.
            a) If longitudinal pairs, use rigid transformation.
            b) If cross-subject with lesions, use affine transformation.
            c) If cross-subject with normal, use TPS_logunif.

    If data is skullstripped and normal, then we:
        1) Loss: can generate segmentations using SynthSeg and minimize Dice loss
            (we can use MSE loss on images directly, but won't for simplicity. Also Dice loss is more robust to noise).
        3) Transformation: can use a more flexible transformation (i.e. TPS) during training.

    If data is not skullstripped and normal, then we:
        1) Loss: cannot use MSE loss and must use Dice loss, because skull and neck are highly variable.
        2) Transformation: can use a more flexible transformation (i.e. TPS) during training.

    If data is skullstripped and has lesions, then we:
        1) Loss: cannot rely on the quality of SynthSeg labels and must use MSE loss.
        3) Transformation: must use a restrictive transformation (i.e. rigid and affine) during training.

    If data is not skullstripped and has lesions, then we:
        1) Loss: cannot use MSE loss or Dice loss


    In summary, there are 3 settings at training:
      1) Skullstripped and normal: Dice loss or MSE loss, TPS_logunif
      2) Skullstripped and lesion: MSE loss, rigid or affine
      3) Non-skullstripped and normal: Dice loss, TPS_logunif
    """

    def __init__(
        self,
        batch_size,
        num_workers,
        include_seg=True,
        transform=None,
        use_raw_data=False,
        group_size=4,
        normal_brains_only=False,
    ):
        # proc_data_dir = "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed"
        # raw_data_dir = "/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base"

        # noskullstrip_data_dir = "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_preprocessed/"
        # skullstrip_data_dir = "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normal_brains_only = normal_brains_only
        self.use_raw_data = use_raw_data

        self.dataset = GigaMedDataset(
            include_seg=include_seg, transform=transform, group_size=group_size
        )

    def print_dataset_stats(self, datasets, prefix=""):
        print(f"\n\n{prefix} dataset family has {len(datasets)} datasets.")
        # print("Conditions:")
        # pprint(conditions)
        # print(str(DatasetType))
        tot_sub = 0
        tot_img = 0
        for name, ds in datasets.items():
            tot_sub += len(ds)
            tot_img += ds.get_total_images()
            print(
                f"-> {name} has {len(ds)} subjects and {ds.get_total_images()} images."
            )
        print("Total subjects: ", tot_sub)
        print("Total images: ", tot_img)

    def get_train_loader(self):
        # MSE loss families
        skullstripped_lesion_sssm_datasets = self.dataset.get_dataset_family(
            {
                "has_train": True,
                "has_lesion": True,
                "is_skullstripped": True,
                "has_longitudinal": True,
            },
            SSSMPairedDataset,
            id=True,
        )
        skullstripped_normal_sssm_datasets = self.dataset.get_dataset_family(
            {
                "has_train": True,
                "has_lesion": False,
                "is_skullstripped": True,
                "has_longitudinal": True,
            },
            SSSMPairedDataset,
            id=True,
        )
        skullstripped_lesion_dssm_datasets = self.dataset.get_dataset_family(
            {
                "has_train": True,
                "has_lesion": True,
                "is_skullstripped": True,
                "has_multiple_modalities": True,
            },
            DSSMPairedDataset,
            id=True,
        )
        # skullstripped_normal_dssm_datasets = ... # This is not used because DSSM is subsumed by skullstrip_paired_datasets
        sssm_datasets = (
            skullstripped_lesion_sssm_datasets | skullstripped_normal_sssm_datasets
        )
        dssm_datasets = skullstripped_lesion_dssm_datasets

        # Dice loss families
        nonskullstrip_paired_datasets = self.dataset.get_dataset_family(
            {"has_lesion": False, "is_skullstripped": False, "has_train": True},
            PairedDataset,
            id=True,
        )
        skullstrip_paired_datasets = self.dataset.get_dataset_family(
            {"has_lesion": False, "is_skullstripped": True, "has_train": True},
            PairedDataset,
            id=True,
        )

        # Print some stats
        self.print_dataset_stats(sssm_datasets, "TRAIN: SSSM")
        self.print_dataset_stats(dssm_datasets, "TRAIN: Lesion DSSM")
        self.print_dataset_stats(
            nonskullstrip_paired_datasets, "TRAIN: Normal nonskullstripped"
        )
        self.print_dataset_stats(
            skullstrip_paired_datasets, "TRAIN: Normal skullstripped"
        )

        if self.normal_brains_only:
            family_datasets = [
                list(nonskullstrip_paired_datasets.values()),
                list(skullstrip_paired_datasets.values()),
            ]
            family_names = ["normal_nonskullstripped", "normal_skullstripped"]
            final_dataset = AggregatedFamilyDataset(family_datasets, family_names)
        else:

            family_datasets = [
                list(sssm_datasets.values()),
                list(dssm_datasets.values()),
                list(nonskullstrip_paired_datasets.values()),
                list(skullstrip_paired_datasets.values()),
            ]
            family_names = [
                "same_sub_same_mod",
                "diff_sub_same_mod",
                "normal_nonskullstripped",
                "normal_skullstripped",
            ]
            final_dataset = AggregatedFamilyDataset(family_datasets, family_names)

        train_loader = DataLoader(
            final_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_eval_loaders(self, id):
        datasets = self.dataset.get_dataset_family(
            {"has_lesion": False, "is_skullstripped": True, "has_test": True},
            SingleSubjectDataset,
            id=id,
        )

        self.print_dataset_stats(datasets, f"EVAL {id}: Normal skullstripped")

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
        datasets = self.dataset.get_dataset_family(
            {"has_lesion": False, "is_skullstripped": True, "has_test": True},
            GroupDataset,
            id=id,
        )

        self.print_dataset_stats(datasets, f"EVAL {id}: Normal group skullstripped")

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
        datasets = self.dataset.get_dataset_family(
            {
                "has_lesion": False,
                "is_skullstripped": True,
                "has_longitudinal": True,
                "has_test": True,
            },
            GroupDataset,
            id=id,
        )

        self.print_dataset_stats(
            datasets, f"EVAL {id}: Normal longitudinal group skullstripped"
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
        datasets = self.dataset.get_dataset_family(
            {
                "has_lesion": True,
                "is_skullstripped": True,
                "has_test": True,
            },
            SingleSubjectDataset,
            id=id,
        )

        self.print_dataset_stats(datasets, f"EVAL {id}: Lesion skullstripped")

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
        """Pretrain on all datasets."""
        datasets = self.dataset.get_dataset_family(
            {
                "has_train": True,
            },
            SingleSubjectDataset,
            id=True,
        )

        self.print_dataset_stats(datasets, "PRETRAIN:")

        train_loader = DataLoader(
            AggregatedFamilyDataset([list(datasets.values())], ["all_datasets"]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def get_reference_subject(self):
        return self.dataset.get_reference_subject()


class GigaMedSynthBrain(GigaMed):
    """Combination of GigaMed + SynthBrain."""

    def get_train_loader(self):
        sssm_datasets = list(self.gigamed_dataset.get_sssm_datasets().values())
        dssm_datasets = list(self.gigamed_dataset.get_dssm_datasets().values())
        ssdm_datasets = list(self.gigamed_dataset.get_ssdm_datasets().values())
        dsdm_datasets = list(self.gigamed_dataset.get_dsdm_datasets().values())
        sb_dataset = SynthBrain(
            self.batch_size, self.num_workers, include_seg=True
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
            self.batch_size, self.num_workers, include_seg=True
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


# if __name__ == "__main__":
#     train_dataset_names = [
#         "Dataset4999_IXIAllModalities",
#         "Dataset1000_PPMI",
#         "Dataset1001_PACS2019",
#         "Dataset1002_AIBL",
#         "Dataset1004_OASIS2",
#         "Dataset1005_OASIS1",
#         "Dataset1006_OASIS3",
#         "Dataset1007_ADNI",
#     ]

#     list_of_id_test_datasets = [
#         # "Dataset4999_IXIAllModalities",
#         "Dataset5083_IXIT1",
#         "Dataset5084_IXIT2",
#         "Dataset5085_IXIPD",
#     ]

#     list_of_ood_test_datasets = [
#         "Dataset6003_AIBL",
#     ]

#     list_of_test_datasets = list_of_id_test_datasets + list_of_ood_test_datasets

#     gigamed = GigaMedDataset()
#     # print(gigamed.get_dataset_names_with_longitudinal(id=True))
#     # print(gigamed.get_dataset_names_with_multiple_modalities(id=True))
#     # print(gigamed.get_dataset_names_with_one_modality(id=True))
#     # assert (
#     #     gigamed.get_dataset_names_with_longitudinal(id=True)
#     #     == datasets_with_longitudinal
#     # )
#     # assert (
#     #     gigamed.get_dataset_names_with_multiple_modalities(id=True)
#     #     == datasets_with_multiple_modalities
#     # )
#     # assert (
#     #     gigamed.get_dataset_names_with_one_modality(id=True)
#     #     == datasets_with_one_modality
#     # )
