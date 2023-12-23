import os
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import random
import torch

data_dir = "/midtier/sablab/scratch/alw4013/data/synthseg_clean"


def one_hot(asegs):
    subset_regs = [
        [0, 0],  # Background
        [13, 52],  # Pallidum
        [18, 54],  # Amygdala
        [11, 50],  # Caudate
        [3, 42],  # Cerebral Cortex
        [17, 53],  # Hippocampus
        [10, 49],  # Thalamus
        [12, 51],  # Putamen
        [2, 41],  # Cerebral WM
        [8, 47],  # Cerebellum Cortex
        [4, 43],  # Lateral Ventricle
        [7, 46],  # Cerebellum WM
        [16, 16],
    ]  # Brain-Stem

    _, dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(chs, dim1, dim2, dim3)

    for i, s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1])
        one_hot[i, :, :, :] = (combined_vol * 1).float()

    mask = one_hot.sum(0).squeeze()
    ones = torch.ones_like(mask)
    non_roi = ones - mask
    one_hot[-1, :, :, :] = non_roi

    assert (
        one_hot.sum(0).sum() == dim1 * dim2 * dim3
    ), "One-hot encoding does not add up to 1"
    return one_hot


def read_subjects_from_disk(root_dir: str, load_seg=True):
    """Creates list of TorchIO subjects."""
    img_dir = os.path.join(root_dir, "image")
    img_data_paths = sorted(
        [
            os.path.join(img_dir, name)
            for name in os.listdir(img_dir)
            if ("image" in name and name.endswith(".nii.gz"))
        ]
    )
    seg_data_paths = [s.replace("image", "labels") for s in img_data_paths]

    # Now, load all subjects, separated by modality
    subject_list = []
    for img_path, seg_path in zip(img_data_paths, seg_data_paths):
        subject_kwargs = {"img": tio.ScalarImage(img_path)}
        if load_seg:
            subject_kwargs["seg"] = tio.LabelMap(seg_path)
        subject = tio.Subject(**subject_kwargs)
        subject_list.append(subject)

    return subject_list


class SingleSubjectDataset(Dataset):
    def __init__(self, root_dir, transform=None, load_seg=True):
        self.subject_list = read_subjects_from_disk(root_dir, load_seg=load_seg)
        self.transform = transform

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, x):
        subject = random.sample(self.subject_list, 1)[0]
        if self.transform:
            subject = self.transform(subject)
        return subject


class PairedSubjectDataset(Dataset):
    """Longitudinal."""

    def __init__(self, root_dir, transform=None, load_seg=True):
        self.subject_list = read_subjects_from_disk(root_dir, load_seg=load_seg)
        self.transform = transform

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, x):
        sub1, sub2 = random.sample(self.subject_list, 2)
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2, "synthbrain"


class SynthBrain:
    def __init__(self, batch_size, num_workers, load_seg=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_seg = load_seg
        self.transform = tio.Compose(
            [
                tio.Lambda(one_hot, include=("seg")),
            ]
        )

    def get_paired_dataset(self):
        dataset = PairedSubjectDataset(data_dir, self.transform, self.load_seg)

        self.print_dataset_stats([dataset], prefix="SynthMorph")
        return dataset

    def get_single_dataset(self):
        dataset = SingleSubjectDataset(data_dir, self.transform, self.load_seg)

        self.print_dataset_stats([dataset], prefix="SynthMorph")
        return dataset

    def get_train_loader(self):
        dataset = PairedSubjectDataset(data_dir, self.transform, self.load_seg)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_pretrain_loader(self):
        dataset = SingleSubjectDataset(data_dir, self.transform, self.load_seg)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

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
