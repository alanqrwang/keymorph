import os
import torch
import numpy as np
import torchio as tio
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random
from itertools import combinations


def read_subjects_from_disk(directory, start_end, modality):
    """
    Get list of TorchIO subjects from disk. Each subject has an image,
    and optionally a mask and segmentation. Each subject is restricted
    to a single modality.

    :param: directory directory where MRI modilities and skull-stripping mask are located
    start_end  : interval of indeces that is use to create the loader
    modality   : string that matches modality on disk

    Return
    ------
        dataset : tio.SubjectsDataset
        loader  : torch.utils.data.DataLoader
    """

    start, end = start_end

    # Images
    img_dir = Path(os.path.join(directory, modality))
    mask_dir = Path(os.path.join(directory, modality + "_mask"))
    seg_dir = Path(os.path.join(directory, modality + "_seg"))

    subjects = [s.split(".")[0] for s in np.sort(os.listdir(img_dir))]
    extensions = [".".join(s.split(".")[1:]) for s in np.sort(os.listdir(img_dir))]
    paths = [img_dir / (sub + "." + e) for sub, e in zip(subjects, extensions)]

    loaded_subjects = []

    # For each subject's image, try to get corresponding mask and segmentation
    for i in range(len(paths)):
        name = subjects[i]
        subject_kwargs = {
            "name": name,
            "img": tio.ScalarImage(paths[i]),
        }
        # Build mask path and segmentation path
        mask_path = mask_dir / (name + "_mask.nii.gz")
        seg_path = seg_dir / (name + "_seg.nii.gz")
        if os.path.exists(mask_path):
            subject_kwargs["mask"] = tio.LabelMap(mask_path)
        if os.path.exists(seg_path):
            subject_kwargs["seg"] = tio.LabelMap(seg_path)

        _sub = tio.Subject(**subject_kwargs)
        loaded_subjects.append(_sub)

    # Split for train, val or test
    loaded_subjects = loaded_subjects[start:end]

    return loaded_subjects


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


def get_loaders(data_dir, batch_size, num_workers, num_test_subjects, mix_modalities):
    modalities = ["T1", "T2", "PD"]

    transform = tio.Compose(
        [
            tio.Lambda(lambda x: x.permute(0, 1, 3, 2)),
            tio.Mask(masking_method="mask"),
            tio.Resize(128),
            # tio.Lambda(one_hot, include=("seg",)),
        ]
    )

    # Pretrain
    pretrain_datasets = []
    for mod in modalities:
        subjects_list = read_subjects_from_disk(
            data_dir,
            (0, 428),
            mod,
        )
        pretrain_datasets.append(
            InfiniteRandomSingleDataset(
                subjects_list,
                transform=transform,
            )
        )
    print(pretrain_datasets)
    pretrain_loader = DataLoader(
        InfiniteRandomAggregatedDataset(pretrain_datasets),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Train
    train_datasets = {}
    for mod in modalities:
        train_datasets[mod] = read_subjects_from_disk(
            data_dir,
            (0, 428),
            mod,
        )
    if mix_modalities:
        mod_pairs = list(combinations(modalities, 2))
    else:
        mod_pairs = [(m, m) for m in modalities]

    paired_datasets = []
    for mod1, mod2 in mod_pairs:
        paired_datasets.append(
            InfiniteRandomPairedDataset(
                train_datasets[mod1],
                train_datasets[mod2],
                transform=transform,
            )
        )
    train_loader = DataLoader(
        InfiniteRandomAggregatedDataset(paired_datasets),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Test
    test_loaders = {}
    for mod in modalities:
        test_subjects = read_subjects_from_disk(
            data_dir, (428, 428 + num_test_subjects), mod
        )
        test_loaders[mod] = DataLoader(
            tio.data.SubjectsDataset(test_subjects, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return pretrain_loader, train_loader, test_loaders


class InfiniteRandomPairedDataset(IterableDataset):
    """General paired dataset.
    Given pair of subject lists, samples pairs of subjects without restriction."""

    def __init__(
        self,
        subject_list1,
        subject_list2,
        transform=None,
    ):
        super().__init__()
        self.subject_list1 = subject_list1
        self.subject_list2 = subject_list2
        self.transform = transform

    def __iter__(self):
        while True:
            sub1 = random.sample(self.subject_list1, 1)[0]
            sub2 = random.sample(self.subject_list2, 1)[0]
            sub1.load()
            sub2.load()
            if self.transform:
                sub1 = self.transform(sub1)
                sub2 = self.transform(sub2)
            yield sub1, sub2


class InfiniteRandomSingleDataset(IterableDataset):
    """Random single dataset."""

    def __init__(
        self,
        subject_list,
        transform=None,
    ):
        super().__init__()
        self.subject_list = subject_list
        self.transform = transform

    def __iter__(self):
        while True:
            sub1 = random.sample(self.subject_list, 1)[0]
            sub1.load()
            if self.transform:
                sub1 = self.transform(sub1)
            yield sub1


class SimpleDatasetIterator:
    """Simple replacement to DataLoader"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # Reset the index each time iter is called.
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            item = self.dataset[self.index]
            self.index += 1
            return item
        else:
            # No more data, stop the iteration.
            raise StopIteration


class InfiniteRandomAggregatedDataset(IterableDataset):
    """Aggregates multiple iterable datasets and returns random samples from them."""

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        while True:
            # Randomly choose one of the iterators
            chosen_iterator = random.choice(iterators)
            sample = next(chosen_iterator)
            yield sample


def create_simple(directory, transform, modality):
    """
    Create dataloader

    Arguments
    ---------
    directory  : directory where MRI modilities and skull-stripping mask are located
    transform  : TorchIO transformation
    modality   : string T1, T2 or PD

    Return
    ------
        dataset : tio.SubjectsDataset
        loader  : torch.utils.data.DataLoader
    """

    modality = modality.upper()

    """Get PATHS"""
    paths = []
    subjects = []
    for d in np.sort(os.listdir(directory + "{}/".format(modality))):
        if "ipynb" in d:
            continue
        paths += [directory + "{}/".format(modality) + d]
        subjects += [d]

    """Making loader"""
    loaded_subjects = []
    for i in range(len(paths)):
        _ls = tio.Subject(mri=tio.ScalarImage(paths[i]), name=subjects[i])
        loaded_subjects.append(_ls)

    dataset = tio.SubjectsDataset(loaded_subjects, transform=transform)

    return dataset
