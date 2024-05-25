import csv
import os
import torchio as tio
from torch.utils.data import DataLoader
from itertools import combinations
from dataset.utils import (
    InfiniteRandomPairedDataset,
    InfiniteRandomSingleDataset,
    InfiniteRandomAggregatedDataset,
)


def read_subjects_from_csv(csv_file, train):
    subjects_dict = {}
    total_subjects = 0

    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if (row["train"].lower() == "true") == train:
                modality = row["modality"]

                if modality not in subjects_dict:
                    subjects_dict[modality] = []

                subject = tio.Subject(
                    img=tio.ScalarImage(os.path.join(row["img_path"])),
                    seg=tio.LabelMap(os.path.join(row["seg_path"])),
                    mask=tio.LabelMap(os.path.join(row["mask_path"])),
                )

                subjects_dict[modality].append(subject)
                total_subjects += 1

    # Print statistics
    print(f"\nSplit train={train}")
    print(f"Total number of subjects: {total_subjects}")
    for modality, subjects in subjects_dict.items():
        print(f"Modality: {modality}, Number of subjects: {len(subjects)}")
    return subjects_dict


def get_loaders(csv_file, batch_size, num_workers, mix_modalities, transform):

    train_mod_dict = read_subjects_from_csv(
        csv_file,
        train=True,
    )
    test_mod_dict = read_subjects_from_csv(
        csv_file,
        train=False,
    )
    train_mods = list(train_mod_dict.keys())

    # Pretrain
    pretrain_loader = {
        k: DataLoader(
            InfiniteRandomAggregatedDataset(
                InfiniteRandomSingleDataset(
                    subjects_list,
                    transform=transform,
                )
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for k, subjects_list in train_mod_dict.items()
    }

    # Train
    if mix_modalities:
        mod_pairs = list(combinations(train_mods, 2))
    else:
        mod_pairs = [(m, m) for m in train_mods]

    paired_datasets = []
    for mod1, mod2 in mod_pairs:
        paired_datasets.append(
            InfiniteRandomPairedDataset(
                train_mod_dict[mod1],
                train_mod_dict[mod2],
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
    for mod, test_subjects in test_mod_dict.items():
        test_loaders[mod] = DataLoader(
            tio.data.SubjectsDataset(test_subjects, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return pretrain_loader, train_loader, test_loaders
