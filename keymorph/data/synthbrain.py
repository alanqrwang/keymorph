import os
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import random

data_dir = "/midtier/sablab/scratch/alw4013/data/synthseg"


def read_subjects_from_disk(root_dir: str, load_seg=True):
    """Creates list of TorchIO subjects."""
    img_data_paths = sorted(
        [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
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
        # self.transform = tio.Compose(
        #     [
        #         tio.OneHot(num_classes=70, include=("seg")),
        #     ]
        # )
        self.transform = tio.Compose(
            [
                tio.RandomNoise(std=0),
            ]
        )

    def get_train_loader(self):
        dataset = PairedSubjectDataset(data_dir, self.transform, self.load_seg)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_pretraining_loaders(self):
        dataset = SingleSubjectDataset(data_dir, self.transform, self.load_seg)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader, None
