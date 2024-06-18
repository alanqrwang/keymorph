import random
import itertools
import torchio as tio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from itertools import combinations


class PairedDataset(Dataset):
    """General paired dataset.
    Given pair of subject lists, samples pairs of subjects without restriction."""

    def __init__(
        self,
        subject_pairs_list,
        transform=None,
    ):
        super().__init__()
        self.subject_list = subject_pairs_list
        self.transform = transform

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, i):
        sub1, sub2 = self.subject_list[i]
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


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


class RandomAggregatedDataset(Dataset):
    """Aggregates multiple datasets and returns random samples from them."""

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        chosen_dataset = random.choice(self.datasets)
        return chosen_dataset[i]


class KeyMorphDataset:
    def _parse_test_mod(self, mod):
        if isinstance(mod, str):
            mod1, mod2 = mod.split("_")
        else:
            mod1, mod2 = mod
        return mod1, mod2

    def get_subjects(self, train):
        raise NotImplementedError

    def get_pretrain_loader(self, batch_size, num_workers, transform):
        subjects = self.get_subjects(
            train=True,
        )
        if isinstance(subjects, dict):
            pretrain_datasets = [
                tio.data.SubjectsDataset(
                    subjects_list,
                    transform=transform,
                )
                for subjects_list in subjects.values()
            ]
            pretrain_dataset = ConcatDataset(pretrain_datasets)
        else:
            pretrain_dataset = tio.data.SubjectsDataset(
                subjects[0] + subjects[1],
                transform=transform,
            )

        return DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def get_train_loader(self, batch_size, num_workers, mix_modalities, transform):
        subjects = self.get_subjects(
            train=True,
        )
        if isinstance(subjects, dict):
            train_mods = list(subjects.keys())
            if mix_modalities:
                mod_pairs = list(combinations(train_mods, 2))
            else:
                mod_pairs = [(m, m) for m in train_mods]

            paired_datasets = []
            for mod1, mod2 in mod_pairs:
                paired_datasets.append(
                    PairedDataset(
                        list(itertools.product(subjects[mod1], subjects[mod2])),
                        transform=transform,
                    )
                )
            train_dataset = ConcatDataset(paired_datasets)
        else:
            train_dataset = PairedDataset(
                list(zip(subjects[0], subjects[1])),
                transform=transform,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return train_loader

    def get_test_loaders(self, batch_size, num_workers, transform, list_of_mods):
        subjects = self.get_subjects(
            train=False,
        )
        if isinstance(subjects, dict):
            test_datasets = []
            for dataset_name in list_of_mods:
                mod1, mod2 = self._parse_test_mod(dataset_name)
                subjects1 = subjects[mod1]
                subjects2 = subjects[mod2]
                test_datasets.append(
                    PairedDataset(list(zip(subjects1, subjects2)), transform=transform)
                )
            test_dataset = ConcatDataset(test_datasets)

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            test_dataset = PairedDataset(
                list(zip(subjects[0], subjects[1])), transform=transform
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        return test_loader

    def get_loaders(
        self, batch_size, num_workers, mix_modalities, transform, list_of_test_mods
    ):
        return (
            self.get_pretrain_loader(batch_size, num_workers, transform),
            self.get_train_loader(batch_size, num_workers, mix_modalities, transform),
            self.get_test_loaders(
                batch_size, num_workers, transform, list_of_test_mods
            ),
        )
