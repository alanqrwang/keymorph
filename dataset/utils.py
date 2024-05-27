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
        subject_list1,
        subject_list2,
        transform=None,
    ):
        super().__init__()
        self.subject_list = list(itertools.product(subject_list1, subject_list2))
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
    def _get_subjects_dict(self, train):
        raise NotImplementedError

    def get_pretrain_loader(self, batch_size, num_workers, transform):
        pretrain_mod_dict = self._get_subjects_dict(
            train=True,
        )
        pretrain_datasets = [
            tio.data.SubjectsDataset(
                subjects_list,
                transform=transform,
            )
            for subjects_list in pretrain_mod_dict.values()
        ]
        aggr_dataset = ConcatDataset(pretrain_datasets)

        return DataLoader(
            aggr_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def get_train_loader(self, batch_size, num_workers, mix_modalities, transform):
        train_mod_dict = self._get_subjects_dict(
            train=True,
        )
        train_mods = list(train_mod_dict.keys())
        if mix_modalities:
            mod_pairs = list(combinations(train_mods, 2))
        else:
            mod_pairs = [(m, m) for m in train_mods]

        paired_datasets = []
        for mod1, mod2 in mod_pairs:
            paired_datasets.append(
                PairedDataset(
                    train_mod_dict[mod1],
                    train_mod_dict[mod2],
                    transform=transform,
                )
            )
        train_loader = DataLoader(
            ConcatDataset(paired_datasets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return train_loader

    def get_test_loaders(self, batch_size, num_workers, transform):
        test_mod_dict = self._get_subjects_dict(
            train=False,
        )
        test_loaders = {}
        for mod, test_subjects in test_mod_dict.items():
            test_loaders[mod] = DataLoader(
                tio.data.SubjectsDataset(test_subjects, transform=transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        return test_loaders

    def get_loaders(self, batch_size, num_workers, mix_modalities, transform):
        return (
            self.get_pretrain_loader(batch_size, num_workers, transform),
            self.get_train_loader(batch_size, num_workers, mix_modalities, transform),
            self.get_test_loaders(batch_size, num_workers, transform),
        )
