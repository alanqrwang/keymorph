from torch.utils.data import Dataset, IterableDataset
import random
import itertools


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
