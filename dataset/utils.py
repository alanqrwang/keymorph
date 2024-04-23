import os
import torchio as tio
from torch.utils.data import Dataset
import random
import re
from collections import defaultdict


def read_subjects_from_disk(
    root_dir: str,
    train: bool,
    include_seg: bool = True,
    include_lesion_seg: bool = False,
    must_have_longitudinal=False,
):
    """Creates dictionary of TorchIO subjects.
    {
        'sub1': {
            'mod1': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
            'mod2': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
            ...
        },
        'sub2': {
            'mod1': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
            'mod2': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
            ...
        },
        ...
    }
    Keys are each unique subject and modality, values are list of all paths with that subject and modality.
    Should be 3 or 4 different timepoints.

    If dataset has timepoints, pathnames are of the form: "<dataset>-time<time_id>_<sub_id>_<mod_id>.nii.gz."
    If dataset has no timepoints, pathnames are of the form: "<dataset>_<sub_id>_<mod_id>.nii.gz."

    TODO: Currently, code assumes all subjects have SynthSeg-generated labels.
    """
    if train:
        split_folder_img, split_folder_seg = "imagesTr", "synthSeglabelsTr"
    else:
        split_folder_img, split_folder_seg = "imagesTs", "synthSeglabelsTs"

    img_data_folder = os.path.join(root_dir, split_folder_img)
    seg_data_folder = os.path.join(root_dir, split_folder_seg)
    if include_lesion_seg:
        split_folder_lesion_seg = "labelsTr"
        lesion_seg_data_folder = os.path.join(root_dir, split_folder_lesion_seg)
        assert os.path.exists(
            lesion_seg_data_folder
        ), "No lesion segmentation found in labelsTr folder."

    # Initialize an empty dictionary using defaultdict for easier nested dictionary handling
    data_structure = defaultdict(lambda: defaultdict(list))

    # Regex pattern to match the filename and extract time_id, subject_id, and modality_id
    # Makes the time part optional and defaults to 0 if not present
    pattern = re.compile(r"(?:-time(\d+))?_(\d+)_(\d+)\.nii\.gz")

    def add_to_dict(subject_id, modality, subject):
        # Append the file path to the list, will sort later
        data_structure[subject_id][modality].append(subject)

    # Iterate through each file in the directory
    for filename in os.listdir(img_data_folder):
        if "mask" in filename or not filename.endswith(".nii.gz"):
            continue
        img_path = os.path.join(img_data_folder, filename)
        seg_path = os.path.join(seg_data_folder, filename)
        # Only include subjects that have a corresponding segmentation file
        if not os.path.exists(seg_path):
            continue
        if include_seg:
            # Construct TorchIO subject
            subject_kwargs = {
                "img": tio.ScalarImage(img_path),
                "seg": tio.LabelMap(seg_path),
            }
        else:
            subject_kwargs = {
                "img": tio.ScalarImage(img_path),
            }
        if include_lesion_seg:
            lesion_seg_path = os.path.join(lesion_seg_data_folder, filename)
            subject_kwargs["lesion_seg"] = tio.LabelMap(lesion_seg_path)
        subject = tio.Subject(**subject_kwargs)

        match = pattern.search(filename)
        if match:
            # Extract time_id, subject_id, and modality_id based on regex groups
            # Default time_id to '0' if not present
            time_id, subject_id, modality_id = match.groups(default="0")
            # Add the extracted information along with the file path to the dictionary
            add_to_dict(subject_id, modality_id, subject)

    if must_have_longitudinal:
        for subject in list(data_structure.keys()):
            for modality in list(data_structure[subject].keys()):
                # Check if there are at least 2 different timepoints for the modality
                if len(data_structure[subject][modality]) < 2:
                    del data_structure[subject][
                        modality
                    ]  # Remove modality if it doesn't meet the criterion
            if not data_structure[
                subject
            ]:  # Check if the subject has no remaining modalities
                del data_structure[
                    subject
                ]  # Remove subject if it has no valid modalities
    if len(data_structure) == 0 and must_have_longitudinal:
        raise ValueError(
            f"No subjects with longitudinal data found in {root_dir}. Do you have time- information in your paths?"
        )
    if len(data_structure) == 0:
        raise ValueError(f"No subjects found in {root_dir}.")

    # Convert defaultdicts back to regular dicts for cleaner output or further processing
    subject_dict = {k: dict(v) for k, v in data_structure.items()}

    # Also create a flattened list of all paths for convenience
    subject_list = []
    for subject in subject_dict:
        for modality in subject_dict[subject]:
            subject_list.extend(subject_dict[subject][modality])
    return subject_dict, subject_list


class SingleSubjectDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train,
        transform=None,
        include_seg=True,
        include_lesion_seg=False,
    ):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir,
            train,
            include_seg=include_seg,
            include_lesion_seg=include_lesion_seg,
        )
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        sub = random.choice(self.subject_list)
        sub.load()
        if self.transform:
            sub = self.transform(sub)
        return sub


class PairedDataset(Dataset):
    """General paired dataset.
    Given subject list, samples pairs of subjects without restriction."""

    def __init__(
        self,
        root_dir,
        train,
        transform=None,
        include_seg=True,
        include_lesion_seg=False,
    ):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir,
            train,
            include_seg=include_seg,
            include_lesion_seg=include_lesion_seg,
        )
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        sub1 = random.sample(self.subject_list, 1)[0]
        sub2 = random.sample(self.subject_list, 1)[0]
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


class SSSMPairedDataset(Dataset):
    """Longitudinal paired dataset.
    Given subject list, samples same-subject, single-modality pairs."""

    def __init__(self, root_dir, train, transform=None, include_seg=True):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir, train, include_seg=include_seg, must_have_longitudinal=True
        )
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        # Randomly select a subject
        subject = random.choice(list(self.subject_dict.keys()))
        # Randomly select a modality for the chosen subject
        modality = random.choice(list(self.subject_dict[subject].keys()))
        # Randomly sample two paths from the chosen subject and modality
        sub1, sub2 = random.sample(self.subject_dict[subject][modality], 2)
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


class DSSMPairedDataset(Dataset):
    """DSSM paired dataset.
    Given subject list, samples different-subject, same-modality pairs."""

    def __init__(self, root_dir, train, transform=None, include_seg=True):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir, train, include_seg=include_seg
        )

        def aggregate_paths_by_modality(data_structure):
            modality_aggregate = {}
            for subject, modalities in data_structure.items():
                for modality, paths in modalities.items():
                    if modality not in modality_aggregate:
                        modality_aggregate[modality] = []
                    modality_aggregate[modality].extend(paths)

            # Ensure that we only consider modalities with at least two paths
            valid_modalities = {
                mod: paths
                for mod, paths in modality_aggregate.items()
                if len(paths) >= 2
            }
            return valid_modalities

        self.modality_dict = aggregate_paths_by_modality(self.subject_dict)
        assert (
            len(self.modality_dict) > 1
        ), f"Must have at least 2 modalities: {root_dir}"
        self.transform = transform
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        mult_mod_list = list(self.modality_dict.values())
        single_mod_list = random.sample(mult_mod_list, 1)[0]
        sub1, sub2 = random.sample(single_mod_list, 2)
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


class LongitudinalPathDataset:
    """At every iteration, returns all paths associated with the same subject and same modality.
    Relies on TorchIO's lazy loading. If no transform is performed, then TorchIO
    won't load the image data into memory."""

    def __init__(self, root_dir, train, transform=None, include_seg=True, group_size=4):
        # super().__init__(root_dir, train, include_seg)
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir, train, include_seg=include_seg, must_have_longitudinal=True
        )
        self.transform = transform  # This is hardcoded to None
        self.group_size = group_size

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        # Randomly select a subject
        subject = random.choice(list(self.subject_dict.keys()))
        # Randomly select a modality for the chosen subject
        modality = random.choice(list(self.subject_dict[subject].keys()))
        # Randomly sample paths from the chosen subject and modality
        single_sub_mod_list = self.subject_dict[subject][modality]
        subs = random.sample(
            single_sub_mod_list, min(len(single_sub_mod_list), self.group_size)
        )
        from torchio.data import SubjectsDataset as TioSubjectsDataset

        return SimpleDatasetIterator(TioSubjectsDataset(subs, transform=self.transform))


class AggregatedFamilyDataset(Dataset):
    """Aggregates multiple ``families'' of datasets into one giant dataset.
    Also appends the name of the family to each sample.

    A family is defined as a list of datasets which share some characteristic.
    """

    def __init__(self, list_of_dataset_families, names):
        """Samples uniformly over multiple families.
        Then, samples uniformly within that family.

        Inputs:
            list_of_dataset_families: A list of lists of datasets.
            names: A list of names for each list of datasets.
        """
        self.list_of_dataset_families = list_of_dataset_families
        self.names = names
        assert len(list_of_dataset_families) == len(names)
        self.num_families = len(list_of_dataset_families)

    def __getitem__(self, i):
        family_idx = random.randrange(self.num_families)
        family = self.list_of_dataset_families[family_idx]
        family_name = self.names[family_idx]
        dataset = random.choice(family)
        return dataset[i], family_name

    def __len__(self):
        l = 0
        for d in self.list_of_dataset_families:
            for sub_ds in d:
                l += len(sub_ds)
        return l


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
