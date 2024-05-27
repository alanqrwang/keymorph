import os
import torch
import numpy as np
import torchio as tio
from pathlib import Path
from torch.utils.data import DataLoader
from itertools import combinations
from dataset.utils import KeyMorphDataset


class IXIDataset(KeyMorphDataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.data_root = data_root
        self.seg_available = True
        self.modalities = ["T1", "T2", "PD"]

    def _get_subjects_dict(self, train):
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

        if train:
            start, end = 0, 428
        else:
            start, end = 428, 528

        subject_dict = {}

        for modality in self.modalities:
            # Images
            img_dir = Path(os.path.join(self.data_root, modality))
            mask_dir = Path(os.path.join(self.data_root, modality + "_mask"))
            seg_dir = Path(os.path.join(self.data_root, modality + "_seg"))

            subjects = [s.split(".")[0] for s in np.sort(os.listdir(img_dir))]
            extensions = [
                ".".join(s.split(".")[1:]) for s in np.sort(os.listdir(img_dir))
            ]
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
            subject_dict[modality] = loaded_subjects[start:end]

        return subject_dict

    def one_hot(self, asegs):
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
