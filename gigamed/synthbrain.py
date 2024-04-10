from torch.utils.data import DataLoader
import torchio as tio
import numpy as np
from SynthSeg.brain_generator import BrainGenerator


class SynthBrainDataset:
    def __init__(self, paired=False, transform=None):
        self.paired = paired
        self.transform = transform

        root_dir = "/home/alw4013/SynthSeg/"

        # input training label maps
        path_label_map = "/midtier/sablab/scratch/alw4013/synthseg/new_label_maps"
        subjects_prob = f"{root_dir}/data/labels_classes_priors/subject_prob.npy"

        # numpy arrays for generation labels, segmentation labels, etc.
        generation_labels = (
            f"{root_dir}/data/labels_classes_priors/generation_labels_2.0.npy"
        )
        generation_classes = (
            f"{root_dir}/data/labels_classes_priors/generation_classes_2.0.npy"
        )
        output_labels = f"{root_dir}/data/labels_classes_priors/synthseg_segmentation_labels_2.0.npy"
        n_neutral_labels = 19  # this has changed !!! It's because the CSF is now separated between cerebral and spinal fluid

        # do not include label 24 in output segmentations. Comment out if you wish to segment label 24.
        output_labels = np.load(output_labels)
        output_labels[4] = 0

        # ---------- Shape and resolution of the outputs ----------

        # number of channel to synthesise for multi-modality settings. Set this to 1 (default) in the uni-modality scenario.
        n_channels = 1

        # We have the possibility to generate training examples at a different resolution than the training label maps (e.g.
        # when using ultra HR training label maps). Here we want to generate at the same resolution as the training label maps,
        # so we set this to None.
        target_res = None

        # The generative model offers the possibility to randomly crop the training examples to a given size.
        # Here we crop them to 160^3, such that the produced images fit on the GPU during training.
        output_shape = 256

        # ---------- GMM sampling parameters ----------

        # Here we use uniform prior distribution to sample the means/stds of the GMM. Because we don't specify prior_means and
        # prior_stds, those priors will have default bounds of [0, 250], and [0, 35]. Those values enable to generate a wide
        # range of contrasts (often unrealistic), which will make the segmentation network contrast-agnostic.
        prior_distributions = "uniform"

        # ---------- Spatial augmentation ----------

        # We now introduce some parameters concerning the spatial deformation. They enable to set the range of the uniform
        # distribution from which the corresponding parameters are selected.
        # We note that because the label maps will be resampled with nearest neighbour interpolation, they can look less smooth
        # than the original segmentations.

        flipping = False  # enable right/left flipping
        scaling_bounds = 0  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
        rotation_bounds = 0  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
        shearing_bounds = 0.0  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
        translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
        nonlin_std = 4.0  # this controls the maximum elastic deformation (higher = more deformation)
        bias_field_std = (
            0.7  # this controls the maximum bias field corruption (higher = more bias)
        )

        # ---------- Resolution parameters ----------

        # This enables us to randomise the resolution of the produces images.
        # Although being only one parameter, this is crucial !!
        randomise_res = False

        # ------------------------------------------------------ Generate ------------------------------------------------------
        # instantiate BrainGenerator object
        self.brain_generator = BrainGenerator(
            labels_dir=path_label_map,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            prior_distributions=prior_distributions,
            generation_classes=generation_classes,
            subjects_prob=subjects_prob,
            output_labels=output_labels,
            n_channels=n_channels,
            target_res=target_res,
            output_shape=output_shape,
            flipping=flipping,
            scaling_bounds=scaling_bounds,
            rotation_bounds=rotation_bounds,
            shearing_bounds=shearing_bounds,
            translation_bounds=translation_bounds,
            nonlin_std=nonlin_std,
            bias_field_std=bias_field_std,
            randomise_res=randomise_res,
        )

    def __getitem__(self, x):
        im, lab = self.brain_generator.generate_brain()
        subject = tio.Subject(
            img=tio.ScalarImage(tensor=im), seg=tio.LabelMap(tensor=lab)
        )
        if self.transform:
            subject = self.transform(subject)

        if self.paired_dataset:
            im, lab = self.brain_generator.generate_brain()
            subject2 = tio.Subject(
                img=tio.ScalarImage(tensor=im), seg=tio.LabelMap(tensor=lab)
            )
            if self.transform:
                subject2 = self.transform(subject2)
            return subject, subject2, "synthbrain"
        return subject

    def __len__(self):
        return 1000000  # arbitrary number


class SynthBrain:
    def __init__(self, batch_size, num_workers, load_seg=True, transform=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_seg = load_seg
        self.transform = transform

    def get_train_loader(self):
        dataset = SynthBrainDataset(paired=True, transform=self.transform)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_pretrain_loader(self):
        dataset = SynthBrainDataset(paired=False, transform=self.transform)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader
