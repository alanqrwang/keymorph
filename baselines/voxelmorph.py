import nibabel as nib
import numpy as np
import os
import subprocess
import torch
import ants
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

import keymorph.utils as utils
import voxelmorph as vxm
import tensorflow as tf


class VoxelMorph:
    def __init__(self, perform_preaffine_register):
        super().__init__()
        self.ref_img_path = "/home/alw4013/voxelmorph/ref_256x256x256.nii.gz"
        self.synthmorph_model_path = (
            "/home/alw4013/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5"
        )
        self.perform_preaffine_register = perform_preaffine_register

    def eval(self):
        pass

    def ants_affine_register_to_ref(self, img, seg):
        moving = ants.from_numpy(img)
        seg_moving = ants.from_numpy(seg)
        fixed = ants.from_numpy(nib.load(self.ref_img_path).get_fdata())

        mytx = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="AffineFast",
            verbose=False,
        )

        moved = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            interpolator="linear",
            transformlist=mytx["fwdtransforms"],
        ).numpy()
        seg_moved = ants.apply_transforms(
            fixed=fixed,
            moving=seg_moving,
            interpolator="nearestNeighbor",
            transformlist=mytx["fwdtransforms"],
        ).numpy()

        return moved, seg_moved

    def vxm_register(self, moving, fixed, seg_moving, model_path, gpu):
        # tensorflow device handling
        device, nb_devices = vxm.tf.utils.setup_device(gpu)

        inshape = moving.shape[1:-1]
        seg_inshape = seg_moving.shape[1:-1]
        nb_feats = moving.shape[-1]
        seg_nb_feats = seg_moving.shape[-1]

        with tf.device(device):
            # load model and predict
            config = dict(inshape=inshape, input_model=None)
            model = vxm.networks.VxmDense.load(model_path, **config)
            warp = model.register(moving, fixed)
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict(
                [moving, warp]
            )
            seg_moved = vxm.networks.Transform(
                seg_inshape, nb_feats=seg_nb_feats
            ).predict([seg_moving, warp])

        return moved, seg_moved, warp

    def vxm_register_cmd(self, moving, fixed, output_dir):
        moving_path = os.path.join(output_dir, "synthmorph_img_m.nii.gz")
        fixed_path = os.path.join(output_dir, "synthmorph_img_f.nii.gz")
        align_path = os.path.join(output_dir, "synthmorph_img_a.nii.gz")
        warp_path = os.path.join(output_dir, "synthmorph_warp.nii.gz")
        nib.save(nib.Nifti1Image(moving, np.eye(4)), moving_path)
        nib.save(nib.Nifti1Image(fixed, np.eye(4)), fixed_path)

        cmd = f"/home/alw4013/voxelmorph/scripts/tf/register.py --moving {moving_path} --fixed {fixed_path} --moved {align_path} --model {self.synthmorph_model_path} --gpu 0 --warp {warp_path}"
        shell_command(cmd)

        return nib.load(warp_path).get_fdata()

    def __call__(self, fixed, moving, transform_type="dense", **kwargs):
        return self.pairwise_register(fixed, moving, transform_type, **kwargs)

    def _vxm_preprocess_imgs(self, img):
        """img is (L, W, H) numpy array"""
        return np.rot90(img, k=1, axes=(1, 2))

    def pairwise_register(
        self, img_f, img_m, transform_type="dense", cmd_line=False, **kwargs
    ):
        original_device = img_f.device
        assert len(img_f) == 1, "Fixed image should be a single image"
        assert len(img_m) == 1, "Moving image should be a single image"

        save_dir = kwargs["save_dir"]
        seg_m = kwargs["seg_m"].argmax(1)[None].float()
        seg_f = kwargs["seg_f"].argmax(1)[None].float()

        # Rotate by 90 degrees to match orientation of reference image
        img_f = self._vxm_preprocess_imgs(img_f[0, 0].cpu().detach().numpy())
        img_m = self._vxm_preprocess_imgs(img_m[0, 0].cpu().detach().numpy())
        seg_f = self._vxm_preprocess_imgs(seg_f[0, 0].cpu().detach().numpy())
        seg_m = self._vxm_preprocess_imgs(seg_m[0, 0].cpu().detach().numpy())

        # Dictionary of results
        result_dict = {}

        for ttype in transform_type:
            print(ttype)

            start_time = time.time()
            if self.perform_preaffine_register:
                # Perform affine registration to reference image
                img_m, seg_m = self.ants_affine_register_to_ref(img_m, seg_m)
                img_f, seg_f = self.ants_affine_register_to_ref(img_f, seg_f)
            preaffine_register_time = time.time() - start_time

            # Crop all images to save memory
            img_m = img_m[48:-48, 48:-48, 32:-32]
            seg_m = seg_m[48:-48, 48:-48, 32:-32]
            img_f = img_f[48:-48, 48:-48, 32:-32]
            seg_f = seg_f[48:-48, 48:-48, 32:-32]

            # fig, axes = plt.subplots(2, 2, figsize=(6, 6))
            # axes[0, 0].imshow(img_f[80, :, :])
            # axes[0, 1].imshow(img_m[80, :, :])
            # axes[1, 0].imshow(seg_f[80, :, :])
            # axes[1, 1].imshow(seg_m[80, :, :])
            # plt.show()

            # Perform Synthmorph registration
            start_time = time.time()
            if cmd_line:
                img_m = img_m[None, ..., None]
                img_f = img_f[None, ..., None]
                seg_m = seg_m[None, ..., None]
                seg_f = seg_f[None, ..., None]
                displacement_field = self.vxm_register_cmd(img_m, img_f, save_dir)
            else:
                img_a, seg_a, displacement_field = self.vxm_register(
                    img_m, img_f, seg_m, self.synthmorph_model_path, 0
                )
            synthmorph_time = time.time() - start_time

            # Convert back to torch
            img_m = torch.tensor(img_m).float()
            img_f = torch.tensor(img_f).float()
            seg_m = torch.tensor(seg_m).float()
            seg_f = torch.tensor(seg_f).float()
            displacement_field = torch.tensor(displacement_field).float()

            # Pad images to original size
            padding = (
                32,
                32,
                48,
                48,
                48,
                48,
            )
            img_m = F.pad(img_m, padding, "constant", 0)[None, None]
            img_f = F.pad(img_f, padding, "constant", 0)[None, None]
            seg_m = F.pad(seg_m, padding, "constant", 0)[None, None]
            seg_f = F.pad(seg_f, padding, "constant", 0)[None, None]
            displacement_field = F.pad(
                displacement_field, (0, 0) + padding, "constant", 0
            )[None]

            grid = utils.displacement2flow(displacement_field)
            img_a = utils.align_img(grid, img_m)
            seg_a = utils.align_img(grid, seg_m, mode="nearest")

            # fig, axes = plt.subplots(2, 3, figsize=(9, 6))
            # axes[0, 0].imshow(img_f[0, 0, 128, :, :])
            # axes[0, 1].imshow(img_m[0, 0, 128, :, :])
            # axes[0, 2].imshow(img_a[0, 0, 128, :, :])
            # axes[1, 0].imshow(seg_f[0, 0, 128, :, :])
            # axes[1, 1].imshow(seg_m[0, 0, 128, :, :])
            # axes[1, 2].imshow(seg_a[0, 0, 128, :, :])
            # plt.show()

            # Convert segmentations back to one-hot
            seg_m = (
                F.one_hot(seg_m[0, 0].long())
                .permute(3, 0, 1, 2)[None]
                .float()
                .to(original_device)
            )
            seg_f = (
                F.one_hot(seg_f[0, 0].long())
                .permute(3, 0, 1, 2)[None]
                .float()
                .to(original_device)
            )
            seg_a = (
                F.one_hot(seg_a[0, 0].long())
                .permute(3, 0, 1, 2)[None]
                .float()
                .to(original_device)
            )

            res = {
                "align_type": ttype,
                "img_m": img_m,
                "img_f": img_f,
                "seg_m": seg_m,
                "seg_f": seg_f,
                "img_a": img_a,
                "seg_a": seg_a,
                "grid": grid.to(original_device),
                "time_synthmorph": synthmorph_time,
                "time_preaffine_register": preaffine_register_time,
                "time": synthmorph_time + preaffine_register_time,
            }
            result_dict[ttype] = res

        return result_dict

    def groupwise_register(self, inputs, transform_type="rigid", **kwargs):
        pass


def shell_command(command):
    print("RUNNING", command)
    subprocess.run(command, shell=True)
