import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology
from scipy.stats import loguniform
import time
import os
from torch.utils import checkpoint

from keymorph.keypoint_aligners import RigidKeypointAligner, AffineKeypointAligner, TPS
from keymorph.layers import (
    LinearRegressor2d,
    LinearRegressor3d,
    CenterOfMass2d,
    CenterOfMass3d,
)
from keymorph.utils import (
    str_or_float,
    rescale_intensity,
    convert_points_norm2voxel,
    convert_points_voxel2norm,
)


class KeyMorph(nn.Module):
    def __init__(
        self,
        backbone,
        num_keypoints,
        dim,
        keypoint_layer="com",
        max_train_keypoints=None,
        use_amp=False,
        use_checkpoint=False,
        weight_keypoints=None,
        align_keypoints_in_real_world_coords=False,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super(KeyMorph, self).__init__()
        self.backbone = backbone
        self.num_keypoints = num_keypoints
        self.dim = dim
        if keypoint_layer == "com":
            if dim == 2:
                self.keypoint_layer = CenterOfMass2d(indexing="ij")
            else:
                self.keypoint_layer = CenterOfMass3d(indexing="ij")
        else:
            if dim == 2:
                self.keypoint_layer = LinearRegressor2d(num_keypoints * dim)
            else:
                self.keypoint_layer = LinearRegressor3d(num_keypoints * dim)
        self.max_train_keypoints = max_train_keypoints
        self.use_amp = use_amp
        self.use_checkpoint = use_checkpoint

        # Keypoint alignment module
        self.supported_transform_type = ["rigid", "affine", "tps"]

        # Weight keypoints
        assert weight_keypoints in [None, "variance", "power"]
        self.weight_keypoints = weight_keypoints
        if self.weight_keypoints == "variance":
            self.scales = nn.Parameter(torch.ones(num_keypoints))
            self.biases = nn.Parameter(torch.zeros(num_keypoints))

        self.align_keypoints_in_real_world_coords = align_keypoints_in_real_world_coords

    def weight_by_variance(self, feat1, feat2):
        feat1, feat2 = F.relu(feat1), F.relu(feat2)
        if self.dim == 2:
            var1 = torch.var(feat1, dim=(2, 3))
            var2 = torch.var(feat2, dim=(2, 3))
        else:
            var1 = torch.var(feat1, dim=(2, 3, 4))
            var2 = torch.var(feat2, dim=(2, 3, 4))

        # Higher var -> lower weight
        # Lower var -> higher weight
        weights1 = 1 / (self.scales * var1 + self.biases)
        weights2 = 1 / (self.scales * var2 + self.biases)

        # Aggregate variances of moving and fixed heatmaps
        weights = weights1 * weights2

        # Return normalized weights
        return weights / weights.sum(dim=1)

    def weight_by_power(self, feat1, feat2):
        feat1, feat2 = F.relu(feat1), F.relu(feat2)
        bs, n_ch = feat1.shape[0], feat1.shape[1]
        feat1 = feat1.reshape(bs, n_ch, -1)
        feat2 = feat2.reshape(bs, n_ch, -1)

        # Higher power -> higher weight
        power1 = feat1.sum(dim=-1)
        power2 = feat2.sum(dim=-1)

        # Aggregate power of moving and fixed heatmaps
        weights = power1 * power2

        # Return normalized weights
        return weights / weights.sum(dim=1)

    def get_keypoints(self, img, return_feat=False):
        """Get keypoints from an image"""
        feat = self.backbone(img)
        points = self.keypoint_layer(feat)
        if return_feat:
            return points, feat
        return points

    @staticmethod
    def _convert_tps_lmbda(num_samples, tps_lmbda):
        """Return a tensor of size num_samples composed of specified tps_lmbda values. Values may be constant or sampled from a distribution.

        :param num_samples: int, Number of samples
        :param tps_lmbda: float or str
        """
        if tps_lmbda == "uniform":
            lmbda = torch.rand(num_samples) * 10
        elif tps_lmbda == "lognormal":
            lmbda = torch.tensor(np.random.lognormal(size=num_samples))
        elif tps_lmbda == "loguniform":
            a, b = 1e-6, 10
            lmbda = torch.tensor(loguniform.rvs(a, b, size=num_samples))
        else:
            lmbda = torch.tensor(tps_lmbda).repeat(num_samples)
        return lmbda

    @staticmethod
    def is_supported_transform_type(s):
        if s in ["affine", "rigid"]:
            return True
        elif re.match(r"^tps_.*$", s):
            return True
        return False

    def forward(self, img_f, img_m, transform_type="affine", **kwargs):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        return_aligned_points = kwargs["return_aligned_points"]
        if not isinstance(transform_type, (list, tuple)):
            transform_type = [transform_type]
        if self.training:
            assert (
                len(transform_type) == 1
            ), "Only one alignment type allowed in training"
        assert all(
            self.is_supported_transform_type(s) for s in transform_type
        ), "Invalid transform_type"

        if self.align_keypoints_in_real_world_coords:
            # TODO: fix this
            assert all(
                [t in ["rigid", "affine"] for t in transform_type]
            ), "TPS alignment not supported in real world coordinates yet."
            aff_f = kwargs["aff_f"]
            aff_m = kwargs["aff_m"]
        else:
            aff_f = None
            aff_m = None

        assert img_f.shape[1] == 1, "Image dimension must be 1"
        assert img_m.shape[1] == 1, "Image dimension must be 1"

        start_time = time.time()

        with torch.amp.autocast(
            device_type="cuda", enabled=self.use_amp, dtype=torch.float16
        ):
            # Extract keypoints
            points_f, feat_f = self.get_keypoints(img_f, return_feat=True)
            points_m, feat_m = self.get_keypoints(img_m, return_feat=True)

            if self.weight_keypoints == "power":
                if self.use_checkpoint:
                    weights = checkpoint.checkpoint(
                        self.weight_by_power, feat_f, feat_m
                    )
                else:
                    weights = self.weight_by_power(feat_f, feat_m)
            else:
                weights = None

        keypoint_extract_time = time.time() - start_time

        if self.align_keypoints_in_real_world_coords:
            # Convert points from normalized to real world coordinates
            shape_m = torch.tensor(img_m.shape[2:]).to(img_m)
            shape_f = torch.tensor(img_f.shape[2:]).to(img_f)
            points_m = convert_points_norm2voxel(points_m, shape_m)
            points_f = convert_points_norm2voxel(points_f, shape_f)

        # Dictionary of results
        result_dict = {}

        for align_type_str in transform_type:
            start_time = time.time()
            if align_type_str.startswith("tps"):
                align_type = "tps"
                tps_lmbda = self._convert_tps_lmbda(
                    len(img_f), str_or_float(align_type_str[4:])
                ).to(img_f.device)
            else:
                align_type = align_type_str
                tps_lmbda = None

            if (
                self.training
                and align_type == "tps"
                and self.max_train_keypoints
                and self.num_keypoints > self.max_train_keypoints
            ):
                # Take mini-batch of keypoints
                key_batch_idx = np.random.choice(
                    self.num_keypoints, size=self.max_train_keypoints, replace=False
                )
                points_f = points_f[:, key_batch_idx]
                points_m = points_m[:, key_batch_idx]
                if self.weight_keypoints is not None:
                    weights = weights[:, key_batch_idx]

            # Align via keypoints
            if align_type == "rigid":
                keypoint_aligner = RigidKeypointAligner(
                    points_m=points_m,
                    points_f=points_f,
                    w=weights,
                    aff_f=aff_f,
                    aff_m=aff_m,
                    shape_f=img_f.shape,
                    shape_m=img_m.shape,
                    dim=self.dim,
                    align_in_real_world_coords=self.align_keypoints_in_real_world_coords,
                )
            elif align_type == "affine":
                keypoint_aligner = AffineKeypointAligner(
                    points_m=points_m,
                    points_f=points_f,
                    w=weights,
                    aff_f=aff_f,
                    aff_m=aff_m,
                    shape_f=img_f.shape,
                    shape_m=img_m.shape,
                    dim=self.dim,
                    align_in_real_world_coords=self.align_keypoints_in_real_world_coords,
                )
            elif align_type == "tps":
                keypoint_aligner = TPS(
                    points_m=points_m,
                    points_f=points_f,
                    lmbda=tps_lmbda,
                    w=weights,
                    dim=self.dim,
                    use_checkpoint=self.use_checkpoint,
                )

            # Get flow field
            grid = keypoint_aligner.get_flow_field(
                img_f.shape,
                compute_on_subgrids=False if self.training else True,
            )

            if return_aligned_points:
                points_a = keypoint_aligner.get_forward_transformed_points(points_m)

            if self.align_keypoints_in_real_world_coords:
                # Convert points from real world to normalized coordinates
                points_m = convert_points_voxel2norm(points_m, shape_m)
                points_f = convert_points_voxel2norm(points_f, shape_f)
                if return_aligned_points:
                    points_a = convert_points_voxel2norm(points_a, shape_f)

            align_time = time.time() - start_time
            res = {
                "grid": grid,
                "points_f": points_f,
                "points_m": points_m,
                "points_weights": weights,
                "tps_lmbda": tps_lmbda,
                "time_keypoint_extract": keypoint_extract_time,
                "time_align": align_time,
                "time": keypoint_extract_time + align_time,
            }
            if align_type in ["rigid", "affine"]:
                res["matrix"] = keypoint_aligner.transform_matrix
            if return_aligned_points:
                res["points_a"] = points_a
            result_dict[align_type_str] = res
        return result_dict

    def pairwise_register(self, *args, **kwargs):
        """Alias for forward()."""
        return self.forward(self, args, kwargs)

    def groupwise_register(self, inputs, transform_type="affine", **kwargs):
        """Groupwise registration.

        Steps:
            1. Extract keypoints from each image
            2. Find mean keypoints by repeating for num_iters:
                b. Compute mean keypoints
                c. Register each image to the mean keypoints
            3. Compute grids for each image from original extracted keypoints to mean keypoints

        inputs can be:
           - directories of images, looks for files img_*.npy
           - list of image paths
           - Torch Tensor stack of images (N, 1, D, H, W)"""

        device = kwargs["device"]
        num_iters = kwargs["num_iters"]
        log_to_console = kwargs["log_to_console"]

        # Load images and segmentations
        if isinstance(inputs, str):
            save_dir = kwargs["save_dir"]
            inputs = sorted(
                [
                    os.path.join(inputs, f)
                    for f in os.listdir(inputs)
                    if f.endswith(".npy")
                ]
            )
        else:
            save_dir = None

        def _groupwise_register_step(
            group_points, keypoint_aligner, lmbda, weights=None
        ):
            """One step of groupwise registration.

            Args:
                group_points: tensor of shape (num_subjects, num_points, dim)
                keypoint_aligner: Keypoint aligner object
                lmbda: Lambda value for TPS
                grid_shape: Grid on which to resample

            Returns:
                grids: All grids for each subject in the group
                points: All transformed points for each subject in the group
            """
            # Compute mean points, to be used as fixed points
            mean_points = torch.mean(group_points, dim=0, keepdim=True)

            # Register each subject's points to the mean points
            new_points = torch.zeros_like(group_points, device=group_points.device)
            for i in range(len(group_points)):
                points_m = group_points[i : i + 1]
                points_a = keypoint_aligner.points_from_points(
                    points_m,
                    mean_points,
                    points_m,
                    lmbda=lmbda,
                    weights=weights,
                )
                new_points[i : i + 1] = points_a
            return new_points, mean_points

        # Load images and segmentations
        group_points = []
        if log_to_console:
            print("Extracting keypoints...")
        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                img_m = torch.tensor(np.load(inputs[i])).float()
            else:
                img_m = inputs[i : i + 1]
            img_m = img_m.to(device)
            points = self.get_keypoints(img_m)

            # if args.weighted_kp_align == "variance":
            #     weights = registration_model.weight_by_variance(feat_f, feat_m)
            # elif args.weighted_kp_align == "power":
            #     weights = registration_model.weight_by_power(feat_f, feat_m)
            # else:
            #     weights = None
            weights = None  # TODO: support weighted groupwise registration??
            group_points.append(points.detach())
            if log_to_console:
                print(f"-> Extracted keypoints from subject {i+1}/{len(inputs)}")

        group_points = torch.cat(group_points, dim=0)

        result_dict = {}
        for align_type_str in transform_type:
            if log_to_console:
                print(f"\nAligning keypoints via {align_type_str}...")
            start_time = time.time()
            if align_type_str.startswith("tps"):
                align_type = "tps"
                tps_lmbda = self._convert_tps_lmbda(
                    len(img_m), str_or_float(align_type_str[4:])
                ).to(img_m.device)
            else:
                align_type = align_type_str
                tps_lmbda = None

            # Align via keypoints
            if align_type == "rigid":
                keypoint_aligner = self.rigid_aligner
            elif align_type == "affine":
                keypoint_aligner = self.affine_aligner
            elif align_type == "tps":
                keypoint_aligner = self.tps_aligner

            curr_points = group_points.clone()
            for j in range(num_iters):
                next_points, mean_points = _groupwise_register_step(
                    curr_points,
                    keypoint_aligner,
                    tps_lmbda,
                    weights=weights,
                )
                curr_points = next_points.clone()
                if log_to_console:
                    print(f"-> Iteration {j+1}/{num_iters}")

            register_time = time.time() - start_time
            res = {
                "time": register_time,
                "grouppoints_m": group_points,
                "grouppoints_a": curr_points,
            }

            save_results_to_disk = kwargs["save_results_to_disk"]
            if save_results_to_disk and save_dir:
                # Compute grid and save to disk, one at a time
                for i in range(len(group_points)):
                    points_m = group_points[i : i + 1]
                    grid = keypoint_aligner.grid_from_points(
                        points_m,
                        mean_points,
                        img_m.shape,
                        lmbda=tps_lmbda,
                        weights=weights,
                        compute_on_subgrids=True,
                    )
                    grid_save_path = f"{save_dir}/{align_type_str}_grid_{i:03}.npy"
                    if log_to_console:
                        print(
                            f"-> Saving grid {i+1}/{len(curr_points)} to {grid_save_path}"
                        )
                    np.save(
                        grid_save_path,
                        grid.cpu().detach().numpy(),
                    )
            else:
                # Compute all grids and return them all in the result dictionary
                grids = []
                for i in range(len(group_points)):
                    points_m = group_points[i : i + 1]
                    grid = keypoint_aligner.grid_from_points(
                        points_m,
                        mean_points,
                        img_m.shape,
                        lmbda=tps_lmbda,
                        weights=weights,
                        compute_on_subgrids=True,
                    )
                    grids.append(grid)
                res["groupgrids"] = torch.cat(grids, dim=0)
            result_dict[align_type_str] = res

        if log_to_console:
            print("Groupwise registration complete!")
        return result_dict


class Simple_Unet(nn.Module):
    """
    Neural network for Brain Extractor

    Arguments
    ---------
    input_ch   : input channel to the network
    out_ch     : output dimension of the network
    use_in     : use instance norm
    enc_nf     : list of int for the encoder filter size
    dec_nf     : list of int for the decoder filter size
    Return
    ------
        model : torch model
    """

    def __init__(self, input_ch, out_ch, use_in, enc_nf, dec_nf):
        super(Simple_Unet, self).__init__()

        self.down = torch.nn.MaxPool3d(2, 2)

        self.block0 = simple_block(input_ch, enc_nf[0], use_in)
        self.block1 = simple_block(enc_nf[0], enc_nf[1], use_in)
        self.block2 = simple_block(enc_nf[1], enc_nf[2], use_in)
        self.block3 = simple_block(enc_nf[2], enc_nf[3], use_in)

        self.block4 = simple_block(enc_nf[3], dec_nf[0], use_in)

        self.block5 = simple_block(dec_nf[0] * 2, dec_nf[1], use_in)
        self.block6 = simple_block(dec_nf[1] * 2, dec_nf[2], use_in)
        self.block7 = simple_block(dec_nf[2] * 2, dec_nf[3], use_in)
        self.block8 = simple_block(dec_nf[3] * 2, out_ch, use_in)

        self.conv = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x_in):
        # Model
        x0 = self.block0(x_in)
        x1 = self.block1(self.down(x0))
        x2 = self.block2(self.down(x1))
        x3 = self.block3(self.down(x2))

        x = self.block4(self.down(x3))
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x = torch.cat([x, x3], 1)
        x = self.block5(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x = torch.cat([x, x2], 1)
        x = self.block6(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x = torch.cat([x, x1], 1)
        x = self.block7(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        x = torch.cat([x, x0], 1)
        x = self.block8(x)

        out = self.conv(x)

        return out


class simple_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_in):
        super(simple_block, self).__init__()

        self.use_in = use_in

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.InstanceNorm3d(out_channels)

        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_in:
            out = self.bn1(out)
        out = self.activation(out)
        return out


"""Helper function for brain extractor"""


def clean_mask(mask, threshold=0.2):
    """
    Remove small predicted segmentation. It finds the largest connected component.
    If there are other region/islands that is less than threshold percentage in size,
    remove those region.

    Arguments
    ---------
    mask         : numpy 3D binary mask for use for brain extraction
    threshold    : remove if not size(obj)/size(largest_component) > threshold

    Return
    ------
        new_mask : cleaned up mask with smaller regions removed
    """

    connected = morphology.label(mask)
    islands = np.unique(connected)[1:]

    islands_size = {}
    max_size = []
    for i in islands:
        size = (connected == i).sum()
        islands_size[i] = size
        max_size += [size]

    max_size = np.max(max_size)

    _island_size = islands_size.copy()
    for key in _island_size:
        if not islands_size[key] / max_size > threshold:
            islands_size.pop(key, None)

    new_mask = np.zeros_like(mask).astype("uint8")
    for key in islands_size:
        new_mask += (connected == key).astype("uint8")

    return new_mask
