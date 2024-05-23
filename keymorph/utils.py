import re
import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict
import os
import argparse
import json

try:
    import wandb
except ImportError as e:
    pass


def align_img(grid, x, mode="bilinear"):
    return F.grid_sample(
        x,
        grid=grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )


# def align_img_elastix(transform_parameters, x):
#     """x: (dim1, dim2, dim3) or (dim1, dim2)
#     Returns: (dim1, dim2, dim3) or (dim1, dim2)"""
#     x = x.cpu().detach().numpy().astype(np.float32)
#     x = itk.image_view_from_array(x)
#     transform_parameters.SetParameter("FinalBSplineInterpolationOrder", "0")
#     result_image = itk.transformix_filter(x, transform_parameters)
#     res = torch.tensor(itk.array_view_from_image(result_image))
#     return res


def displacement2flow(displacement_field):
    """displacement_field: (N, D, H, W, 3).
    Assumes original space is physical space (voxel units), 256x256x256."""
    W, H, D = displacement_field.shape[1:-1]

    # Step 1: Create the original grid for 3D
    coords_x, coords_y, coords_z = torch.meshgrid(
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, D),
        indexing="ij",
    )
    coords = torch.stack([coords_z, coords_y, coords_x], dim=-1)  # Shape: (D, H, W, 3)
    coords = coords.unsqueeze(0)  # Shape: (N, 3, D, H, W), N=1

    # Step 2: Normalize the displacement field
    # Convert physical displacement values to the [-1, 1] range
    # Assuming the displacement field is given in voxel units (physical coordinates)
    for i, dim_size in enumerate(
        [W, H, D]
    ):  # Note the order matches x, y, z as per the displacement_field
        # Normalize such that the displacement of 1 full dimension length corresponds to a move from -1 to 1
        displacement_field[..., i] = 2 * displacement_field[..., i] / (dim_size - 1)

    # Step 3: Add the displacement field to the original grid to get the transformed coordinates
    return coords + displacement_field


def rescale_intensity(array, out_range=(0, 1), percentiles=(0, 100)):
    if isinstance(array, torch.Tensor):
        array = array.float()

    if percentiles != (0, 100):
        cutoff = np.percentile(array, percentiles)
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]
    in_min = array.min()
    in_range = array.max() - in_min
    out_min = out_range[0]
    out_range = out_range[1] - out_range[0]

    array -= in_min
    array /= in_range
    array *= out_range
    array += out_min
    return array


def parse_test_mod(mod):
    if isinstance(mod, str):
        mod1, mod2 = mod.split("_")
    else:
        mod1, mod2 = mod
    return mod1, mod2


def parse_test_aug(aug):
    if "rot" in aug:
        if aug == "rot0":
            rot_aug = 0
        elif aug == "rot45":
            rot_aug = math.pi / 4
        elif aug == "rot90":
            rot_aug = math.pi / 2
        elif aug == "rot135":
            rot_aug = 3 * math.pi / 4
        elif aug == "rot180":
            rot_aug = math.pi
        aug_param = (0, 0, rot_aug, 0)
    else:
        raise NotImplementedError()

    return aug_param


def str_or_float(x):
    try:
        return float(x)
    except ValueError:
        return x


def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}


def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(**config.wandb_kwargs)
    wandb.config.update(config)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    state = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = state["state_dict"]

    # Sometimes the model is saved with "backbone" prefix
    new_state_dict = {
        key.replace(".backbone", ""): value for key, value in state_dict.items()
    }
    missing_keys, _ = model.backbone.load_state_dict(new_state_dict, strict=True)
    print("Missing keys when loading checkpoint: ", missing_keys)

    res = (state, model)

    if optimizer:
        optimizer.load_state_dict(state["optimizer"])
        res += (optimizer,)
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
        res += (scheduler,)

    return res


# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split("=")
            if value_str.replace("-", "").isnumeric():
                processed_val = int(value_str)
            elif value_str.replace("-", "").replace(".", "").isnumeric():
                processed_val = float(value_str)
            elif value_str in ["True", "true"]:
                processed_val = True
            elif value_str in ["False", "false"]:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val


def sample_valid_coordinates(x, num_points, dim):
    """
    x: input img, (1,1,dim1,dim2) or (1,1,dim1,dim2,dim3)
    num_points: how many points within the brain
    dim: Dimension, either 2 or 3

    Returns:
      points: Normalized coordinates in [0, 1], (1, num_points, dim)
    """
    if dim == 2:
        coords = sample_valid_coordinates_2d(x, num_points)
    elif dim == 3:
        coords = sample_valid_coordinates_3d(x, num_points)
    else:
        raise NotImplementedError
    return coords


def sample_valid_coordinates_2d(x, num_points):
    eps = 0
    mask = x > eps
    indices = []
    for _ in range(num_points):
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            sample[:, :, dim1, dim2] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                indices.append([dim2 / x.size(3), dim1 / x.size(2)])

    return torch.tensor(indices).view(1, num_points, 2)


def sample_valid_coordinates_3d(x, num_points):
    eps = 1e-1
    mask = x > eps
    indices = []
    for _ in range(num_points):
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            dim3 = np.random.randint(0, x.size(4))
            sample[:, :, dim1, dim2, dim3] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                indices.append([dim3 / x.size(4), dim2 / x.size(3), dim1 / x.size(2)])

    return torch.tensor(indices).view(1, num_points, 3)


def summary(network):
    """Print model summary."""
    print("")
    print("Model Summary")
    print("---------------------------------------------------------------")
    for name, _ in network.named_parameters():
        print(name)
    print(
        "Total parameters:",
        sum(p.numel() for p in network.parameters() if p.requires_grad),
    )
    print("---------------------------------------------------------------")
    print("")


def save_dict_as_json(dict, save_path):
    with open(save_path, "w") as outfile:
        json.dump(dict, outfile, sort_keys=True, indent=4)


def load_dict_from_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def get_latest_epoch_file(directory_path):
    max_epoch = -1
    latest_epoch_file = None

    # Compile a regular expression pattern to extract the epoch number
    epoch_pattern = re.compile(r"epoch(\d+)_trained_model.pth.tar")

    # List all files in the given directory
    for filename in os.listdir(directory_path):
        match = epoch_pattern.match(filename)
        if match:
            # Extract the epoch number and convert it to an integer
            epoch_num = int(match.group(1))
            # Update the max_epoch and latest_epoch_file if this file has a larger epoch number
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_epoch_file = filename

    # Return the path of the file with the largest epoch number
    if latest_epoch_file is not None:
        return os.path.join(directory_path, latest_epoch_file)
    else:
        return None


def one_hot_eval(asegs):
    subset_regs = [
        [0, 24],  # Background and CSF
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
        [16, 16],  # Brain-Stem
    ]

    N, _, dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(N, chs, dim1, dim2, dim3)

    for i, s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1])
        one_hot[:, i, :, :, :] = (combined_vol * 1).float()

    mask = one_hot.sum(1).squeeze()
    ones = torch.ones_like(mask)
    non_roi = ones - mask
    one_hot[:, -1, :, :, :] = non_roi

    assert (
        one_hot.sum(1).sum() == N * dim1 * dim2 * dim3
    ), "One-hot encoding does not add up to 1"
    return one_hot


def one_hot(seg):
    """Converts a segmentation to one-hot encoding.

    seg: (N, 1, D, H, W) tensor of integer labels
    """
    return F.one_hot(seg)[:, 0].permute(0, 4, 1, 2, 3)


def one_hot_subsampled_pair(seg1, seg2, subsample_num=14):
    # Determine the unique integers in both segmentations
    unique_vals1 = np.unique(seg1.cpu().detach().numpy())
    unique_vals2 = np.unique(seg2.cpu().detach().numpy())

    # Take intersection
    unique_vals = np.intersect1d(unique_vals1, unique_vals2, assume_unique=True)

    # Subsample (if more than subsample_num values)
    if len(unique_vals) > subsample_num:
        selected_vals = np.random.choice(unique_vals, subsample_num, replace=False)
    else:
        selected_vals = unique_vals
        subsample_num = len(unique_vals)

    # Step 3: Create a mapping for the selected integers
    mapping = {val: i for i, val in enumerate(selected_vals)}

    # Step 4: Apply one-hot encoding to both segmentations with the mapping
    def apply_one_hot(asegs, mapping, subsample_num):
        one_hot_maps = torch.zeros(
            (asegs.shape[0], subsample_num, *asegs.shape[2:]),
            dtype=torch.float32,
            device=asegs.device,
        )
        for val, new_idx in mapping.items():
            one_hot_maps[:, new_idx] = (asegs == val).float()
        return one_hot_maps

    one_hot_maps1 = apply_one_hot(seg1, mapping, subsample_num)
    one_hot_maps2 = apply_one_hot(seg2, mapping, subsample_num)

    return one_hot_maps1, one_hot_maps2
