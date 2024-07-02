import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def str_or_float(x):
    try:
        return float(x)
    except ValueError:
        return x


def align_img(grid, x, mode="bilinear"):
    return F.grid_sample(
        x,
        grid=grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )


def displacement2pytorchflow(displacement_field):
    """Converts displacement field in index coordinates into a flow-field usable by F.grid_sample.
    Assumes original space is in index (voxel) units, 256x256x256.
    Output will be in the [-1, 1] space.

    :param: displacement_field: (N, D, H, W, 3).
    """
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


def pytorchflow2displacement(flow):
    """Converts pytorch flow-field in [-1, 1] to a displacement field in index (voxel) units

    :param: flow: (N, D, H, W, 3).
    """
    flow = flow.permute(0, 4, 1, 2, 3)  # Bring channels to second dimension
    shape = flow.shape[2:]

    # Scale normalized flow to pixel indices
    for i in range(3):
        flow[:, i, ...] = (flow[:, i, ...] + 1) / 2 * (shape[i] - 1)

    # Create an image grid for the target size
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids, dim=0).unsqueeze(0).to(flow.device, dtype=torch.float32)

    # Calculate displacements from the image grid
    disp = flow - grid
    return disp


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


def sample_valid_coordinates(x, num_points, dim, point_space="norm", indexing="xy"):
    """
    x: input img, (1,1,dim1,dim2) or (1,1,dim1,dim2,dim3)
    num_points: number of points
    dim: Dimension, either 2 or 3

    Returns:
      points: Normalized coordinates in [0, 1], (1, num_points, dim)
    """
    if dim == 2:
        coords = sample_valid_coordinates_2d(x, num_points, point_space=point_space)
    elif dim == 3:
        coords = sample_valid_coordinates_3d(x, num_points, point_space=point_space)
    else:
        raise NotImplementedError
    if indexing == "ij":
        coords = coords.flip(-1)
    return coords


def sample_valid_coordinates_2d(x, num_points, point_space="norm"):
    eps = 0
    mask = x > eps
    indices = []
    for i in range(num_points):
        print(f"{i+1}/{num_points}")
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            sample[:, :, dim1, dim2] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                if point_space == "norm":
                    indices.append([dim2 / x.size(3), dim1 / x.size(2)])
                else:
                    indices.append([dim2, dim1])

    return torch.tensor(indices).view(1, num_points, 2)


def sample_valid_coordinates_3d(x, num_points, point_space="norm"):
    eps = 1e-1
    mask = x > eps
    indices = []
    for i in range(num_points):
        print(f"{i+1}/{num_points}")
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            dim3 = np.random.randint(0, x.size(4))
            sample[:, :, dim1, dim2, dim3] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                if point_space == "norm":
                    indices.append(
                        [dim3 / x.size(4), dim2 / x.size(3), dim1 / x.size(2)]
                    )
                else:
                    indices.append([dim3, dim2, dim1])

    return torch.tensor(indices).view(1, num_points, 3)


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


def convert_points_norm2voxel(points, grid_sizes):
    """
    Rescale points from [-1, 1] to a uniform voxel grid with different sizes along each dimension.

    Args:
        points (bs, num_points, dim): Array of points in the normalized space [-1, 1].
        grid_sizes (bs, dim): Array of grid sizes for each dimension.

    Returns:
        Array of points in voxel space.
    """
    grid_sizes = torch.tensor(grid_sizes).to(points.device)
    assert grid_sizes.shape[-1] == points.shape[-1], "Dimensions don't match"
    translated_points = points + 1
    scaled_points = (translated_points * grid_sizes) / 2
    rescaled_points = scaled_points - 0.5
    return rescaled_points


def convert_points_voxel2norm(points, grid_sizes):
    """
    Reverse rescale points from a uniform voxel grid to the normalized space [-1, 1].

    Args:
        points (bs, num_points, dim): Array of points in the voxel space.
        grid_sizes (bs, dim): Array of grid sizes for each dimension.

    Returns:
        Array of points in the normalized space [-1, 1].
    """
    grid_sizes = torch.tensor(grid_sizes).to(points.device)
    assert grid_sizes.shape[-1] == points.shape[-1], "Dimensions don't match"
    rescaled_points_shifted = points + 0.5
    normalized_points = (2 * rescaled_points_shifted / grid_sizes) - 1
    return normalized_points


def convert_points_voxel2real(points, affine):
    """
    Convert points from uniform voxel grid to real world coordinates.

    Args:
        points (bs, num_points, dim): points in the normalized space [-1, 1].
        affine (bs, dim+1, dim+1): Square affine matrix
    """
    batch_size, num_points, _ = points.shape
    # Convert to homogeneous coordinates
    ones = torch.ones(batch_size, num_points, 1).to(points.device)
    points = torch.cat([points, ones], dim=2)

    # Apply the affine matrix
    real_world_points = torch.bmm(affine, points.permute(0, 2, 1)).permute(0, 2, 1)

    # Remove the homogeneous coordinate
    return real_world_points[:, :, :-1]


def convert_points_real2voxel(points, affine):
    """
    Convert points from uniform voxel grid to real world coordinates.

    Args:
        points (bs, num_points, dim): points in the normalized space [-1, 1].
        affine (bs, dim+1, dim+1): Square affine matrix
    """

    batch_size, num_points, _ = points.shape

    # Step 1: Convert to homogeneous coordinates by adding a column of ones
    ones = torch.ones(batch_size, num_points, 1).to(points.device)
    points = torch.cat([points, ones], dim=2)

    # Step 2: Compute the inverse affine matrices
    inverse_affine = torch.inverse(affine)

    # Step 3: Apply the inverse affine matrix
    points = torch.bmm(inverse_affine, points.permute(0, 2, 1)).permute(0, 2, 1)

    # Remove the homogeneous coordinate
    return points[:, :, :-1]


def convert_points_norm2real(points, affine_matrices, voxel_sizes):
    """
    Converts points from voxel coordinates (in the range [-1, 1]) to real world coordinates using batch-specific affine matrices.

    Args:
        points (torch.Tensor): The tensor of points with shape (batch_size, num_points, dimension).
        affine_matrices (torch.Tensor): The batch of affine matrices with shape (batch_size, dimension+1, dimension+1).
        voxel_sizes (torch.Tensor): The batch of voxel sizes with shape (batch_size, dimension).

    Returns:
        torch.Tensor: The points in real world coordinates with shape (batch_size, num_points, dimension).
    """
    denormalized_points = convert_points_norm2voxel(points, voxel_sizes)
    return convert_points_voxel2real(denormalized_points, affine_matrices)


def convert_points_real2norm(real_world_points, affine_matrices, voxel_sizes):
    """
    Converts points from real world coordinates to voxel coordinates (in the range [-1, 1]) using batch-specific affine matrices.

    Args:
        real_world_points (torch.Tensor): The tensor of real world points with shape (batch_size, num_points, dimension).
        affine_matrices (torch.Tensor): The batch of affine matrices with shape (batch_size, dimension+1, dimension+1).
        voxel_sizes (torch.Tensor): The batch of voxel sizes with shape (batch_size, dimension).

    Returns:
        torch.Tensor: The points in voxel coordinates with shape (batch_size, num_points, dimension).
    """
    voxel_points = convert_points_real2voxel(real_world_points, affine_matrices)
    return convert_points_voxel2norm(voxel_points, voxel_sizes)


def convert_flow_voxel2norm(flow, dim_sizes):
    """
    Parameters:
    - flow (torch.Tensor): The flow field tensor of shape (N, D, H, W, 3) in voxel coordinates.
    """
    # Convert physical displacement values to the [-1, 1] range
    # Assuming the displacement field is given in voxel units (physical coordinates)
    for i, dim_size in enumerate(
        dim_sizes
    ):  # Note the order matches x, y, z as per the displacement_field
        # Normalize such that the displacement of 1 full dimension length corresponds to a move from -1 to 1
        flow[..., i] = 2 * (flow[..., i] + 0.5) / dim_size - 1

    return flow


class AffineAligner(nn.Module):
    """Keypoint aligner for transformations that can be represented as a matrix.

    All keypoints must be passed in with matrix/image/voxel ordering (aka 'ij' indexing).
    When indexing into a 3D volume, volume[points[0], points[1], points[2]]
    In a flattened grid, the first dimension varies the fastest.
    """

    def __init__(
        self,
        matrix=None,
        inverse_matrix=None,
        dim=3,
        grid_space="norm",
        m_shape=None,
    ):
        """
        Args:
            One of matrix or inverse_matrix must be provided
            dim: Dimensionality of the points (2 or 3)
        """
        super().__init__()
        self.dim = dim
        assert grid_space in ["norm", "voxel"]
        self.grid_space = grid_space
        if grid_space == "voxel":
            assert m_shape is not None, "Need m_shape for voxel space"
            self.m_shape = m_shape
        if matrix is not None and inverse_matrix is None:
            self.transform_matrix = matrix
            self.inverse_transform_matrix = torch.inverse(matrix)
        elif matrix is None and inverse_matrix is not None:
            self.inverse_transform_matrix = inverse_matrix
            self.transform_matrix = torch.inverse(inverse_matrix)
        else:
            raise ValueError("Only one of matrix or inverse_matrix should be provided")

    def _square(self, matrix):
        square = torch.eye(self.dim + 1)[None]
        square[:, : self.dim, : self.dim + 1] = matrix
        return square

    def uniform_voxel_grid(self, grid_shape):
        if self.dim == 2:
            x = torch.arange(grid_shape[2])
            y = torch.arange(grid_shape[3])
            grid = torch.meshgrid(x, y, indexing="ij")
        else:
            x = torch.arange(grid_shape[2])
            y = torch.arange(grid_shape[3])
            z = torch.arange(grid_shape[4])
            grid = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack(grid, dim=-1).float()
        return grid

    def uniform_norm_grid(self, grid_shape):
        if self.dim == 2:
            x = torch.linspace(-1, 1, grid_shape[2])
            y = torch.linspace(-1, 1, grid_shape[3])
            grid = torch.meshgrid(x, y, indexing="ij")
        else:
            x = torch.linspace(-1, 1, grid_shape[2])
            y = torch.linspace(-1, 1, grid_shape[3])
            z = torch.linspace(-1, 1, grid_shape[4])
            grid = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack(grid, dim=-1).float()
        return grid

    def affine_grid(self, grid_shape):
        """Create a grid of affine transformed coordinates with specified shape to be used in F.grid_sample().

        Args:
            grid_shape (bs, 1, H, W) or (bs, 1, H, W, D): Shape of the grid to create
            m_shape: (bs, 1, H, W) or (bs, 1, H, W, D): Resolution of moving image being interpolating

        Returns:
            transformed_grid (bs, 1, H, W, D, dim): Affine grid in [-1, 1] space
        """
        # Create grid of voxel coordinates where leftmost dimension varies first
        if self.grid_space == "voxel":
            grid = self.uniform_voxel_grid(grid_shape)
        else:
            grid = self.uniform_norm_grid(grid_shape)

        # Flatten the grid to (N x dim) array of voxel coordinates
        grid_flat = (
            grid.reshape(-1, self.dim).unsqueeze(0).to(self.transform_matrix)
        )  # Add batch dimension

        moving_voxel_coords = self.get_inverse_transformed_points(grid_flat)

        transformed_grid = moving_voxel_coords.reshape(1, *grid_shape[2:], self.dim)

        if self.grid_space == "voxel":
            transformed_grid = convert_flow_voxel2norm(
                transformed_grid, self.m_shape[2:]
            )

        return transformed_grid

    def get_flow_field(
        self,
        grid_shape,
        **kwargs,
    ):
        """
        If align_in_real_world_coords is False, we create the grid directly in [-1, 1] space.
        If align_in_real_world_coords is True, we assume the points are in voxel coordinates.
            Mathematically: y = A_f^-1 A A_m x = B x
            But, F.grid_sample requires the inverse: B^-1 = A_m^-1 A^-1 A_f
            Note that self.matrix already represents the inverse of A, because
            we register the fixed keypoints to moving keypoints in fit().

        Args:
            grid_shape (bs, 1, H, W, D): Shape of the grid to create
        """
        grid = self.affine_grid(grid_shape)
        # Flip because grid_sample expects 'xy' ordering.
        # See make_base_grid_5d() in https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/AffineGridGenerator.cpp
        return grid.flip(-1)

    def get_forward_transformed_points(self, points):
        """Transforms a set of points in moving space to fixed space using the fitted matrix.
        If align_in_real_world_coords is False, computes:
            p_f = A p_m.
        If align_in_real_world_coords is True, points must be in voxel coordinates, computes:
            p_f = A_f^-1 A A_m p_m.
        """
        batch_size, num_points, _ = points.shape
        transform_matrix = self.transform_matrix[:, :-1, :]

        # Convert to homogeneous coordinates
        ones = torch.ones(batch_size, num_points, 1).to(points.device)
        points = torch.cat([points, ones], dim=2)
        points = torch.bmm(transform_matrix, points.permute(0, 2, 1)).permute(0, 2, 1)

        return points

    def get_inverse_transformed_points(self, points):
        """Transforms a set of points in fixed space to moving space using the fitted matrix.
        If align_in_real_world_coords is False, computes:
            p_m = A p_f.
        If align_in_real_world_coords is True, points must be in voxel coordinates, computes:
            p_m = A_m^-1 A^-1 A_f p_f.
        """
        batch_size, num_points, _ = points.shape
        transform_matrix = self.inverse_transform_matrix[:, :-1, :]

        # Transform real-world coordinates to the moving image space using the registration affine
        # Convert to homogeneous coordinates
        ones = torch.ones(batch_size, num_points, 1).to(points.device)
        points = torch.cat([points, ones], dim=2)
        points = torch.bmm(transform_matrix, points.permute(0, 2, 1)).permute(0, 2, 1)

        return points


# import torch
# from itertools import product

# assert hasattr(torch, "bucketize"), "Need torch >= 1.7.0; install at pytorch.org"


# class RegularGridInterpolator:

#     def __init__(self, points, values):
#         self.points = points
#         self.values = values

#         assert isinstance(self.points, tuple) or isinstance(self.points, list)
#         assert isinstance(self.values, torch.Tensor)

#         self.ms = list(self.values.shape)
#         self.n = len(self.points)

#         assert len(self.ms) == self.n

#         for i, p in enumerate(self.points):
#             assert isinstance(p, torch.Tensor)
#             assert p.shape[0] == self.values.shape[i]

#     def __call__(self, points_to_interp):
#         assert self.points is not None
#         assert self.values is not None

#         assert len(points_to_interp) == len(self.points)
#         K = points_to_interp[0].shape[0]
#         for x in points_to_interp:
#             assert x.shape[0] == K

#         idxs = []
#         dists = []
#         overalls = []
#         for p, x in zip(self.points, points_to_interp):
#             idx_right = torch.bucketize(x, p)
#             idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
#             idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
#             dist_left = x - p[idx_left]
#             dist_right = p[idx_right] - x
#             dist_left[dist_left < 0] = 0.0
#             dist_right[dist_right < 0] = 0.0
#             both_zero = (dist_left == 0) & (dist_right == 0)
#             dist_left[both_zero] = dist_right[both_zero] = 1.0

#             idxs.append((idx_left, idx_right))
#             dists.append((dist_left, dist_right))
#             overalls.append(dist_left + dist_right)

#         numerator = 0.0
#         for indexer in product([0, 1], repeat=self.n):
#             as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
#             bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
#             numerator += self.values[as_s] * torch.prod(torch.stack(bs_s), dim=0)
#         denominator = torch.prod(torch.stack(overalls), dim=0)
#         return numerator / denominator


# def points_real_world_affine_registration(
#     points, fixed_affine, moving_affine, registration_affine
# ):
#     # Convert voxel coordinates to real-world coordinates in the fixed image space
#     grid_flat_homogeneous = torch.cat(
#         [points, torch.ones((1, points.shape[1], 1))], dim=2
#     )
#     fixed_world_coords = torch.bmm(fixed_affine, grid_flat_homogeneous.permute(0, 2, 1))

#     # Transform real-world coordinates to the moving image space using the registration affine
#     moving_world_coords = torch.bmm(
#         torch.linalg.inv(registration_affine), fixed_world_coords
#     )
#     # moving_world_coords = torch.bmm(registration_affine, fixed_world_coords)
#     moving_voxel_coords_homogeneous = torch.bmm(
#         torch.linalg.inv(moving_affine), moving_world_coords
#     )
#     moving_voxel_coords = (
#         moving_voxel_coords_homogeneous[:, :3, :].permute(0, 2, 1).squeeze(0)
#     )
#     return moving_voxel_coords


# def transform_gridsample(
#     fixed_data, moving_data, fixed_affine, moving_affine, registration_affine
# ):
#     grid_shape = fixed_data.shape[2:]
#     print("grid shape:", grid_shape)
#     print("moving shape:", moving_data.shape)

#     # Create a grid of voxel coordinates in the fixed image space
#     x = torch.arange(grid_shape[0])
#     y = torch.arange(grid_shape[1])
#     z = torch.arange(grid_shape[2])
#     grid = torch.meshgrid(x, y, z, indexing="ij")
#     grid = torch.stack(grid, dim=-1).float()

#     # Flatten the grid to N x 3 array of voxel coordinates and add batch dimension
#     grid_flat = grid.reshape(-1, 3).unsqueeze(0)  # Add batch dimension

#     moving_voxel_coords = points_real_world_affine_registration(
#         grid_flat, fixed_affine, moving_affine, registration_affine
#     )

#     transformed_grid = moving_voxel_coords.reshape(1, *grid_shape, 3)
#     transformed_grid = convert_flow_voxel2norm(transformed_grid, moving_data.shape[2:])

#     # Flip because grid_sample orders the dimensions differently (?)
#     transformed_grid = torch.flip(transformed_grid, dims=[-1])
#     print(transformed_grid)
#     # transformed_grid = transformed_grid[..., [2, 0, 1]]  # 012, 021, 102, 120, 201, 210

#     # Use grid_sample to interpolate the moving image at the transformed coordinates
#     aligned_data = F.grid_sample(
#         moving_data,
#         transformed_grid,
#         mode="bilinear",
#         padding_mode="border",
#         align_corners=False,
#     )
#     return aligned_data


# def transform_regulargridinterp(
#     fixed_data, moving_data, fixed_affine, moving_affine, registration_affine
# ):
#     """Same as transform_gridsample but with RegularGridInterpolator instead of F.grid_sample."""
#     # Get the shape of the fixed image
#     fixed_shape = fixed_data.shape[2:]  # Exclude batch dimension

#     # Create a grid of voxel coordinates in the fixed image space
#     x = torch.arange(fixed_shape[0])
#     y = torch.arange(fixed_shape[1])
#     z = torch.arange(fixed_shape[2])
#     grid = torch.meshgrid(x, y, z, indexing="ij")
#     grid = torch.stack(grid, dim=-1).float()

#     # Flatten the grid to N x 3 array of voxel coordinates and add batch dimension
#     grid_flat = grid.reshape(-1, 3).unsqueeze(0)  # Add batch dimension

#     moving_voxel_coords = points_real_world_affine_registration(
#         grid_flat, fixed_affine, moving_affine, registration_affine
#     )

#     moving_voxel_coords = [moving_voxel_coords[..., i] for i in range(3)]
#     print("transformed", moving_voxel_coords)

#     # Interpolate the values in the moving image at the transformed coordinates
#     # Assuming you have a RegularGridInterpolator function implemented in PyTorch
#     points = [
#         torch.arange(moving_data.shape[2]),
#         torch.arange(moving_data.shape[3]),
#         torch.arange(moving_data.shape[4]),
#     ]
#     values = moving_data.squeeze(0).squeeze(0)
#     aligned_data_flat = RegularGridInterpolator(points, values)(moving_voxel_coords)

#     aligned_data = (
#         aligned_data_flat.reshape(fixed_shape).unsqueeze(0).unsqueeze(0)
#     )  # Add batch dimension back
#     return aligned_data


# def transform_affinegrid_gridsample(
#     fixed_data, moving_data, fixed_affine, moving_affine, registration_affine
# ):

#     def _build_rescaling_matrix(input_ranges, output_ranges):
#         """
#         Constructs an augmented affine matrix for rescaling from arbitrary input ranges to arbitrary output ranges.

#         Parameters:
#         input_ranges (list of tuples): List of tuples [(a1, b1), (a2, b2), ..., (an, bn)] representing input ranges for each dimension.
#         output_ranges (list of tuples): List of tuples [(c1, d1), (c2, d2), ..., (cn, dn)] representing output ranges for each dimension.

#         Returns:
#         torch.Tensor: Augmented affine matrix (1, n+1, n+1) for the transformation.
#         """
#         assert len(input_ranges) == len(
#             output_ranges
#         ), "Input and output ranges must have the same length"
#         # assert len(input_ranges) == self.dim

#         n = len(input_ranges)
#         A = torch.zeros((n, n))
#         B = torch.zeros(n)

#         for i in range(n):
#             a, b = input_ranges[i]
#             c, d = output_ranges[i]

#             # Compute the scale factor
#             scale = (d - c) / (b - a)

#             # Compute the translation factor
#             translation = (c + d) / 2 - scale * (a + b) / 2

#             # Fill in the diagonal of A
#             A[i, i] = scale

#             # Fill in the translation vector
#             B[i] = translation

#         # Construct the augmented affine matrix
#         augmented_matrix = torch.eye(n + 1)
#         augmented_matrix[:n, :n] = A
#         augmented_matrix[:n, n] = B

#         return augmented_matrix[None]  # Add batch dimension

#     fixed_dims = fixed_data.shape[2:]
#     moving_dims = moving_data.shape[2:]
#     print(fixed_dims, moving_dims)
#     rescale_norm2voxel = _build_rescaling_matrix(
#         [(-1, 1) for _ in range(3)],
#         [(-0.5, D - 0.5) for D in fixed_dims],
#     ).to(fixed_data)
#     rescale_voxel2norm = _build_rescaling_matrix(
#         [(-0.5, D - 0.5) for D in moving_dims],
#         [(-1, 1) for _ in range(3)],
#     ).to(fixed_data)

#     perm_mat = torch.eye(4).to(fixed_data)
#     perm_mat = perm_mat[None, [0, 2, 1, 3], :]  # 012, 021, 102, 120, 201, 210

#     # Calculate the overall transformation matrix from moving to fixed image space
#     overall_affine = torch.bmm(rescale_voxel2norm, torch.inverse(moving_affine))
#     overall_affine = torch.bmm(overall_affine, torch.inverse(registration_affine))
#     overall_affine = torch.bmm(overall_affine, fixed_affine)
#     overall_affine = torch.bmm(overall_affine, rescale_norm2voxel)

#     overall_affine = torch.bmm(perm_mat, overall_affine)

#     # Convert the overall affine matrix to the format required by affine_grid
#     overall_affine_3x4 = overall_affine[
#         :, :3, :
#     ]  # Extract the 3x4 part of the 4x4 matrix

#     # Create a grid of coordinates in the fixed image space
#     grid = F.affine_grid(overall_affine_3x4, fixed_data.size(), align_corners=False)
#     print(grid.shape)

#     # Grid_sample orders the dimensions differently.
#     # We need to reorder the 3-d points in each grid location...
#     grid = grid.flip(dims=[-1])
#     # ...as well as the grid itself.
#     # grid = grid.permute(0, 3, 2, 1, 4)

#     # Use grid_sample to interpolate the moving image at the transformed coordinates
#     aligned_data = F.grid_sample(
#         moving_data, grid, mode="bilinear", padding_mode="border", align_corners=False
#     )
#     return aligned_data
