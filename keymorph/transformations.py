import torch
import torch.nn as nn

from keymorph.utils import uniform_norm_grid


class AffineTransform(nn.Module):
    """Affine transformations."""

    def __init__(
        self,
        matrix=None,
        inverse_matrix=None,
        dim=3,
    ):
        """
        Args:
            One of matrix or inverse_matrix must be provided
            dim: Dimensionality of the points (2 or 3)
        """
        super().__init__()
        self.dim = dim
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

    def affine_grid(self, grid_shape):
        """Create a grid of affine transformed coordinates with specified shape to be used in F.grid_sample().

        Args:
            grid_shape (bs, 1, H, W) or (bs, 1, H, W, D): Shape of the grid to create
            m_shape: (bs, 1, H, W) or (bs, 1, H, W, D): Resolution of moving image being interpolating

        Returns:
            transformed_grid (bs, 1, H, W, D, dim): Affine grid in [-1, 1] space
        """
        grid = uniform_norm_grid(grid_shape, dim=self.dim)

        # Flatten the grid to (N x dim) array of voxel coordinates
        grid_flat = (
            grid.reshape(-1, self.dim).unsqueeze(0).to(self.transform_matrix)
        )  # Add batch dimension

        moving_voxel_coords = self.get_inverse_transformed_points(grid_flat)

        transformed_grid = moving_voxel_coords.reshape(1, *grid_shape[2:], self.dim)

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
