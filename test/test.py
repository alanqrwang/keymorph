import unittest

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.testing import assert_close
from keymorph.layers import CenterOfMass2d, CenterOfMass3d
from keymorph.keypoint_aligners import RigidKeypointAligner, AffineKeypointAligner
from keymorph import utils


class TestCenterOfMass2d(unittest.TestCase):
    """Test center of mass layer for 2D inputs.

    Outputs of CoM layers put x coordinate first, then y coordinate.
    Note this is flipped from the (row, column) order.
    """

    def test_com2d_0(self):
        com_layer = CenterOfMass2d()

        input = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float()
        input = input.view(1, 1, 3, 3)
        true = torch.tensor([0, 0]).float()
        true = true.view(1, 1, 2)
        result = com_layer(input)
        assert_close(result, true)

    def test_com2d_1(self):
        com_layer = CenterOfMass2d()

        input = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 1]]).float()
        input = input.view(1, 1, 3, 3)
        true = torch.tensor([0, 0]).float()
        true = true.view(1, 1, 2)
        result = com_layer(input)
        assert_close(result, true)

    def test_com2d_2(self):
        com_layer = CenterOfMass2d()

        # Generate a 2D array of zeros
        image = np.zeros((513, 513))
        # Set a single pixel to 1
        image[256, 256] = 1
        # Apply a Gaussian filter to the image
        sigma = 50
        filtered_image = ndimage.gaussian_filter(image, sigma)

        input = torch.tensor(filtered_image).float()
        input = input.view(1, 1, 513, 513)
        true = torch.tensor([0, 0]).float()
        true = true.view(1, 1, 2)
        result = com_layer(input)
        assert_close(result, true)

    def test_com2d_3(self):
        com_layer = CenterOfMass2d()

        # Generate a 2D array of zeros
        image = np.zeros((513, 257))
        # Set a single pixel to 1
        image[256, 128] = 1
        # Apply a Gaussian filter to the image
        sigma = 50
        filtered_image = ndimage.gaussian_filter(image, sigma)

        input = torch.tensor(filtered_image).float()
        input = input.view(1, 1, 513, 257)
        true = torch.tensor([0, 0]).float()
        true = true.view(1, 1, 2)
        result = com_layer(input)
        assert_close(result, true)

    def test_com2d_4(self):
        com_layer = CenterOfMass2d()

        # Generate a 2D array of zeros
        image = np.zeros((101, 101))
        # Set a single pixel to 1
        image[50, 25] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image = ndimage.gaussian_filter(image, sigma)

        input = torch.tensor(filtered_image).float()
        input = input.view(1, 1, 101, 101)
        true = torch.tensor([-0.5, 0]).float()
        true = true.view(1, 1, 2)
        result = com_layer(input)
        assert_close(result, true)

    def test_com2d_5(self):
        """Test batched inputs."""
        com_layer = CenterOfMass2d()

        # Generate a 2D array of zeros
        image1 = np.zeros((101, 101))
        # Set a single pixel to 1
        image1[50, 25] = 1
        image2 = np.zeros((101, 101))
        image2[25, 50] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image1 = ndimage.gaussian_filter(image1, sigma)
        filtered_image2 = ndimage.gaussian_filter(image2, sigma)
        image = np.stack([filtered_image1, filtered_image2], axis=0)[:, np.newaxis]

        input = torch.tensor(image).float()
        true = torch.tensor([[-0.5, 0], [0, -0.5]]).float()
        true = true.view(2, 1, 2)
        result = com_layer(input)
        assert_close(result, true)


class TestCenterOfMass3d(unittest.TestCase):
    """Test center of mass layer for 3D inputs."""

    def test_com3d_0(self):
        """Test input with point in center."""
        com_layer = CenterOfMass3d()

        input = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ).float()
        input = input.view(1, 1, 3, 3, 3)
        true = torch.tensor([0, 0, 0]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_1(self):
        """Test input with 3 spread out points."""
        com_layer = CenterOfMass3d()

        input = torch.tensor(
            [
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ]
        ).float()
        input = input.view(1, 1, 3, 3, 3)
        true = torch.tensor([0, 0, 0]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_3(self):
        """Center points smoothed with Gaussian filter."""
        com_layer = CenterOfMass3d()

        # Generate a 2D array of zeros
        image = np.zeros((101, 101, 101))
        # Set a single pixel to 1
        image[50, 50, 50] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image = ndimage.gaussian_filter(image, sigma)

        input = torch.tensor(filtered_image).float()
        input = input.view(1, 1, 101, 101, 101)
        true = torch.tensor([0, 0, 0]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_4(self):
        """Input with non-uniform dimensions."""
        com_layer = CenterOfMass3d()

        # Generate a 2D array of zeros
        image = np.zeros((101, 51, 51))
        # Set a single pixel to 1
        image[50, 25, 25] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image = ndimage.gaussian_filter(image, sigma)

        input = torch.tensor(filtered_image).float()
        input = input.view(1, 1, 101, 51, 51)
        true = torch.tensor([0, 0, 0]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_5(self):
        """Input with uniform dimensions but point is off-center."""
        com_layer = CenterOfMass3d()

        # Generate a 2D array of zeros
        image = np.zeros((101, 101, 101))
        # Set a single pixel to 1
        image[50, 25, 25] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image = ndimage.gaussian_filter(image, sigma)

        input = torch.tensor(filtered_image).float()
        input = input.view(1, 1, 101, 101, 101)
        true = torch.tensor([-0.5, -0.5, 0]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_6(self):
        """Batched inputs."""
        com_layer = CenterOfMass3d()

        # Generate a 2D array of zeros
        image1 = np.zeros((101, 101, 101))
        # Set a single pixel to 1
        image1[50, 25, 25] = 1
        image2 = np.zeros((101, 101, 101))
        image2[25, 50, 50] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image1 = ndimage.gaussian_filter(image1, sigma)
        filtered_image2 = ndimage.gaussian_filter(image2, sigma)
        image = np.stack([filtered_image1, filtered_image2], axis=0)[:, np.newaxis]

        input = torch.tensor(image).float()
        true = torch.tensor([[-0.5, -0.5, 0], [0, 0, -0.5]]).float()
        true = true.view(2, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_7(self):
        """Batched inputs, ij ordering."""
        com_layer = CenterOfMass3d(indexing="ij")

        # Generate a 2D array of zeros
        image1 = np.zeros((101, 101, 101))
        # Set a single pixel to 1
        image1[50, 25, 25] = 1
        image2 = np.zeros((101, 101, 101))
        image2[25, 50, 50] = 1
        # Apply a Gaussian filter to the image
        sigma = 5
        filtered_image1 = ndimage.gaussian_filter(image1, sigma)
        filtered_image2 = ndimage.gaussian_filter(image2, sigma)
        image = np.stack([filtered_image1, filtered_image2], axis=0)[:, np.newaxis]

        input = torch.tensor(image).float()
        true = torch.tensor([[0, -0.5, -0.5], [-0.5, 0, 0]]).float()
        true = true.view(2, 1, 3)
        result = com_layer(input)
        assert_close(result, true)


class TestRigidAligner(unittest.TestCase):
    """Test rigid aligner."""

    def test_rigid_0(self):
        """Test simple translation in 1 dimension of 3d points."""

        input1 = torch.tensor(
            [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]]
        ).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor(
            [[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3], [0, 0, 0.4]]
        ).float()
        input2 = input2.view(1, 4, 3)

        rigid_aligner = RigidKeypointAligner(input1, input2, dim=3)
        true = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]]
        ).float()
        true = true.view(1, 4, 4)
        result = rigid_aligner.transform_matrix
        assert_close(result, true)

    def test_rigid_0_forward_inverse(self):
        """Test simple translation in 1 dimension of 3d points, and whether inverse matrix given keypoint pair is equal to forward of flipped keypoint pairs, and vice versa."""

        input1 = torch.tensor(
            [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]]
        ).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor(
            [[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3], [0, 0, 0.4]]
        ).float()
        input2 = input2.view(1, 4, 3)

        rigid_aligner = RigidKeypointAligner(input1, input2, dim=3)
        result_12_forward = rigid_aligner.transform_matrix
        result_12_inverse = rigid_aligner.inverse_transform_matrix

        rigid_aligner = RigidKeypointAligner(input2, input1, dim=3)
        result_21_forward = rigid_aligner.transform_matrix
        result_21_inverse = rigid_aligner.inverse_transform_matrix
        assert_close(result_12_forward, result_21_inverse)
        assert_close(result_21_forward, result_12_inverse)

    # def test_rigid_1(self):
    #     """Test simple translation in 3 dimensions of 3d points.
    #     TODO: This test always fails. The reason is because determinant of resulting R matrix is not 1. See Arun et al.!
    #     """
    #     input1 = torch.tensor(
    #         [[0.3, 0, 0], [0.5, -0.1, 0.1], [0.7, -0.2, 0.2], [0.9, -0.3, 0.3]]
    #     ).float()
    #     input1 = input1.view(1, 4, 3)
    #     input2 = torch.tensor(
    #         [[0.1, -0.1, 0.1], [0.3, -0.2, 0.2], [0.5, -0.3, 0.3], [0.7, -0.4, 0.4]]
    #     ).float()
    #     input2 = input2.view(1, 4, 3)
    #     true = torch.tensor(
    #         [[1, 0, 0, -0.2], [0, 1, 0, -0.1], [0, 0, 1, 0.1], [0, 0, 0, 1]]
    #     ).float()
    #     true = true.view(1, 4, 4)

    #     rigid_aligner = RigidKeypointAligner(input1, input2, dim=3)
    #     result = rigid_aligner.transform_matrix
    #     assert_close(result, true)

    def test_rigid_1_flipped(self):
        """Test simple translation in 3 dimensions of 3d points."""
        input1 = torch.tensor(
            [[0.1, -0.1, 0.1], [0.3, -0.2, 0.2], [0.5, -0.3, 0.3], [0.7, -0.4, 0.4]]
        ).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor(
            [[0.3, 0, 0], [0.5, -0.1, 0.1], [0.7, -0.2, 0.2], [0.9, -0.3, 0.3]]
        ).float()
        input2 = input2.view(1, 4, 3)
        true = torch.tensor(
            [[1, 0, 0, 0.2], [0, 1, 0, 0.1], [0, 0, 1, -0.1], [0, 0, 0, 1]]
        ).float()
        true = true.view(1, 4, 4)

        rigid_aligner = RigidKeypointAligner(input1, input2, dim=3)
        result = rigid_aligner.transform_matrix
        assert_close(result, true)

    def test_rigid_2(self):
        """Test 2d rotation around z-axis for 3d points."""

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [
                [np.cos(r), -np.sin(r), 0, 0],
                [np.sin(r), np.cos(r), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).float()
        true = true.view(1, 4, 4)

        rigid_aligner = RigidKeypointAligner(input1, input2, dim=3)
        result = rigid_aligner.transform_matrix
        assert_close(result, true)

    def test_rigid_3(self):
        """Test 2d rotation around z-axis plus scaling for 3d points. Should ignore scaling."""

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        # Scaling
        input2 *= 0.5
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [
                [np.cos(r), -np.sin(r), 0, 0],
                [np.sin(r), np.cos(r), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).float()
        true = true.view(1, 4, 4)

        rigid_aligner = RigidKeypointAligner(input1, input2, dim=3)
        result = rigid_aligner.transform_matrix
        assert_close(result, true)

    def test_rigid_4(self):
        """Test 2d rotation around z-axis plus scaling for 3d points, with trivial weights. Should ignore scaling."""

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        # Scaling
        input2 *= 0.5
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [
                [np.cos(r), -np.sin(r), 0, 0],
                [np.sin(r), np.cos(r), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).float()
        true = true.view(1, 4, 4)
        weights = torch.tensor([1, 1, 1, 1]).float().view(1, -1)

        rigid_aligner = RigidKeypointAligner(input1, input2, w=weights, dim=3)
        result = rigid_aligner.transform_matrix
        assert_close(result, true)


class TestAffineAligner(unittest.TestCase):
    def test_affine_0(self):
        """Test 2d rotation around z-axis for 2d points."""

        input1 = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float()
        input1 = input1.view(1, 4, 2)
        input2 = torch.tensor([[0, -1], [-1, 0], [0, 1], [1, 0]]).float()
        input2 = input2.view(1, 4, 2)

        r = -np.pi / 2
        true = torch.tensor(
            [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
        ).float()
        true = true.view(1, 3, 3)

        affine_aligner = AffineKeypointAligner(input1, input2, dim=2)
        result = affine_aligner.transform_matrix
        assert_close(result, true)

    def test_affine_1(self):
        """Test 2d rotation around z-axis for 2d points plus scaling."""

        input1 = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float()
        input2 = torch.tensor([[0, -2], [-2, 0], [0, 2], [2, 0]]).float()
        input1 = input1.view(1, 4, 2)
        input2 = input2.view(1, 4, 2)

        r = -np.pi / 2
        rot_mat = torch.tensor(
            [[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]
        ).float()
        sca_mat = torch.tensor([[2, 0], [0, 2]]).float()
        # input2 is rotated, and then scaled by 2

        # Output should be [A, b] concatenated matrix
        true = torch.eye(3, 3)
        true[:2, :2] = rot_mat @ sca_mat
        true = true.view(1, 3, 3)

        affine_aligner = AffineKeypointAligner(input1, input2, dim=2)
        result = affine_aligner.transform_matrix
        assert_close(result, true)

    def test_affine_2(self):
        """Test 3d rotation around z-axis for 2d points plus scaling."""

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [
                [np.cos(r), -np.sin(r), 0, 0],
                [np.sin(r), np.cos(r), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).float()
        true = true.view(1, 4, 4)

        affine_aligner = AffineKeypointAligner(input1, input2, dim=3)
        result = affine_aligner.transform_matrix
        assert_close(result, true)

    def test_real_world_affine_grid_0(self):
        """
        utils.real_world_affine_grid with identity transformation and no affine matrices.
        """

        # No keypoint mismatch
        input1 = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float()
        input1 = input1.view(1, 4, 2)
        input2 = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float()
        input2 = input2.view(1, 4, 2)

        # No affine matrices
        aff_m = torch.eye(3)[None].to(input1)
        aff_f = torch.eye(3)[None].to(input2)
        shape_f = (1, 1, 5, 5)
        shape_m = (1, 1, 5, 5)
        affine_aligner = AffineKeypointAligner(
            input2,
            input1,
            dim=2,
            align_in_real_world_coords=True,
            aff_m=aff_m,
            aff_f=aff_f,
            shape_f=shape_f,
            shape_m=shape_m,
        )  # Not using this class, we only need the real_world_affine_grid() function.
        affine_matrix = torch.eye(3)[None].float()

        grid_shape = (1, 1, 5, 5)
        true = F.affine_grid(affine_matrix[:, :2, :], grid_shape)
        mygrid = affine_aligner.real_world_affine_grid(grid_shape)
        assert_close(true, mygrid)

    def test_real_world_affine_grid_1(self):
        """
        utils.real_world_affine_grid with simple translation and no affine matrices.
        """

        input1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[2, 1, 4], [5, 4, 7], [8, 7, 10], [11, 10, 13]]).float()
        input2 = input2.view(1, 4, 3)

        # No affine matrices
        aff_m = torch.eye(4)[None].to(input1)
        aff_f = torch.eye(4)[None].to(input2)
        shape_f = (1, 1, 5, 5, 5)
        shape_m = (1, 1, 5, 5, 5)
        affine_aligner = AffineKeypointAligner(
            input2,
            input1,
            dim=3,
            align_in_real_world_coords=True,
            aff_m=aff_m,
            aff_f=aff_f,
            shape_f=shape_f,
            shape_m=shape_m,
        )
        affine_matrix = torch.tensor(
            [[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]]
        )[None].float()

        grid_shape = (1, 1, 5, 5, 5)
        true = F.affine_grid(affine_matrix[:, :3, :], grid_shape)
        mygrid = affine_aligner.real_world_affine_grid(grid_shape)
        assert_close(true, mygrid)


class TestRealWorldCoordinates(unittest.TestCase):
    def test_convert_points_norm2voxel(self):
        """
        utils.convert_points_norm2voxel rescales points in normalized space to voxel coordinates
        """

        points = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float()
        points = points.view(1, 4, 2)

        # For a 5x5 grid, -1 maps to -0.5 and 1 maps to 4.5.
        # This is because the center of the pixel is the coordinate.
        grid_size = torch.tensor([[5, 5]]).float()

        points_true = (
            torch.tensor([[4.5, 2.0], [2.0, -0.5], [-0.5, 2.0], [2.0, 4.5]])
            .float()
            .view(1, 4, 2)
        )
        points_real = utils.convert_points_norm2voxel(points, grid_size)
        assert_close(points_real, points_true)

    def test_convert_points_voxel2norm(self):
        """
        utils.convert_points_norm2voxel rescales points in normalized space to voxel coordinates
        """

        points = (
            torch.tensor([[4.5, 2.0], [2.0, -0.5], [-0.5, 2.0], [2.0, 4.5]])
            .float()
            .view(1, 4, 2)
        )

        # For a 5x5 grid, -1 maps to -0.5 and 1 maps to 4.5.
        # This is because the center of the pixel is the coordinate.
        grid_size = torch.tensor([[5, 5]]).float()

        points_true = (
            torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float().view(1, 4, 2)
        )
        points_real = utils.convert_points_voxel2norm(points, grid_size)
        assert_close(points_real, points_true)

    def test_convert_points_voxel2real(self):
        """
        utils.convert_points_voxel2real applies affine transformation to get real world coordinates
        """

        points = torch.tensor(
            [[4.5, 2.0], [2.0, -0.5], [-0.5, 2.0], [2.0, 4.5]]
        ).float()
        points = points.view(1, 4, 2)
        r = -np.pi / 2  # 90 degree rotation about origin
        affine = torch.tensor(
            [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
        ).float()[None]

        points_true = (
            torch.tensor([[2, -4.5], [-0.5, -2], [2, 0.5], [4.5, -2]])
            .float()
            .view(1, 4, 2)
        )
        points_real = utils.convert_points_voxel2real(points, affine)
        assert_close(points_real, points_true)

    def test_convert_points_real2voxel(self):
        """
        utils.convert_points_real2voxel applies inverse affine transformation to get voxel coordinates
        """

        points = (
            torch.tensor([[2, -4.5], [-0.5, -2], [2, 0.5], [4.5, -2]])
            .float()
            .view(1, 4, 2)
        )
        r = -np.pi / 2  # 90 degree rotation about origin
        affine = torch.tensor(
            [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
        ).float()[None]

        points_true = (
            torch.tensor([[4.5, 2.0], [2.0, -0.5], [-0.5, 2.0], [2.0, 4.5]])
            .float()
            .view(1, 4, 2)
        )
        points_real = utils.convert_points_real2voxel(points, affine)
        assert_close(points_real, points_true)

    def test_convert_points_norm2real(self):
        """
        utils.convert_points_norm2real does 2 things:
            1. Rescale points in normalized space to voxel coordinates
            2. Apply affine transformation to get real world coordinates
        """

        points = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float()
        points = points.view(1, 4, 2)
        r = -np.pi / 2  # 90 degree rotation
        affine = torch.tensor(
            [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
        ).float()[None]

        # align_corners in torch.nn.functional.grid_sample is False.
        # Therefore, 5 here refers to the corners of the grid.
        # Therefore, 1 in [-1, 1] space maps to [0, 5] space.
        # E.g., [0, 1] -> [2.5, 5]
        voxel_sizes = torch.tensor([[5, 5]]).float()

        points_true = (
            torch.tensor([[2, -4.5], [-0.5, -2], [2, 0.5], [4.5, -2]])
            .float()
            .view(1, 4, 2)
        )
        points_real = utils.convert_points_norm2real(points, affine, voxel_sizes)
        assert_close(points_real, points_true)

    def test_convert_points_real2norm(self):
        """
        utils.convert_points_norm2real does 2 things:
            1. Rescale points in normalized space to voxel coordinates
            2. Apply affine transformation to get real world coordinates
        """

        points = (
            torch.tensor([[2, -4.5], [-0.5, -2], [2, 0.5], [4.5, -2]])
            .float()
            .view(1, 4, 2)
        )
        r = -np.pi / 2  # 90 degree rotation
        affine = torch.tensor(
            [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
        ).float()[None]

        # align_corners in torch.nn.functional.grid_sample is False.
        # Therefore, 5 here refers to the corners of the grid.
        # Therefore, 1 in [-1, 1] space maps to [0, 5] space.
        # E.g., [0, 1] -> [2.5, 5]
        voxel_sizes = torch.tensor([[5, 5]]).float()

        points_true = (
            torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1]]).float().view(1, 4, 2)
        )
        points_real = utils.convert_points_real2norm(points, affine, voxel_sizes)
        assert_close(points_real, points_true)

    def test_convert_flow_voxel2norm(self):
        """
        utils.convert_points_norm2real does 2 things:
            1. Rescale points in normalized space to voxel coordinates
            2. Apply affine transformation to get real world coordinates
        """

        flow_real = torch.tensor(
            [
                [[-0.5, 4.5], [2, 2], [2, -0.5], [4.5, -0.5]],
                [[-0.5, 4.5], [2, 2], [2, -0.5], [4.5, -0.5]],
            ]
        ).float()
        # align_corners in torch.nn.functional.grid_sample is False.
        # Therefore, 1 in [-1, 1] space maps to [-0.5, 4.5] in voxel space.
        # E.g., [0, 1] -> [2, 4.5]
        voxel_sizes = (5, 5)

        flow_true = torch.tensor(
            [
                [[-1, 1], [0, 0], [0, -1], [1, -1]],
                [[-1, 1], [0, 0], [0, -1], [1, -1]],
            ]
        ).float()
        flow_norm = utils.convert_flow_voxel2norm(flow_real, voxel_sizes)
        assert_close(flow_norm, flow_true)


if __name__ == "__main__":
    unittest.main()
