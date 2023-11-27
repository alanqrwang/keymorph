import unittest

import numpy as np
import torch
from scipy import ndimage
from torch.testing import assert_close
from keymorph.layers import CenterOfMass2d, CenterOfMass3d
from keymorph.keypoint_aligners import ClosedFormRigid


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


class TestRigidAligner(unittest.TestCase):
    """Test rigid aligner."""

    def test_rigid_0(self):
        """Test simple translation in 1 dimension of 3d points."""
        rigid_aligner = ClosedFormRigid(dim=2)

        input1 = torch.tensor(
            [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]]
        ).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor(
            [[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3], [0, 0, 0.4]]
        ).float()
        input2 = input2.view(1, 4, 3)
        true = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1]]).float()
        true = true.view(1, 3, 4)
        result = rigid_aligner.get_rigid_matrix(input1, input2)
        assert_close(result, true)

    def test_rigid_1(self):
        """Test simple translation in 3 dimensions of 3d points."""
        rigid_aligner = ClosedFormRigid(dim=2)

        input1 = torch.tensor(
            [[0.3, 0, 0], [0.5, -0.1, 0.1], [0.7, -0.2, 0.2], [0.9, -0.3, 0.3]]
        ).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor(
            [[0.1, -0.1, 0.1], [0.3, -0.2, 0.2], [0.5, -0.3, 0.3], [0.7, -0.4, 0.4]]
        ).float()
        input2 = input2.view(1, 4, 3)
        true = torch.tensor([[1, 0, 0, -0.2], [0, 1, 0, -0.1], [0, 0, 1, 0.1]]).float()
        true = true.view(1, 3, 4)
        result = rigid_aligner.get_rigid_matrix(input1, input2)
        assert_close(result, true)

    def test_rigid_2(self):
        """Test 2d rotation around z-axis for 3d points."""
        rigid_aligner = ClosedFormRigid(dim=2)

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [[np.cos(r), -np.sin(r), 0, 0], [np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0]]
        ).float()
        true = true.view(1, 3, 4)
        result = rigid_aligner.get_rigid_matrix(input1, input2)
        assert_close(result, true)

    def test_rigid_3(self):
        """Test 2d rotation around z-axis plus scaling for 3d points. Should ignore scaling."""
        rigid_aligner = ClosedFormRigid(dim=2)

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        # Scaling
        input2 *= 0.5
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [[np.cos(r), -np.sin(r), 0, 0], [np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0]]
        ).float()
        true = true.view(1, 3, 4)
        result = rigid_aligner.get_rigid_matrix(input1, input2)
        assert_close(result, true)

    def test_rigid_4(self):
        """Test 2d rotation around z-axis plus scaling for 3d points, with trivial weights. Should ignore scaling."""
        rigid_aligner = ClosedFormRigid(dim=2)

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input1 = input1.view(1, 4, 3)
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        # Scaling
        input2 *= 0.5
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [[np.cos(r), -np.sin(r), 0, 0], [np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0]]
        ).float()
        true = true.view(1, 3, 4)
        weights = torch.tensor([1, 1, 1, 1]).float().view(1, -1)
        result = rigid_aligner.get_rigid_matrix(input1, input2, w=weights)
        assert_close(result, true)

    def test_rigid_5(self):
        """Test 2d rotation around z-axis plus scaling for 3d points, with non-trivial weights. Should ignore scaling."""
        rigid_aligner = ClosedFormRigid(dim=2)

        input1 = torch.tensor([[1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0]]).float()
        input2 = torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]).float()
        # Scaling
        input2 *= 2
        input1 = input1.view(1, 4, 3)
        input2 = input2.view(1, 4, 3)

        r = -np.pi / 2
        true = torch.tensor(
            [[np.cos(r), -np.sin(r), 0, 0], [np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0]]
        ).float()
        true = true.view(1, 3, 4)

        weights = torch.tensor([0, 0, 0, 1]).float().view(1, -1)
        result = rigid_aligner.get_rigid_matrix(input1, input2, w=weights)
        assert_close(result, true)


if __name__ == "__main__":
    unittest.main()
