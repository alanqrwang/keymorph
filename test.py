import unittest

import numpy as np
import torch
from scipy import ndimage
from torch.testing import assert_close
from keymorph.layers import CenterOfMass2d, CenterOfMass3d


class TestCenterOfMass2d(unittest.TestCase):
    """Test center of mass layer for 2D inputs."""

    def test_com2d_0(self):
        com_layer = CenterOfMass2d()

        input = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float()
        input = input.view(1, 1, 3, 3)
        true = torch.tensor([0.5, 0.5]).float()
        true = true.view(1, 1, 2)
        result = com_layer(input)
        assert_close(result, true)

    def test_com2d_1(self):
        com_layer = CenterOfMass2d()

        input = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 1]]).float()
        input = input.view(1, 1, 3, 3)
        true = torch.tensor([0.5, 0.5]).float()
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
        true = torch.tensor([0.5, 0.5]).float()
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
        true = torch.tensor([0.5, 0.5]).float()
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
        true = torch.tensor([0.5, 0.25]).float()
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
        true = torch.tensor([[0.5, 0.25], [0.25, 0.5]]).float()
        true = true.view(2, 1, 2)
        result = com_layer(input)
        assert_close(result, true)


class TestCenterOfMass3d(unittest.TestCase):
    def test_com3d_0(self):
        """
        Test center of mass layer for 2D inputs
        """
        com_layer = CenterOfMass3d()

        input = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ).float()
        input = input.view(1, 1, 3, 3, 3)
        true = torch.tensor([0.5, 0.5, 0.5]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_1(self):
        """
        Test center of mass layer for 2D inputs
        """
        com_layer = CenterOfMass3d()

        input = torch.tensor(
            [
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ]
        ).float()
        input = input.view(1, 1, 3, 3, 3)
        true = torch.tensor([0.5, 0.5, 0.5]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_3(self):
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
        true = torch.tensor([0.5, 0.5, 0.5]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_4(self):
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
        true = torch.tensor([0.5, 0.5, 0.5]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_5(self):
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
        true = torch.tensor([0.5, 0.25, 0.25]).float()
        true = true.view(1, 1, 3)
        result = com_layer(input)
        assert_close(result, true)

    def test_com3d_6(self):
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
        true = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.5, 0.5]]).float()
        true = true.view(2, 1, 3)
        result = com_layer(input)
        assert_close(result, true)


if __name__ == "__main__":
    unittest.main()
