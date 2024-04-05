import itk
import numpy as np
import torch
import time
import os

from keymorph.utils import utils


class ITKElastix:
    def __init__(self):
        super().__init__()

    def eval(self):
        pass

    def __call__(self, fixed, moving, transform_type="rigid", **kwargs):
        return self.pairwise_register(fixed, moving, transform_type, **kwargs)

    def pairwise_register(self, fixed, moving, transform_type="rigid", **kwargs):
        original_device = fixed.device
        save_dir = kwargs["save_dir"]
        num_resolutions = kwargs["num_resolutions_for_itkelastix"]
        assert len(fixed) == 1, "Fixed image should be a single image"
        assert len(moving) == 1, "Moving image should be a single image"

        fixed = fixed.cpu().detach().numpy().astype(np.float32)[0, 0]
        moving = moving.cpu().detach().numpy().astype(np.float32)[0, 0]
        fixed_image = itk.image_view_from_array(fixed)
        moving_image = itk.image_view_from_array(moving)

        # Dictionary of results
        result_dict = {}

        for ttype in transform_type:
            print(ttype)
            start_time = time.time()
            parameter_object = itk.ParameterObject.New()
            if ttype == "rigid":
                default_parameter_map = parameter_object.GetDefaultParameterMap(
                    "rigid", num_resolutions
                )
                parameter_object.AddParameterMap(default_parameter_map)
            elif ttype == "affine":
                default_parameter_map = parameter_object.GetDefaultParameterMap(
                    "affine", num_resolutions
                )
                default_parameter_map["FinalBSplineInterpolationOrder"] = ["0"]
                parameter_object.AddParameterMap(default_parameter_map)
            elif ttype == "bspline":
                default_affine_parameter_map = parameter_object.GetDefaultParameterMap(
                    "affine", num_resolutions
                )
                default_affine_parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_object.AddParameterMap(default_affine_parameter_map)
                default_bspline_parameter_map = parameter_object.GetDefaultParameterMap(
                    "bspline", num_resolutions
                )
                default_bspline_parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_object.AddParameterMap(default_bspline_parameter_map)

            # Call registration function
            print("SAVE DIR", save_dir)
            _, result_transform_parameters = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_console=False,
                # output_directory=save_dir,
            )

            # Get deformation field
            displacement_field = itk.transformix_deformation_field(
                moving_image, result_transform_parameters
            )
            displacement_field = torch.tensor(
                itk.array_view_from_image(displacement_field)
            )[None]

            # Convert displacement to grid
            grid = utils.displacement2flow(displacement_field)

            register_time = time.time() - start_time
            res = {
                "align_type": ttype,
                "grid": grid.to(original_device),
                "time": register_time,
            }
            result_dict[ttype] = res

        return result_dict

    def groupwise_register(self, inputs, transform_type="rigid", **kwargs):
        """Groupwise register.

        inputs can be:
         - directories of images, looks for files img_*.npy
         - list of image paths
         - Torch Tensor stack of images (N, 1, D, H, W)"""
        log_to_console = kwargs["log_to_console"]
        num_resolutions = kwargs["num_resolutions_for_itkelastix"]

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
        if isinstance(inputs[0], str):
            group_imgs_m = [np.load(p)[0, 0] for p in inputs]
            for img in group_imgs_m:
                print(img.shape)
            group_imgs_m = np.stack(group_imgs_m)
        elif isinstance(inputs, torch.Tensor):
            group_imgs_m = inputs[:, 0].cpu().detach().numpy()

        # Ensure the stacked array is contiguous in memory
        group_imgs_m = np.ascontiguousarray(group_imgs_m).astype(np.float32)
        itk_group_imgs = itk.image_view_from_array(group_imgs_m)

        # List of results
        result_dict = {}

        for ttype in transform_type:
            start_time = time.time()
            # Create Groupwise Parameter Object
            parameter_object = itk.ParameterObject.New()
            groupwise_parameter_map = parameter_object.GetDefaultParameterMap(
                "groupwise", num_resolutions
            )
            if ttype == "rigid":
                groupwise_parameter_map["FinalBSplineInterpolationOrder"] = ["0"]
                groupwise_parameter_map["Transform"] = ["EulerStackTransform"]
            groupwise_parameter_map["AutomaticScalesEstimation"] = ["true"]
            groupwise_parameter_map["AutomaticScalesEstimationStackTransform"] = [
                "true"
            ]
            parameter_object.AddParameterMap(groupwise_parameter_map)

            # Call registration function
            # both fixed and moving image should be set with the vector_itk to prevent elastix from throwing errors
            result_images, result_transform_parameters = (
                itk.elastix_registration_method(
                    itk_group_imgs,
                    itk_group_imgs,
                    parameter_object=parameter_object,
                    log_to_console=log_to_console,
                    output_directory=save_dir,
                )
            )

            # Get deformation fields
            # ITK returns a 4th dimension and idk what it is, so just remove it
            displacement_field = itk.transformix_deformation_field(
                itk_group_imgs, result_transform_parameters
            )
            displacement_field = torch.tensor(
                itk.array_view_from_image(displacement_field)
            )[..., :3].permute(0, 4, 1, 2, 3)

            if kwargs["plot"]:
                result_images = torch.tensor(
                    itk.array_view_from_image(result_images)
                )  # .permute(3, 0, 1, 2)[None]
                # plot result_images
                import matplotlib.pyplot as plt

                for i in range(result_images.shape[0]):
                    plt.imshow(result_images[i, 128, :, :])
                    plt.show()

            # Convert displacement to grid
            group_size, D, H, W = group_imgs_m.shape

            # Step 1: Create the original grid for 3D
            coords_x, coords_y, coords_z = torch.meshgrid(
                torch.linspace(-1, 1, W),
                torch.linspace(-1, 1, H),
                torch.linspace(-1, 1, D),
                indexing="ij",
            )
            coords = torch.stack(
                [coords_z, coords_y, coords_x], dim=0
            )  # Shape: (3, D, H, W)
            coords = coords.unsqueeze(0).repeat(
                group_size, 1, 1, 1, 1
            )  # Shape: (N, 3, D, H, W), N=group_size

            # Step 2: Normalize the displacement field
            # Convert physical displacement values to the [-1, 1] range
            # Assuming the displacement field is given in voxel units (physical coordinates)
            for i, dim_size in enumerate(
                [W, H, D]
            ):  # Note the order matches x, y, z as per the displacement_field
                # Normalize such that the displacement of 1 full dimension length corresponds to a move from -1 to 1
                displacement_field[:, i, ...] = (
                    2 * displacement_field[:, i, ...] / (dim_size - 1)
                )

            # Step 3: Add the displacement field to the original grid to get the transformed coordinates
            grid = coords + displacement_field
            grid = grid.permute(0, 2, 3, 4, 1)

            register_time = time.time() - start_time

            res = {
                "time": register_time,
            }
            save_results_to_disk = kwargs["save_results_to_disk"]
            if save_results_to_disk and save_dir:
                for i in range(len(grid)):
                    np.save(
                        f"{save_dir}/{ttype}_grid_{i:03}.npy",
                        grid[i : i + 1].cpu().detach().numpy(),
                    )
            else:
                res["groupgrids"] = grid
            result_dict[ttype] = res

        return result_dict
