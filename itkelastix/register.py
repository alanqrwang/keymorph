import itk
import numpy as np
import torch
import time


class ITKElastix:
    def __init__(self):
        super().__init__()

    def eval(self):
        pass

    def __call__(self, fixed, moving, transform_type="rigid", **kwargs):
        return self.pairwise_register(fixed, moving, transform_type, **kwargs)

    def pairwise_register(self, fixed, moving, transform_type="rigid", **kwargs):
        original_device = fixed.device
        assert len(fixed) == 1, "Fixed image should be a single image"
        assert len(moving) == 1, "Moving image should be a single image"

        fixed = fixed.cpu().detach().numpy().astype(np.float32)[0, 0]
        moving = moving.cpu().detach().numpy().astype(np.float32)[0, 0]
        fixed_image = itk.image_view_from_array(fixed)
        moving_image = itk.image_view_from_array(moving)

        # List of results
        result_list = []

        num_resolutions = 1
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
            _, result_transform_parameters = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_console=False,
            )

            # Get deformation field
            displacement_field = itk.transformix_deformation_field(
                moving_image, result_transform_parameters
            )
            displacement_field = torch.tensor(
                itk.array_view_from_image(displacement_field)
            ).permute(3, 0, 1, 2)[None]

            # Convert displacement to grid
            D, H, W = moving.shape

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
            coords = coords.unsqueeze(0)  # Shape: (N, 3, D, H, W), N=1

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

            register_time = time.time() - start_time
            res = {
                "align_type": ttype,
                "grid": grid.permute(0, 2, 3, 4, 1).to(original_device),
                "time": register_time,
            }
            result_list.append(res)

        return result_list

    def groupwise_register(self, group_imgs_m, transform_type="rigid", **kwargs):
        # Ensure the stacked array is contiguous in memory
        group_imgs_m = group_imgs_m.cpu().detach().numpy()[:, 0]
        group_imgs_m = np.ascontiguousarray(group_imgs_m).astype(np.float32)
        print(group_imgs_m.shape)
        itk_group_imgs = itk.image_view_from_array(group_imgs_m)

        # List of results
        result_list = []

        num_resolutions = 1
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
                    log_to_console=True,
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

            # result_images = torch.tensor(
            #     itk.array_view_from_image(result_images)
            # )  # .permute(3, 0, 1, 2)[None]
            # print(result_images.shape)
            # # plot result_images
            # import matplotlib.pyplot as plt

            # for i in range(result_images.shape[0]):
            #     plt.imshow(result_images[i, 128, :, :])
            #     plt.show()

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

            register_time = time.time() - start_time

            res = {
                "align_type": ttype,
                "grids": grid.permute(0, 2, 3, 4, 1),
                "time": register_time,
            }
            result_list.append(res)

        return result_list
