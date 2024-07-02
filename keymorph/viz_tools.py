import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib


# global settings for plotting
matplotlib.rcParams["lines.linewidth"] = 2
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 22
plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def imshow_registration_2d(
    img_m, img_f, img_a, points_m, points_f, points_a, save_dir=None, save_name=None
):
    """
    Moved, aligned, and fixed should be images with 2 dimensions.
    Points should be normalized in [-1,1].

    Convention is that (-1, -1) is on the top left, (1, -1) is top right,
    (-1, 1) is bottom left, and (1, 1) is bottom right.
    Essentially, just like (x, y) coordinates in math, except y values
    get bigger as you go down.
    Thus, when using plt.imshow(), I must set origin='upper', so that
    the origin is at the upper left-hand corner.

    What this implies is that in generating the grid that goes into
    F.grid_sample, the grid must follow this same convention; i.e.
    the first coordinate corresponds to the x-coordinate and the second
    coordinate correspond to the y-coordinate. This means that when
    the grid is "linearized" or "flattened", the first coordinate must
    vary first, i.e. (-1, -1), (0, -1), (1, -1), (-1, 0) ....
    This way, when you do normal C-style reshaping where the last dimension
    gets filled in first, the rows get filled with varying first coordinates,
    as we desire.
    """
    points_m = (points_m + 1) / 2
    points_f = (points_f + 1) / 2
    points_a = (points_a + 1) / 2
    fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 1 * 4))
    [ax.set_xticks([0, img_m.shape[-1]]) for ax in axes.ravel()]
    [ax.set_xticklabels([-1, 1]) for ax in axes.ravel()]
    [ax.set_yticks([0, img_m.shape[-1]]) for ax in axes.ravel()]
    [ax.set_yticklabels([-1, 1]) for ax in axes.ravel()]

    colors = cm.Reds(np.linspace(0, 1, len(points_m)))

    axes[0].imshow(img_m, origin="upper", cmap="gray")
    axes[0].set_title("Moving")
    for i, c in zip(range(len(points_m)), colors):
        axes[0].scatter(
            points_m[i, 0] * img_m.shape[1],
            points_m[i, 1] * img_m.shape[0],
            marker="+",
            s=100,
            color=c,
        )

    axes[1].imshow(img_f, origin="upper", cmap="gray")
    axes[1].set_title("Fixed")
    for i, c in zip(range(len(points_f)), colors):
        axes[1].scatter(
            points_f[i, 0] * img_f.shape[1],
            points_f[i, 1] * img_f.shape[0],
            marker="+",
            s=100,
            color=c,
        )

    axes[2].imshow(img_a, origin="upper", cmap="gray")
    axes[2].set_title("Aligned")
    for i, c in zip(range(len(points_a)), colors):
        axes[2].scatter(
            points_f[i, 0] * img_f.shape[1],
            points_f[i, 1] * img_f.shape[0],
            marker=".",
            color=c,
            s=100,
        )
        axes[2].scatter(
            points_a[i, 0] * img_a.shape[1],
            points_a[i, 1] * img_a.shape[0],
            marker="+",
            color=c,
            s=100,
        )

    if save_name is not None:
        fig.savefig(
            os.path.join(save_dir, save_name),
            format="png",
            dpi=100,
            bbox_inches="tight",
        )
    fig.show()
    plt.show()
    plt.close()


def rot90_points(points, k, img_dims=(256, 256)):
    assert k in [0, 1, 2, 3]
    rotated_points = np.zeros_like(points)

    # Calculate the center of the image
    center_y, center_x = (np.array(img_dims) - 1) / 2.0

    # Translate points to origin for rotation
    translated_points = points - np.array([center_x, center_y])

    rotated_points = np.zeros_like(translated_points)
    if k == 3:
        rotated_points[:, 0] = translated_points[:, 1]
        rotated_points[:, 1] = -translated_points[:, 0]
    elif k == 1:
        rotated_points[:, 0] = -translated_points[:, 1]
        rotated_points[:, 1] = translated_points[:, 0]
    elif k == 2:
        rotated_points[:, 0] = -translated_points[:, 0]
        rotated_points[:, 1] = -translated_points[:, 1]
    else:
        rotated_points = translated_points

    # Translate points back
    rotated_points += np.array([center_x, center_y])
    return rotated_points


def convert_points_norm2voxel(points, grid_sizes):
    """
    Rescale points from [-1, 1] to a uniform voxel grid with different sizes along each dimension.

    Args:
        points (bs, num_points, dim): Array of points in the normalized space [-1, 1].
        grid_sizes (bs, dim): Array of grid sizes for each dimension.

    Returns:
        Array of points in voxel space.
    """
    grid_sizes = np.array(grid_sizes)
    assert grid_sizes.shape[-1] == points.shape[-1], "Dimensions don't match"
    translated_points = points + 1
    scaled_points = (translated_points * grid_sizes) / 2
    rescaled_points = scaled_points - 0.5
    return rescaled_points


def imshow_img_and_points_3d(
    img=None,
    all_points=None,
    all_weights=None,
    projection=False,
    slab_thickness=10,
    axes=None,
    rotate_90_deg=0,
    markers=(".", "x"),
    point_space="norm",
    suptitle=None,
    keypoint_indexing="ij",
):
    """
    Plots 3 orthogonal views of a 3D image and overlays points on them.

    Points are assumed to lie in voxel ordering (also called ij indexing). So:
        - If img view is img[ind, :, :], then overlaid 2d points must be in dimension axes (1, 2).
        - If img view is img[:, ind, :], then overlaid 2d points must be in dimension axes (0, 2).
        - If img view is img[:, :, ind], then overlaid 2d points must be in dimension axes (0, 1).

    Args:
        img: (H, W, D) image
        all_points: (N, 3) or (1, N, 3) or (2, N, 3) array of points.
        point_space: 'norm' or 'voxel'. Points can be defined on:
            1. the normalized Pytorch grid, i.e. in [-1, 1]
            2. the voxel (image) grid, e.g. in [0, 256]
        all_weights: (N,) or (1, N) or (2, N) array of weights
        projection: If True, plot all points keypoints in each view, regardless of depth.
            Color will be different for every point, but the same across views.
            If False, plot only points in a slab around the center of the image.
            Color will correspond to depth of the point within the slab.
        slab_thickness: Thickness of the slab in which to plot points. Only used if projection=False.
    """
    img_dims = img.shape
    assert len(img_dims) == 3, "Image must be 3D"
    ind_12, ind_02, ind_01 = img_dims[0] // 2, img_dims[1] // 2, img_dims[2] // 2

    def _normalize(data):
        # Don't normalize if there's only one datapoint to avoid division by 0
        # Or, if the data is constant
        if data.shape[0] != 1 and np.max(data) != np.min(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            return data

    plot_img = True if img is not None else None
    plot_points = True if all_points is not None else None

    if plot_points:
        if len(all_points.shape) == 2:
            all_points = all_points[None]
        if isinstance(markers, str):
            markers = (markers,)

        if keypoint_indexing == "xy":
            print("Flipping points into ij indexing...")
            all_points = np.flip(all_points, axis=-1)

        if point_space == "norm":
            print("Converting points from [-1, 1] to image coordinates...")
            all_points = convert_points_norm2voxel(all_points, img_dims)

        if all_weights is None:
            all_weights = np.ones(all_points.shape[:2])
        else:
            if len(all_weights.shape) == 1:
                all_weights = all_weights[None]
            all_weights = _normalize(all_weights)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if plot_img:
        img_12 = img[ind_12, :, :]
        img_02 = img[:, ind_02, :]
        img_01 = img[:, :, ind_01]

        if rotate_90_deg != 0:
            img_12 = np.rot90(img_12, k=rotate_90_deg)
            img_02 = np.rot90(img_02, k=rotate_90_deg)
            img_01 = np.rot90(img_01, k=rotate_90_deg)

        axes[0].imshow(img_12, cmap="gray")
        axes[1].imshow(img_02, cmap="gray")
        axes[2].imshow(img_01, cmap="gray")

    if plot_points:
        cmap = matplotlib.colormaps["bwr"]
        for points, weights, marker in zip(all_points, all_weights, markers):
            points_12, depth_12 = points[:, [1, 2]], points[:, 0]
            points_02, depth_02 = points[:, [0, 2]], points[:, 1]
            points_01, depth_01 = points[:, [0, 1]], points[:, 2]

            if projection:
                colors_12 = cmap(np.linspace(0, 1, len(points)))
                colors_02 = cmap(np.linspace(0, 1, len(points)))
                colors_01 = cmap(np.linspace(0, 1, len(points)))
                plot_12, plot_02, plot_01 = True, True, True

            else:
                start, stop = ind_12 - slab_thickness / 2, ind_12 + slab_thickness / 2
                indices = (depth_12 > start).nonzero() and (depth_12 < stop).nonzero()
                if len(indices[0]) == 0:
                    plot_12 = False
                else:
                    plot_12 = True
                    points_12 = points_12[indices]
                    depth_12 = depth_12[indices]
                    weights_12 = weights[indices]

                start, stop = ind_02 - slab_thickness / 2, ind_02 + slab_thickness / 2
                indices = (depth_02 > start).nonzero() and (depth_02 < stop).nonzero()
                if len(indices[0]) == 0:
                    plot_02 = False
                else:
                    plot_02 = True
                    points_02 = points_02[indices]
                    depth_02 = depth_02[indices]
                    weights_02 = weights[indices]

                start, stop = ind_01 - slab_thickness / 2, ind_01 + slab_thickness / 2
                indices = (depth_01 > start).nonzero() and (depth_01 < stop).nonzero()
                if len(indices[0]) == 0:
                    plot_01 = False
                else:
                    plot_01 = True
                    points_01 = points_01[indices]
                    depth_01 = depth_01[indices]
                    weights_01 = weights[indices]

                # Set alpha to be proportional to weights
                if plot_12:
                    colors_12 = cmap(_normalize(depth_12))
                    colors_12[:, -1] = _normalize(weights_12)
                if plot_02:
                    colors_02 = cmap(_normalize(depth_02))
                    colors_02[:, -1] = _normalize(weights_02)
                if plot_01:
                    colors_01 = cmap(_normalize(depth_01))
                    colors_01[:, -1] = _normalize(weights_01)

            if rotate_90_deg != 0:
                if plot_12:
                    points_12 = rot90_points(
                        points_12, rotate_90_deg, img_dims=img_12.shape
                    )
                if plot_02:
                    points_02 = rot90_points(
                        points_02, rotate_90_deg, img_dims=img_02.shape
                    )
                if plot_01:
                    points_01 = rot90_points(
                        points_01, rotate_90_deg, img_dims=img_01.shape
                    )

            # Note: plt.scatter plots points in (x, y) order, which flips voxel ordering
            if plot_12:
                axes[0].scatter(
                    points_12[:, 1],
                    points_12[:, 0],
                    marker=marker,
                    s=100,
                    color=colors_12,
                )
            if plot_02:
                axes[1].scatter(
                    points_02[:, 1],
                    points_02[:, 0],
                    marker=marker,
                    s=100,
                    color=colors_02,
                )
            if plot_01:
                axes[2].scatter(
                    points_01[:, 1],
                    points_01[:, 0],
                    marker=marker,
                    s=100,
                    color=colors_01,
                )
    if suptitle:
        fig.suptitle(suptitle)

    return axes


def imshow_registration_3d(
    img_m,
    img_f,
    img_a,
    points_m=None,
    points_f=None,
    points_a=None,
    weights=None,
    resize=None,
    projection=False,
    slab_thickness=10,
    rotate_90_deg=0,
    suptitle=None,
    save_path=None,
):
    """
    Moved, aligned, and fixed should be volumes with 3 dimensions.
    Points should be normalized in [-1,1].

    Plots 3 orthogonal views of a 3D image and overlays points on them.

    Points are assumed to lie in voxel ordering (also called ij indexing). So:
        - If img view is img[ind, :, :], then overlaid 2d points must be in dimension axes (1, 2).
        - If img view is img[:, ind, :], then overlaid 2d points must be in dimension axes (0, 2).
        - If img view is img[:, :, ind], then overlaid 2d points must be in dimension axes (0, 1).

    Args:
        img: (H, W, D) image
        all_points: (N, 3) or (1, N, 3) or (2, N, 3) array of points.
        all_weights: (N,) or (1, N) or (2, N) array of weights
        resize: tuple, resizes the images to be the same size for easier spatial comparison. Use if your images have different resolutions.
        projection: If True, plot all keypoints in each view, regardless of depth.
            Color will be different for every point, but the same across views.
            If False, plot only points in a slab around the center of the image.
            Color will correspond to depth of the point within the slab.
        slab_thickness: Thickness of the slab in which to plot points. Only used if projection=False.
    """

    if resize is not None:
        import torchio as tio

        subject_m = tio.Subject(img=tio.ScalarImage(tensor=img_m[None]))
        subject_f = tio.Subject(img=tio.ScalarImage(tensor=img_f[None]))
        subject_a = tio.Subject(img=tio.ScalarImage(tensor=img_a[None]))
        transform = tio.Resize(resize)
        subject_m = transform(subject_m)
        subject_f = transform(subject_f)
        subject_a = transform(subject_a)
        img_m = subject_m.img.data[0]
        img_f = subject_f.img.data[0]
        img_a = subject_a.img.data[0]

    if len(points_m.shape) == 2:
        points_m = points_m[None]
    if len(points_f.shape) == 2:
        points_f = points_f[None]
    if len(points_a.shape) == 2:
        points_a = points_a[None]
    points_af = np.concatenate([points_a, points_f], axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))

    imshow_img_and_points_3d(
        img_m,
        points_m,
        weights,
        markers=".",
        axes=(axes[0, 0], axes[1, 0], axes[2, 0]),
        projection=projection,
        slab_thickness=slab_thickness,
        rotate_90_deg=rotate_90_deg,
        point_space="norm",
    )
    imshow_img_and_points_3d(
        img_f,
        points_f,
        weights,
        markers="x",
        axes=(axes[0, 1], axes[1, 1], axes[2, 1]),
        projection=projection,
        slab_thickness=slab_thickness,
        rotate_90_deg=rotate_90_deg,
        point_space="norm",
    )
    imshow_img_and_points_3d(
        img_a,
        points_af,
        weights,
        markers=(".", "x"),
        axes=(axes[0, 2], axes[1, 2], axes[2, 2]),
        projection=projection,
        slab_thickness=slab_thickness,
        rotate_90_deg=rotate_90_deg,
        point_space="norm",
    )

    axes[0, 0].set_title("Moving")
    axes[0, 1].set_title("Fixed")
    axes[0, 2].set_title("Warped")
    if suptitle:
        fig.suptitle(suptitle)
    if save_path is not None:
        fig.savefig(
            save_path,
            format="png",
            dpi=100,
            bbox_inches="tight",
        )
    fig.show()
    plt.show()
    plt.close()


def plot_groupwise_register(list_of_moving_imgs, list_of_aligned_imgs):
    fig, axes = plt.subplots(
        2, len(list_of_moving_imgs), figsize=(len(list_of_moving_imgs) * 4, 2 * 4)
    )
    for i, img in enumerate(list_of_moving_imgs):
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
    for i, img in enumerate(list_of_aligned_imgs):
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].axis("off")
    fig.show()
    plt.show()
