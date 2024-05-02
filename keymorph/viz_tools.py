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


def rot90_points(points, k):
    assert k in [0, 1, 2, 3]
    rotated_points = np.zeros_like(points)

    # Calculate the center of the image
    center_y, center_x = (np.array((256, 256)) - 1) / 2.0

    # Translate points to origin for rotation
    translated_points = points - np.array([center_x, center_y])

    rotated_points = np.zeros_like(translated_points)
    if k == 1:
        rotated_points[:, 0] = translated_points[:, 1]
        rotated_points[:, 1] = -translated_points[:, 0]
    elif k == 3:
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


def imshow_img_and_points_3d(
    img=None,
    all_points=None,
    all_weights=None,
    projection=False,
    center_slices=(128, 128, 128),
    slab_thickness=10,
    axes=None,
    rotate_90_deg=0,
    img_dims=(256, 256, 256),
):
    """
    Plots 3 orthogonal slices of a 3D image and overlays points on them.

    img: (H, W, D) image
    all_points: (N, 3) or (B, N, 3) array of points.
        Points can be defined on:
         1. the Pytorch grid, i.e. in [-1, 1]
         2. the image grid, e.g. in [0, 256]
        The code converts Pytorch grids to image grids by internally checking if
    all_weights: (N,) or (B, N) array of weights
    """

    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    plot_img = True if img is not None else None
    plot_points = True if all_points is not None else None

    if plot_points:
        if len(all_points.shape) == 2:
            all_points = all_points[None]

        # If points are in [-1, 1], convert to image coordinates
        if np.all(all_points >= -1) or np.all(all_points <= 1):
            print("Converting points from [-1, 1] to image coordinates")
            all_points = (all_points + 1) / 2 * img_dims[-1]
        if np.any(all_points < 0):
            raise ValueError("Points must be non-negative")

        if all_weights is None:
            all_weights = np.ones(all_points.shape[:2])
        else:
            if len(all_weights.shape) == 1:
                all_weights = all_weights[None]
            all_weights = normalize(all_weights)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    xy_ind, xz_ind, yz_ind = center_slices
    if plot_img:
        # Set global vmin and vmax so intensities values are comparable relative to each other
        vmin = min(img.min(), img.min(), img.min())
        vmax = max(img.max(), img.max(), img.max())

        [ax.set_xticks([]) for ax in axes]
        [ax.set_yticks([]) for ax in axes]

        img_xy = img[:, :, xy_ind]
        img_xz = img[:, xz_ind, :]
        img_yz = img[yz_ind, :, :]

        if rotate_90_deg != 0:
            img_xy = np.rot90(img_xy, k=rotate_90_deg)
            img_xz = np.rot90(img_xz, k=rotate_90_deg)
            img_yz = np.rot90(img_yz, k=rotate_90_deg)

        axes[0].imshow(img_xy, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].imshow(img_xz, cmap="gray", vmin=vmin, vmax=vmax)
        axes[2].imshow(img_yz, cmap="gray", vmin=vmin, vmax=vmax)

    if plot_points:
        markers = [".", "+", "x"]
        for points, weights, marker in zip(all_points, all_weights, markers):
            points_xy, depth_xy = points[:, 1:], points[:, 0]
            points_xz, depth_xz = (
                np.stack((points[:, 0], points[:, 2]), axis=-1),
                points[:, 1],
            )
            points_yz, depth_yz = points[:, :2], points[:, 2]
            if projection:
                colors_xy = [[1, 0, 0, 1] for _ in range(len(points))]
                weights_xy = weights

                colors_xz = [[1, 0, 0, 1] for _ in range(len(points))]
                weights_xy = weights

                colors_yz = [[1, 0, 0, 1] for _ in range(len(points))]
                weights_xy = weights

            else:
                cmap = matplotlib.colormaps["bwr"]

                start, stop = xy_ind - slab_thickness / 2, xy_ind + slab_thickness / 2
                indices = (depth_xy > start).nonzero() and (depth_xy < stop).nonzero()
                points_xy = points_xy[indices]
                depth_xy = depth_xy[indices]
                weights_xy = weights[indices]

                start, stop = xz_ind - slab_thickness / 2, xz_ind + slab_thickness / 2
                indices = (depth_xz > start).nonzero() and (depth_xz < stop).nonzero()
                points_xz = points_xz[indices]
                depth_xz = depth_xz[indices]
                weights_xz = weights[indices]

                start, stop = yz_ind - slab_thickness / 2, yz_ind + slab_thickness / 2
                indices = (depth_yz > start).nonzero() and (depth_yz < stop).nonzero()
                points_yz = points_yz[indices]
                depth_yz = depth_yz[indices]
                weights_yz = weights[indices]

            # Set alpha to be proportional to weights
            colors_xy = cmap(normalize(depth_xy))
            colors_xy[:, -1] = weights_xy
            colors_xz = cmap(normalize(depth_xz))
            colors_xz[:, -1] = weights_xz
            colors_yz = cmap(normalize(depth_yz))
            colors_yz[:, -1] = weights_yz

            if rotate_90_deg != 0:
                points_xy = rot90_points(points_xy, rotate_90_deg)
                points_xz = rot90_points(points_xz, rotate_90_deg)
                points_yz = rot90_points(points_yz, rotate_90_deg)

            axes[0].scatter(
                points_xy[:, 0],
                points_xy[:, 1],
                marker=marker,
                s=100,
                color=colors_xy,
            )
            axes[1].scatter(
                points_xz[:, 0],
                points_xz[:, 1],
                marker=marker,
                s=100,
                color=colors_xz,
            )
            axes[2].scatter(
                points_yz[:, 0],
                points_yz[:, 1],
                marker=marker,
                s=100,
                color=colors_yz,
            )
    [ax.set_xlim([0, img_dims[-1]]) for ax in axes]
    [ax.set_ylim([img_dims[-1], 0]) for ax in axes]

    return axes


def imshow_registration_3d(
    img_m,
    img_f,
    img_a,
    points_m=None,
    points_f=None,
    points_a=None,
    weights=None,
    suptitle=None,
    save_path=None,
):
    """
    Moved, aligned, and fixed should be volumes with 3 dimensions.
    Points should be normalized in [-1,1].

    Convention is that (-1, -1, -1) is on the top left, front-most
    corner. x-values get bigger to the right, y-values get bigger
    going down, and z-values get bigger going back.
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
    points_af = np.stack([points_a, points_f], axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))

    imshow_img_and_points_3d(
        img_m,
        points_m,
        weights,
        axes=(axes[0, 0], axes[1, 0], axes[2, 0]),
        img_dims=(256, 256, 256),
    )
    imshow_img_and_points_3d(
        img_f,
        points_f,
        weights,
        axes=(axes[0, 1], axes[1, 1], axes[2, 1]),
        img_dims=(256, 256, 256),
    )
    imshow_img_and_points_3d(
        img_a,
        points_af,
        weights,
        axes=(axes[0, 2], axes[1, 2], axes[2, 2]),
        img_dims=(256, 256, 256),
    )

    axes[0, 0].set_title("Moving")
    axes[0, 1].set_title("Fixed")
    axes[0, 2].set_title("Warped")
    if save_path is not None:
        fig.savefig(
            save_path,
            format="png",
            dpi=100,
            bbox_inches="tight",
        )
    if suptitle:
        fig.suptitle(suptitle)
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
