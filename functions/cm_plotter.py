import torch
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

# global settings for plotting
matplotlib.rcParams['lines.linewidth'] = 2
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def blur_cm_plot(Cm_plot, sigma):
    """
    Blur the keypoints/center-of-masses for better visualiztion
    
    Arguments
    ---------
    Cm_plot : tensor with the center-of-masses
    sigma   : how much to blur

    Return
    ------
        out : blurred points
    """

    n_batch = Cm_plot.shape[0]
    n_reg = Cm_plot.shape[1]
    out = []
    for n in range(n_batch):
        cm_plot = Cm_plot[n, :, :, :]
        blur_cm_plot = []
        for r in range(n_reg):
            _blur_cm_plot = gaussian(cm_plot[r, :, :, :],
                                     sigma=sigma,
                                     mode='nearest')
            _blur_cm_plot = torch.from_numpy(_blur_cm_plot).float().unsqueeze(0)
            blur_cm_plot += [_blur_cm_plot]

        blur_cm_plot = torch.cat(blur_cm_plot, 0)
        out += [blur_cm_plot.unsqueeze(0)]
    return torch.cat(out, 0)


def get_cm_plot(Y_cm, dim0, dim1, dim2):
    """
    Convert the coordinate of the keypoint/center-of-mass to points in an tensor
    
    Arguments
    ---------
    Y_cm : keypoints coordinates/center-of-masses[n_bath, 3, n_reg]
    dim  : dim of the image

    Return
    ------
        out : tensor it assigns value of 1 where keypoints are located otherwise 0
    """

    n_batch = Y_cm.shape[0]

    out = []
    for n in range(n_batch):
        Y = Y_cm[n, :, :]
        n_reg = Y.shape[1]

        axis2 = torch.linspace(-1, 1, dim2).float()
        axis1 = torch.linspace(-1, 1, dim1).float()
        axis0 = torch.linspace(-1, 1, dim0).float()

        index0 = []
        for i in range(n_reg):
            index0.append(torch.argmin((axis0 - Y[2, i]) ** 2).item())

        index1 = []
        for i in range(n_reg):
            index1.append(torch.argmin((axis1 - Y[1, i]) ** 2).item())

        index2 = []
        for i in range(n_reg):
            index2.append(torch.argmin((axis2 - Y[0, i]) ** 2).item())

        cm_plot = torch.zeros(n_reg, dim0, dim1, dim2)
        for i in range(n_reg):
            cm_plot[i, index0[i], index1[i], index2[i]] = 1

        out += [cm_plot.unsqueeze(0)]

    return torch.cat(out, 0)

def show_warped(moved, aligned, fixed, 
                seg_moved, seg_aligned, seg_fixed,
                ctl_points, tgt_points, 
                save_dir=None, save_name=None, 
                extra_start=None, extra_end=None):

  '''
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
  '''
  tgt_points = (tgt_points+1) / 2
  ctl_points = (ctl_points+1) / 2
  disp_points = tgt_points - ctl_points
  fig, axes = plt.subplots(2, 3, figsize=(16,16))
  [ax.set_xticks([0, moved.shape[-1]]) for ax in axes.ravel()]
  [ax.set_xticklabels([-1, 1]) for ax in axes.ravel()]
  [ax.set_yticks([0, moved.shape[-1]]) for ax in axes.ravel()]
  [ax.set_yticklabels([-1, 1]) for ax in axes.ravel()]

  colors = cm.Reds(np.linspace(0, 1, len(ctl_points)))

  axes[0,0].imshow(moved, origin='upper', cmap='gray')
  axes[0,0].set_title('Moved')
  for i, c in zip(range(len(ctl_points)), colors):
    axes[0,0].scatter(ctl_points[i, 0]*moved.shape[1], ctl_points[i, 1]*moved.shape[0], marker='+', s=100, color=c)

  axes[0,1].imshow(aligned, origin='upper', cmap='gray')
  axes[0,1].set_title('Warped')
  for i, c in zip(range(len(ctl_points)), colors):
    axes[0,1].scatter(tgt_points[i, 0]*aligned.shape[1], tgt_points[i, 1]*aligned.shape[0], marker='+', s=100, color=c)

  axes[0,2].imshow(fixed, origin='upper', cmap='gray')
  axes[0,2].set_title('Fixed')
  for i, c in zip(range(len(ctl_points)), colors):
    axes[0,2].arrow(ctl_points[i, 0]*fixed.shape[1], ctl_points[i, 1]*fixed.shape[0], 
                  disp_points[i, 0]*fixed.shape[1], disp_points[i, 1]*fixed.shape[0], color=c, width=1.0)

  axes[1,0].imshow(seg_moved, origin='upper')
  axes[1,0].set_title('Moved')

  axes[1,1].imshow(seg_aligned, origin='upper')
  axes[1,1].set_title('Warped')

  axes[1,2].imshow(seg_fixed, origin='upper')
  axes[1,2].set_title('Fixed')

  if extra_start is not None:
    print(extra_start.shape, extra_end.shape)
    extra_start = (extra_start+1) / 2
    extra_end = (extra_end+1) / 2
    disp_points = extra_end - extra_start
    colors = cm.Blues(np.linspace(0, 1, len(extra_start)))

    for i, c in zip(range(len(extra_start)), colors):
      axes[0,0].scatter(extra_start[i, 0]*aligned.shape[1], extra_start[i, 1]*aligned.shape[0], marker='+', s=100, color=c)
      axes[0,1].scatter(extra_end[i, 0]*aligned.shape[1], extra_end[i, 1]*aligned.shape[0], marker='+', s=100, color=c)

      axes[0,2].arrow(extra_start[i, 0]*fixed.shape[1], extra_start[i, 1]*fixed.shape[0], 
                  disp_points[i, 0]*fixed.shape[1], disp_points[i, 1]*fixed.shape[0], color=c, width=2.0)

  if save_name is not None:
    fig.savefig(os.path.join(save_dir, save_name),
            format='png', dpi=100, bbox_inches='tight')
  fig.show()
  plt.show()
  plt.close()

def show_pretrain(moved, fixed, init_points, pred_points, tgt_points,
                  save_dir=None, save_name=None):
  '''
  Moved and fixed should be images with 2 dimensions.
  Points should be normalized in [0,1].
  '''
  fig, axes = plt.subplots(1, 3, figsize=(16,8))
  axes[0].axis('off')
  axes[1].axis('off')
  axes[2].axis('off')

  colors = cm.rainbow(np.linspace(0, 1, len(pred_points)))
  axes[0].imshow(fixed, origin='upper', cmap='gray')
  axes[0].set_title('Initial')
  for i, c in zip(range(len(init_points)), colors):
    axes[0].scatter(init_points[i, 0]*fixed.shape[1], init_points[i, 1]*fixed.shape[0], marker='+', s=100, color=c)

  colors = cm.rainbow(np.linspace(0, 1, len(pred_points)))
  axes[1].imshow(moved, origin='upper', cmap='gray')
  axes[1].set_title('Prediction')
  for i, c in zip(range(len(pred_points)), colors):
    axes[1].scatter(pred_points[i, 0]*fixed.shape[1], pred_points[i, 1]*fixed.shape[0], marker='+', s=100, color=c)

  axes[2].imshow(moved, origin='upper', cmap='gray')
  axes[2].set_title('Target')
  for i, c in zip(range(len(tgt_points)), colors):
    axes[2].scatter(tgt_points[i, 0]*moved.shape[1], tgt_points[i, 1]*moved.shape[0], marker='+', s=100, color=c)

  if save_name is not None:
    fig.savefig(os.path.join(save_dir, save_name),
            format='png', dpi=100, bbox_inches='tight')
  fig.show()
  plt.show()

def show_warped_vol(moving, fixed, warped, 
                seg_moving, seg_aligned, seg_fixed,
                ctl_points, tgt_points, warped_points,
                suptitle=None,
                save_dir=None, save_name=None):

  '''
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
  '''

  img_dim = moving.shape[-1]
  tgt_points = (tgt_points+1) / 2
  ctl_points = (ctl_points+1) / 2
  warped_points = (warped_points+1) / 2

  fig, axes = plt.subplots(3, 3, figsize=(16,16))
  [ax.set_xticks([0, moving.shape[-1]]) for ax in axes.ravel()]
  [ax.set_xticklabels([-1, 1]) for ax in axes.ravel()]
  [ax.set_yticks([0, moving.shape[-1]]) for ax in axes.ravel()]
  [ax.set_yticklabels([-1, 1]) for ax in axes.ravel()]
  [ax.set_xlim([-0*img_dim, 1*img_dim]) for ax in axes.ravel()]
  [ax.set_ylim([-0*img_dim, 1*img_dim]) for ax in axes.ravel()]

  colors = cm.viridis(np.linspace(0, 1, len(ctl_points)))

  xy_ind = moving.shape[-1] // 2
  xz_ind = moving.shape[-2] // 2
  yz_ind = moving.shape[-3] // 2

  all_xy = (moving[:, :, xy_ind], fixed[:, :, xy_ind], warped[:, :, xy_ind])
  all_xz = (moving[:, xz_ind, :], fixed[:, xz_ind, :], warped[:, xz_ind, :])
  all_yz = (moving[yz_ind, :, :], fixed[yz_ind, :, :], warped[yz_ind, :, :])

  # all_seg_moving = (seg_moving[:, :, xy_ind], seg_moving[:, xz_ind, :], seg_moving[yz_ind, :, :])
  # all_seg_aligned = (seg_aligned[:, :, xy_ind], seg_aligned[:, xz_ind, :], seg_aligned[yz_ind, :, :])
  # all_seg_fixed = (seg_fixed[:, :, xy_ind], seg_fixed[:, xz_ind, :], seg_fixed[yz_ind, :, :])

  # Loop over different dimensions, i.e. axial, sagittal, coronal.
  for index, ((m, f, w), (i, j)) in enumerate(zip([all_xy, all_xz, all_yz], [(0, 1), (0, 2), (1, 2)])):
    if index == 0:
      p_index = 2
      # m = np.rot90(m, k=2)
      # f = np.rot90(f, k=2)
      # w = np.rot90(w, k=2)
    elif index == 1:
      p_index = 1
    elif index == 2:
      p_index = 0
    axes[index,0].imshow(m, origin='upper', cmap='gray')
    for k, c in zip(range(len(ctl_points)), colors):
      axes[p_index,0].scatter(ctl_points[k, i]*img_dim, ctl_points[k, j]*img_dim, marker='+', s=100, color=c)

    axes[index,1].imshow(f, origin='upper', cmap='gray')
    for k, c in zip(range(len(ctl_points)), colors):
      axes[p_index,1].scatter(tgt_points[k, i]*img_dim, tgt_points[k, j]*img_dim, marker='+', s=100, color=c)

    axes[index,2].imshow(w, origin='upper', cmap='gray')
    for k, c in zip(range(len(ctl_points)), colors):
      axes[p_index,2].scatter(warped_points[k, i]*img_dim, warped_points[k, j]*img_dim, marker='.', s=100, color=c)
      axes[p_index,2].scatter(tgt_points[k, i]*img_dim, tgt_points[k, j]*img_dim, marker='+', s=100, color=c)

  # axes[1,0].imshow(seg_moved, origin='lower')
  # axes[1,0].set_title('Moved')

  # axes[1,1].imshow(seg_aligned, origin='lower')
  # axes[1,1].set_title('Warped')

  # axes[1,2].imshow(seg_fixed, origin='lower')
  # axes[1,2].set_title('Fixed')

  axes[0,0].set_title('Moving')
  axes[0,1].set_title('Fixed')
  axes[0,2].set_title('Warped')
  axes[0,0].set_ylabel('y')
  axes[0,0].set_xlabel('x')

  axes[1,0].set_xlabel('x')
  axes[1,0].set_ylabel('z')

  axes[2,0].set_xlabel('y')
  axes[2,0].set_ylabel('z')
  if save_name is not None:
    fig.savefig(os.path.join(save_dir, save_name),
            format='png', dpi=100, bbox_inches='tight')
  if suptitle:
    fig.suptitle(suptitle)
  fig.show()
  plt.show()
  plt.close()