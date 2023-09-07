import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import loguniform
import torchio as tio

from keymorph import loss_ops
from keymorph.cm_plotter import show_warped, show_warped_vol
from keymorph.augmentation import augment_moving


def _get_tps_lmbda(num_samples, args, is_train=True):
    if not is_train and args.tps_lmbda in ['uniform', 'lognormal', 'loguniform']:
        choices = [0, 0.01, 0.1, 1.0, 10]
        lmbda = torch.tensor(np.random.choice(choices, size=num_samples)).to(args.device)

    if args.tps_lmbda is None:
        assert args.kp_align_method != 'tps', 'Need to implement this'
        lmbda = None
    elif args.tps_lmbda == 'uniform':
        lmbda = torch.rand(num_samples).to(args.device) * 10
    elif args.tps_lmbda == 'lognormal':
        lmbda = torch.tensor(np.random.lognormal(size=num_samples)).to(args.device)
    elif args.tps_lmbda == 'loguniform':
        a, b = 1e-6, 10
        lmbda = torch.tensor(loguniform.rvs(a, b, size=num_samples)).to(args.device)
    else:
        lmbda = torch.tensor(args.tps_lmbda).repeat(num_samples).to(args.device)
    return lmbda

def _align_img(grid, x):
    return F.grid_sample(x,
                         grid=grid,
                         mode='bilinear',
                         padding_mode='border',
                         align_corners=False)

def step(fixed, moving, 
         network, kp_aligner, 
         args, 
         optimizer=None, 
         aug_params=None, 
         is_train=True):
    '''Forward pass for one mini-batch step. 
    
    :param fixed, moving: Fixed and moving TorchIO Subjects
    :param network: Feature extractor network
    :param optimizer: Optimizer
    :param kp_aligner: Affine or TPS keypoint alignment module
    :param args: Other script parameters
    '''
    if is_train:
       assert network.training
       assert optimizer is not None
    
    # Get images, masks, and segmentations
    img_f, img_m = fixed['img'][tio.DATA], moving['img'][tio.DATA]
    if 'seg' in fixed and 'seg' in moving:
        seg_available = True
        seg_f, seg_m = fixed['seg'][tio.DATA], moving['seg'][tio.DATA]
    else:
        seg_available = False

    if hasattr(args, 'loss_fn') and args.loss_fn == 'dice':
        assert seg_available, 'Need segmentation for dice loss'

    # Move to device
    img_f = img_f.float().to(args.device)
    img_m = img_m.float().to(args.device)
    if seg_available:
        seg_f = seg_f.float().to(args.device)
        seg_m = seg_m.float().to(args.device)

    if is_train:
        # Explicitly augment moving image
        if seg_available:
            img_m, seg_m = augment_moving(img_m, args, seg=seg_m, fixed_params=aug_params)
        else:
            img_m = augment_moving(img_m, args, fixed_params=aug_params)

        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        # Extract keypoints
        points_f = network(img_f)
        points_m = network(img_m)
        points_f = points_f.view(-1, args.num_keypoints, args.dim)
        points_m = points_m.view(-1, args.num_keypoints, args.dim)

        if args.num_keypoints > 256: # Take mini-batch of keypoints
            key_batch_idx = np.random.choice(args.num_keypoints, size=256, replace=False)
            points_f = points_f[:, key_batch_idx]
            points_m = points_m[:, key_batch_idx]
        
        # Align via keypoints
        lmbda = _get_tps_lmbda(len(points_m), args, is_train=is_train)
        grid = kp_aligner.grid_from_points(points_m, points_f, img_f.shape, lmbda=lmbda)
        img_a = _align_img(grid, img_m)
        if seg_available:
            seg_a = _align_img(grid, seg_m)
        points_a = kp_aligner.points_from_points(points_m, points_f, points_m, lmbda=lmbda)

        # Compute metrics
        mse = loss_ops.MSELoss()(img_f, img_a)
        if seg_available:
            soft_dice = loss_ops.DiceLoss()(seg_a, seg_f)
            hard_dice = loss_ops.DiceLoss(hard=True)(seg_a, seg_f,
                                                    ign_first_ch=True)
            hausd = loss_ops.hausdorff_distance(seg_a, seg_f)
            grid = grid.permute(0, 4, 1, 2, 3)
            jdstd = loss_ops.jdstd(grid)
            jdlessthan0 = loss_ops.jdlessthan0(grid, as_percentage=True)

        # Compute loss
        if args.loss_fn == 'mse':
          loss = mse
        elif args.loss_fn == 'dice':
          loss = soft_dice

        # Perform backward pass
        if is_train:
            loss.backward()
            optimizer.step()

        if seg_available:
            metrics = {
              'loss': loss.cpu().detach().numpy(),
              'mse': mse.cpu().detach().numpy(),
              'softdice': 1-soft_dice.cpu().detach().numpy(),
              'harddice': 1-hard_dice[0].cpu().detach().numpy(),
              # 'harddiceroi': 1-hard_dice[1].cpu().detach().numpy(),
              'hausd': hausd,
              'jdstd': jdstd,
              'jdlessthan0': jdlessthan0,
            }
        else:
            metrics = {
              'loss': loss.cpu().detach().numpy(),
              'mse': mse.cpu().detach().numpy(),
            }

    if args.visualize:
        if args.dim == 2:
            show_warped(
                    img_m[0,0].cpu().detach().numpy(),
                    img_a[0,0].cpu().detach().numpy(),
                    img_f[0,0].cpu().detach().numpy(),
                    seg_m[0,0].cpu().detach().numpy(),
                    seg_a[0,0].cpu().detach().numpy(),
                    seg_f[0,0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(), 
                    points_f[0].cpu().detach().numpy(),
                    save_dir='./training_output/',
                    save_name='0.png')
        else:
            show_warped_vol(
                    img_m[0,0].cpu().detach().numpy(),
                    img_f[0,0].cpu().detach().numpy(),
                    img_a[0,0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(), 
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    save_dir='./training_output',
                    save_name='0.png')
            show_warped_vol(
                    seg_m.argmax(1)[0].cpu().detach().numpy(),
                    seg_f.argmax(1)[0].cpu().detach().numpy(),
                    seg_a.argmax(1)[0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(), 
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    save_dir='./training_output',
                    save_name='0.png')

    if seg_available:
        return metrics, \
              (img_f, img_m, img_a), \
              (seg_f, seg_m, seg_a), \
              (points_f, points_m, points_a), \
              grid
    else:
        return metrics, \
              (img_f, img_m, img_a), \
              (points_f, points_m, points_a), \
              grid