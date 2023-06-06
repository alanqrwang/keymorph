import torch
import torch.nn.functional as F
import math
import numpy as np
from functions import augmentation
from collections import defaultdict

def parse_test_metric(mod, aug):
    mod1, mod2 = mod.split('_')
    if 'rot' in aug:
        if aug == 'rot0':
            rot_aug = [0, 0, 0]
        elif aug == 'rot45':
            rot_aug = np.random.choice([0, math.pi/4], size=3)
        elif aug == 'rot90':
            rot_aug = np.random.choice([0, math.pi/2], size=3)
        elif aug == 'rot135':
            rot_aug = np.random.choice([0, 3*math.pi/4], size=3)
        elif aug == 'rot180':
            rot_aug = np.random.choice([0, math.pi], size=3)
        aug_param = [(0,0,0), (0,0,0), rot_aug, (0,0,0,0,0,0)]
    else:
        raise NotImplementedError()
    
    return mod1, mod2, aug_param

def augment_moving(x, seg, args, max_random_params=(0.2, 0.2, 3.1416, 0.1), fixed_params=None):
    '''Augment moving image.
    
    max_random_params: 4-tuple of floats, max value of each transformation for random augmentation
    fixed_params: Fixed parameters for transformation. Fixed augmentation if set.
    '''
    if fixed_params:
        s, o, a, z = fixed_params
        if args.dim == 2:
            scale = torch.tensor([e+1 for e in s]).unsqueeze(0).float()
            offset = torch.tensor(o).unsqueeze(0).float()
            theta = torch.tensor(a).unsqueeze(0).float()
            shear = torch.tensor(z).unsqueeze(0).float()
        else:
            scale = torch.tensor([e+1 for e in s]).unsqueeze(0).float()
            offset = torch.tensor(o).unsqueeze(0).float()
            theta = torch.tensor(a).unsqueeze(0).float()
            shear = torch.tensor(z).unsqueeze(0).float()
    else:
        s, o, a, z = max_random_params
        if args.dim == 2:
            scale = torch.FloatTensor(1, 2).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 2).uniform_(-o, o)
            theta = torch.FloatTensor(1, 1).uniform_(-a, a)
            shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        else:
            scale = torch.FloatTensor(1, 3).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 3).uniform_(-o, o)
            theta = torch.FloatTensor(1, 3).uniform_(-a, a)
            shear = torch.FloatTensor(1, 6).uniform_(-z, z)

    params = (scale, offset, theta, shear)

    if args.dim == 2:
        augmenter = augmentation.AffineDeformation2d(device=args.device)
    else:
        augmenter = augmentation.AffineDeformation3d(device=args.device)
    x = augmenter(x, params=params, interp_mode='bilinear')
    seg = augmenter(seg, params=params, interp_mode='nearest')
    return x, seg

def augment_pair(x1, x2, args, params=(0.2, 0.2, 3.1416, 0.1), random=True):
    s, o, a, z = params
    if random:
        if args.dim == 2:
            scale = torch.FloatTensor(1, 2).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 2).uniform_(-o, o)
            theta = torch.FloatTensor(1, 1).uniform_(-a, a)
            shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        else:
            scale = torch.FloatTensor(1, 3).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 3).uniform_(-o, o)
            theta = torch.FloatTensor(1, 3).uniform_(-a, a)
            shear = torch.FloatTensor(1, 6).uniform_(-z, z)
    else:
        if args.dim == 2:
            scale = torch.FloatTensor(1, 2).fill_(1-s, 1+s)
            offset = torch.FloatTensor(1, 2).fill_(-o, o)
            theta = torch.FloatTensor(1, 1).fill_(-a, a)
            shear = torch.FloatTensor(1, 2).fill_(-z, z)
        else:
            scale = torch.FloatTensor(1, 3).fill_(1+s)
            offset = torch.FloatTensor(1, 3).fill_(o)
            theta = torch.FloatTensor(1, 3).fill_(a)
            shear = torch.FloatTensor(1, 6).fill_(z)

    params = (scale, offset, theta, shear)

    if args.dim == 2:
        augmenter = augmentation.AffineDeformation2d(device=args.device)
    else:
        augmenter = augmentation.AffineDeformation3d(device=args.device)
    x1 = augmenter(x1, params=params, interp_mode='bilinear')
    x2 = augmenter(x2, params=params, interp_mode='bilinear')
    return x1, x2

def augment_moving_points(x_fixed, points, args, params=(0.2, 0.2, 3.1416, 0.1)):
    s, o, a, z = params
    s = np.clip(s*args.epoch / args.affine_slope, None, s)
    o = np.clip(o*args.epoch / args.affine_slope, None, o)
    a = np.clip(a*args.epoch / args.affine_slope, None, a)
    z = np.clip(z*args.epoch / args.affine_slope, None, z)
    if args.dim == 2:
        scale = torch.FloatTensor(1, 2).uniform_(1-s, 1+s)
        offset = torch.FloatTensor(1, 2).uniform_(-o, o)
        theta = torch.FloatTensor(1, 1).uniform_(-a, a)
        shear = torch.FloatTensor(1, 2).uniform_(-z, z)
    else:
        scale = torch.FloatTensor(1, 3).uniform_(1-s, 1+s)
        offset = torch.FloatTensor(1, 3).uniform_(-o, o)
        theta = torch.FloatTensor(1, 3).uniform_(-a, a)
        shear = torch.FloatTensor(1, 6).uniform_(-z, z)

    params = (scale, offset, theta, shear)

    if args.dim == 2:
        augmenter = augmentation.AffineDeformation2d(device=args.device)
    else:
        augmenter = augmentation.AffineDeformation3d(device=args.device)
    x_moving = augmenter.deform_img(x_fixed, params)
    points = augmenter.deform_points(points, params)
    return x_moving, points

def align_moving_img(grid, x_moving, seg_moving):
    x_aligned = F.grid_sample(x_moving,
                              grid=grid,
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=False)
    seg_aligned = F.grid_sample(seg_moving,
                                grid=grid, 
                                mode='bilinear',
                                padding_mode='border',
                                align_corners=False)
    return x_aligned, seg_aligned


def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}