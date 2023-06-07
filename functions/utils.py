import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict
from scipy.stats import loguniform

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

def get_lmbda(num_samples, args, is_train=True):
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