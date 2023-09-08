import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict
import wandb
import os
import argparse

def align_img(grid, x):
    return F.grid_sample(x,
                         grid=grid,
                         mode='bilinear',
                         padding_mode='border',
                         align_corners=False)

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

def str_or_float(x):
    try:
        return float(x)
    except ValueError:
        return x

def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}

def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(**config.wandb_kwargs)
    wandb.config.update(config)

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val