import re
import torch
import math
from collections import defaultdict
import os
import argparse
import json

try:
    import wandb
except ImportError as e:
    pass


def parse_test_mod(mod):
    if isinstance(mod, str):
        mod1, mod2 = mod.split("_")
    else:
        mod1, mod2 = mod
    return mod1, mod2


def parse_test_aug(aug):
    if "rot" in aug:
        if aug == "rot0":
            rot_aug = 0
        elif aug == "rot45":
            rot_aug = math.pi / 4
        elif aug == "rot90":
            rot_aug = math.pi / 2
        elif aug == "rot135":
            rot_aug = 3 * math.pi / 4
        elif aug == "rot180":
            rot_aug = math.pi
        aug_param = (0, 0, rot_aug, 0)
    else:
        raise NotImplementedError()

    return aug_param


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


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    state = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = state["state_dict"]

    # Sometimes the model is saved with "backbone" prefix
    new_state_dict = {
        key.replace(".backbone", ""): value for key, value in state_dict.items()
    }
    missing_keys, _ = model.backbone.load_state_dict(new_state_dict, strict=True)
    print("Missing keys when loading checkpoint: ", missing_keys)

    res = (state, model)

    if optimizer:
        optimizer.load_state_dict(state["optimizer"])
        res += (optimizer,)
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
        res += (scheduler,)

    return res


# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split("=")
            if value_str.replace("-", "").isnumeric():
                processed_val = int(value_str)
            elif value_str.replace("-", "").replace(".", "").isnumeric():
                processed_val = float(value_str)
            elif value_str in ["True", "true"]:
                processed_val = True
            elif value_str in ["False", "false"]:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val


def summary(network):
    """Print model summary."""
    print("")
    print("Model Summary")
    print("---------------------------------------------------------------")
    for name, _ in network.named_parameters():
        print(name)
    print(
        "Total parameters:",
        sum(p.numel() for p in network.parameters() if p.requires_grad),
    )
    print("---------------------------------------------------------------")
    print("")


def save_dict_as_json(dict, save_path):
    with open(save_path, "w") as outfile:
        json.dump(dict, outfile, sort_keys=True, indent=4)


def load_dict_from_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def get_latest_epoch_file(directory_path, args):
    max_epoch = -1
    latest_epoch_file = None

    # Compile a regular expression pattern to extract the epoch number
    if args.run_mode == "pretrain":
        epoch_pattern = re.compile(r"pretrained_epoch(\d+)_model\.pth\.tar")
    else:
        epoch_pattern = re.compile(r"epoch(\d+)_trained_model.pth.tar")

    # List all files in the given directory
    for filename in os.listdir(directory_path):
        match = epoch_pattern.match(filename)
        if match:
            # Extract the epoch number and convert it to an integer
            epoch_num = int(match.group(1))
            # Update the max_epoch and latest_epoch_file if this file has a larger epoch number
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_epoch_file = filename

    # Return the path of the file with the largest epoch number
    if latest_epoch_file is not None:
        return os.path.join(directory_path, latest_epoch_file)
    else:
        return None
