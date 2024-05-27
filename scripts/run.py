import os
from pprint import pprint
import torch
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path
import wandb
import torchio as tio
import json
from copy import deepcopy
from keymorph.unet3d.model import UNet2D, UNet3D, TruncatedUNet3D

from keymorph.net import ConvNet
from keymorph.model import KeyMorph
from keymorph import utils
from keymorph.utils import (
    ParseKwargs,
    initialize_wandb,
    save_dict_as_json,
)
from dataset import csv_dataset, ixi_dataset
import scripts.hyperparameters as hps
from scripts.train import run_train
from scripts.pretrain import run_pretrain
from scripts.pairwise_register_eval import run_eval


def parse_args():
    parser = ArgumentParser()

    # I/O
    parser.add_argument(
        "--job_name",
        type=str,
        default="keymorph",
        help="Name of job",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/",
        help="Path to the folder where outputs are saved",
    )
    parser.add_argument(
        "--load_path", type=str, default=None, help="Load checkpoint at .h5 path"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume checkpoint, must set --load_path"
    )
    parser.add_argument(
        "--resume_latest",
        action="store_true",
        help="Resume latest checkpoint available",
    )
    parser.add_argument(
        "--save_eval_to_disk",
        action="store_true",
        help="Save evaluation results to disk",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize images and points"
    )
    parser.add_argument(
        "--log_interval", type=int, default=25, help="Frequency of logs"
    )

    # KeyMorph
    parser.add_argument(
        "--num_keypoints", type=int, required=True, help="Number of keypoints"
    )
    parser.add_argument("--loss_fn", type=str, required=True, help="Loss function")
    parser.add_argument(
        "--transform_type", type=str, required=True, help="Type of keypoint alignment"
    )
    parser.add_argument(
        "--max_train_keypoints",
        type=int,
        default=64,
        help="Number of keypoints to subsample TPS, to save memory",
    )
    parser.add_argument(
        "--max_train_seg_channels",
        type=int,
        default=None,
        help="Number of channels to compute Dice loss, to save memory",
    )
    parser.add_argument(
        "--kp_layer",
        type=str,
        default="com",
        choices=["com", "linear"],
        help="Keypoint layer module to use",
    )
    parser.add_argument(
        "--kpconsistency_coeff",
        type=float,
        default=0,
        help="Minimize keypoint consistency loss",
    )
    parser.add_argument(
        "--weighted_kp_align",
        type=str,
        default=None,
        choices=["variance", "power"],
        help="Type of weighting to use for keypoints",
    )
    parser.add_argument(
        "--compute_subgrids_for_tps",
        action="store_true",
        help="Use subgrids for computing TPS",
    )
    parser.add_argument(
        "--num_subgrids",
        type=int,
        default=4,
        help="Number of subgrids for computing TPS",
    )
    parser.add_argument(
        "--max_random_affine_augment_params",
        nargs="*",
        default=(0.0, 0.0, 0.0, 0.0),
        help="Maximum of affine augmentations during training",
    )

    # CNN backbone
    parser.add_argument(
        "--backbone",
        type=str,
        default="conv",
        help="Keypoint extractor module to use",
    )
    parser.add_argument(
        "--num_truncated_layers_for_truncatedunet",
        type=int,
        default=1,
        help="Number of truncated layers for truncated unet",
    )
    parser.add_argument(
        "--num_levels_for_unet",
        type=int,
        default=4,
        help="Number of levels for unet",
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Training dataset to use",
    )
    parser.add_argument(
        "--mix_modalities",
        action="store_true",
        help="Whether or not to mix modalities amongst image pairs",
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Num workers")
    parser.add_argument(
        "--num_test_subjects", type=int, default=100, help="Num test subjects"
    )

    # ML
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--norm_type",
        type=str,
        default="instance",
        choices=["none", "instance", "batch", "group"],
        help="Normalization type",
    )
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Training Epochs")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=32,
        help="Number of gradient steps per epoch",
    )
    parser.add_argument(
        "--affine_slope",
        type=int,
        default=1,
        help="Constant to control how slow to increase augmentation. If negative, disabled.",
    )

    # Miscellaneous
    parser.add_argument(
        "--run_mode",
        required=True,
        choices=["train", "pretrain", "eval"],
        help="Run mode",
    )
    parser.add_argument("--debug_mode", action="store_true", help="Debug mode")
    parser.add_argument(
        "--seed", type=int, default=23, help="Random seed use to sort the training data"
    )
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--use_amp", action="store_true", help="Use AMP")
    parser.add_argument(
        "--early_stop_eval_subjects",
        type=int,
        default=None,
        help="Early stop number of test subjects for fast eval",
    )
    parser.add_argument(
        "--num_resolutions_for_itkelastix", type=int, default=4, help="Num resolutions"
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Use torch.utils.checkpoint",
    )
    parser.add_argument(
        "--use_profiler",
        action="store_true",
        help="Use torch profiler",
    )

    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb")
    parser.add_argument(
        "--wandb_api_key_path",
        type=str,
        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.",
    )
    parser.add_argument(
        "--wandb_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for wandb.init() passed as key1=value1 key2=value2",
    )

    args = parser.parse_args()
    return args


def create_dirs(args):
    arg_dict = vars(deepcopy(args))

    # Path to save outputs
    if args.run_mode == "eval":
        prefix = "__eval__"
    elif args.run_mode == "pretrain":
        prefix = "__pretrain__"
    else:
        prefix = "__training__"
    arguments = (
        prefix
        + args.job_name
        + "_keypoints"
        + str(args.num_keypoints)
        + "_batch"
        + str(args.batch_size)
        + "_lr"
        + str(args.lr)
    )

    args.model_dir = Path(args.save_dir) / arguments
    if not os.path.exists(args.model_dir) and not args.debug_mode:
        print("Creating directory: {}".format(args.model_dir))
        os.makedirs(args.model_dir)

    if args.run_mode == "eval" and args.save_eval_to_disk:
        args.model_eval_dir = args.model_dir / "eval"
        if not os.path.exists(args.model_eval_dir) and not args.debug_mode:
            os.makedirs(args.model_eval_dir)

    else:
        args.model_ckpt_dir = args.model_dir / "checkpoints"
        if not os.path.exists(args.model_ckpt_dir) and not args.debug_mode:
            os.makedirs(args.model_ckpt_dir)
        args.model_img_dir = args.model_dir / "train_img"
        if not os.path.exists(args.model_img_dir) and not args.debug_mode:
            os.makedirs(args.model_img_dir)

    # Write arguments to json
    if not args.debug_mode:
        with open(os.path.join(args.model_dir, "args.json"), "w") as outfile:
            json.dump(arg_dict, outfile, sort_keys=True, indent=4)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_data(transform, args):
    if args.train_dataset == "csv":
        dataset = csv_dataset.CSVDataset(args.data_path)
    elif args.train_dataset == "ixi":
        dataset = ixi_dataset.IXIDataset(args.data_path)
    else:
        raise ValueError('Invalid dataset "{}"'.format(args.train_dataset))

    pretrain_loader, train_loader, id_eval_loaders = dataset.get_loaders(
        args.batch_size,
        args.num_workers,
        args.mix_modalities,
        transform=transform,
    )
    args.seg_available = dataset.seg_available
    return {
        "pretrain": pretrain_loader,
        "train": train_loader,
        "eval": id_eval_loaders,
    }


def get_model(args):
    # CNN, i.e. keypoint extractor
    if args.backbone == "conv":
        network = ConvNet(
            args.dim,
            1,
            args.num_keypoints,
            norm_type=args.norm_type,
        )
    elif args.backbone == "unet":
        if args.dim == 2:
            network = UNet2D(
                1,
                args.num_keypoints,
                final_sigmoid=False,
                f_maps=64,
                layer_order="gcr",
                num_groups=8,
                num_levels=args.num_levels_for_unet,
                is_segmentation=False,
                conv_padding=1,
            )
        if args.dim == 3:
            network = UNet3D(
                1,
                args.num_keypoints,
                final_sigmoid=False,
                f_maps=32,  # Used by nnUNet
                layer_order="gcr",
                num_groups=8,
                num_levels=args.num_levels_for_unet,
                is_segmentation=False,
                conv_padding=1,
                use_checkpoint=args.use_checkpoint,
            )
    elif args.backbone == "truncatedunet":
        if args.dim == 3:
            network = TruncatedUNet3D(
                1,
                args.num_keypoints,
                args.num_truncated_layers_for_truncatedunet,
                final_sigmoid=False,
                f_maps=32,  # Used by nnUNet
                layer_order="gcr",
                num_groups=8,
                num_levels=args.num_levels_for_unet,
                is_segmentation=False,
                conv_padding=1,
            )
    else:
        raise ValueError('Invalid keypoint extractor "{}"'.format(args.backbone))
    network = torch.nn.DataParallel(network)

    # Keypoint model
    registration_model = KeyMorph(
        network,
        args.num_keypoints,
        args.dim,
        use_amp=args.use_amp,
        use_checkpoint=args.use_checkpoint,
        max_train_keypoints=args.max_train_keypoints,
        weight_keypoints=args.weighted_kp_align,
    )
    registration_model.to(args.device)
    utils.summary(registration_model)
    return registration_model


def main():
    args = parse_args()
    if args.debug_mode:
        args.steps_per_epoch = 3
        args.early_stop_eval_subjects = 1
    pprint(vars(args))

    # Create run directories
    create_dirs(args)

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        print("WARNING! No GPU available, using the CPU instead...")
        args.device = torch.device("cpu")
    print("Number of GPUs: {}".format(torch.cuda.device_count()))

    # Set seed
    set_seed(args)

    # Data
    transform = hps.TRANSFORM
    loaders = get_data(transform, args)

    # Model
    registration_model = get_model(args)

    # Optimizer
    optimizer = torch.optim.Adam(registration_model.parameters(), lr=args.lr)

    # Checkpoint loading
    if args.resume_latest:
        args.resume = True
        args.load_path = utils.get_latest_epoch_file(args.model_ckpt_dir)
        if args.load_path is None:
            raise ValueError(
                f"No checkpoint found to resume from: {args.model_ckpt_dir}"
            )
    if args.load_path is not None:
        print(f"Loading checkpoint from {args.load_path}")
        ckpt_state, registration_model, optimizer = utils.load_checkpoint(
            args.load_path,
            registration_model,
            optimizer,
            device=args.device,
        )

    if args.run_mode == "eval":
        assert args.load_path is not None, "Need to load a checkpoint for eval"
        assert args.batch_size == 1, ":("
        registration_model.eval()

        list_of_eval_unimodal_names = hps.EVAL_UNI_NAMES
        list_of_eval_multimodal_names = hps.EVAL_MULTI_NAMES
        list_of_eval_metrics = hps.EVAL_METRICS
        list_of_eval_augs = hps.EVAL_AUGS
        list_of_eval_aligns = hps.EVAL_KP_ALIGNS

        print("\n\nStarting evaluation...")
        # Pairwise Unimodal
        experiment_name = "unimodal"
        json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
        if not os.path.exists(json_path):
            eval_metrics = run_eval(
                loaders["eval"],
                registration_model,
                list_of_eval_metrics,
                list_of_eval_unimodal_names,
                list_of_eval_augs,
                list_of_eval_aligns,
                args,
                save_dir_prefix=experiment_name,
            )
            if not args.debug_mode:
                save_dict_as_json(eval_metrics, json_path)
        else:
            print("-> Skipping eval on", experiment_name)

        # Pairwise Multimodal
        experiment_name = "multimodal"
        json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
        if not os.path.exists(json_path):
            eval_metrics = run_eval(
                loaders["eval"],
                registration_model,
                list_of_eval_metrics,
                list_of_eval_multimodal_names,
                list_of_eval_augs,
                list_of_eval_aligns,
                args,
                save_dir_prefix=experiment_name,
            )
            if not args.debug_mode:
                save_dict_as_json(eval_metrics, json_path)
        else:
            print("-> Skipping eval on", experiment_name)

        print("\nEvaluation done!")

    elif args.run_mode == "pretrain":
        registration_model.train()
        train_loss = []

        if args.use_wandb and not args.debug_mode:
            initialize_wandb(args)

        if args.resume:
            start_epoch = ckpt_state["epoch"] + 1
            # Load random keypoints from checkpoint
            random_points = state["random_points"]
        else:
            start_epoch = 1
            # Extract random keypoints from reference subject, which is any random training subject
            ref_subject = next(iter(loaders["pretrain"]))
            ref_img = ref_subject["img"][tio.DATA].float()
            print("sampling random keypoints...")
            random_points = utils.sample_valid_coordinates(
                ref_img, args.num_keypoints, args.dim
            )
            random_points = random_points * 2 - 1
            random_points = random_points.repeat(args.batch_size, 1, 1)
            # if args.visualize:
            #     show_warped_vol(
            #         ref_img[0, 0].cpu().detach().numpy(),
            #         ref_img[0, 0].cpu().detach().numpy(),
            #         ref_img[0, 0].cpu().detach().numpy(),
            #         random_points[0].cpu().detach().numpy(),
            #         random_points[0].cpu().detach().numpy(),
            #         random_points[0].cpu().detach().numpy(),
            #     )
            del ref_subject

        for epoch in range(start_epoch, args.epochs + 1):
            args.curr_epoch = epoch
            epoch_stats = run_pretrain(
                loaders["pretrain"],
                random_points,
                registration_model,
                optimizer,
                args,
            )

            train_loss.append(epoch_stats["loss"])

            print(f"Epoch {epoch}/{args.epochs}")
            for name, metric in epoch_stats.items():
                print(f"[Train Stat] {name}: {metric:.5f}")

            if args.use_wandb and not args.debug_mode:
                wandb.log(epoch_stats)

            # Save model
            if epoch % args.log_interval == 0 and not args.debug_mode:
                state = {
                    "epoch": epoch,
                    "args": args,
                    "state_dict": registration_model.backbone.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "random_points": random_points,
                }
                torch.save(
                    state,
                    os.path.join(
                        args.model_ckpt_dir,
                        "pretrained_epoch{}_model.pth.tar".format(epoch),
                    ),
                )
    else:
        registration_model.train()
        train_loss = []

        if args.use_wandb and not args.debug_mode:
            initialize_wandb(args)

        if args.resume:
            start_epoch = ckpt_state["epoch"] + 1
        else:
            start_epoch = 1

        for epoch in range(start_epoch, args.epochs + 1):
            args.curr_epoch = epoch
            epoch_stats = run_train(
                loaders["train"],
                registration_model,
                optimizer,
                args,
            )
            train_loss.append(epoch_stats["loss"])

            print(f"\nEpoch {epoch}/{args.epochs}")
            for metric_name, metric in epoch_stats.items():
                print(f"[Train Stat] {metric_name}: {metric:.5f}")

            if args.use_wandb and not args.debug_mode:
                wandb.log(epoch_stats)

            # Save model
            if epoch % args.log_interval == 0 and not args.debug_mode:
                state = {
                    "epoch": epoch,
                    "args": args,
                    "state_dict": registration_model.backbone.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state,
                    os.path.join(
                        args.model_ckpt_dir,
                        "epoch{}_trained_model.pth.tar".format(epoch),
                    ),
                )


if __name__ == "__main__":
    main()
