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

from keymorph.net import ConvNet, RXFM_Net, MedNeXt
from keymorph.model import KeyMorph
from keymorph import utils
from keymorph.utils import (
    ParseKwargs,
    initialize_wandb,
    save_dict_as_json,
)
from dataset import ixi, gigamed, synthbrain
from keymorph.cm_plotter import show_warped_vol
import scripts.gigamed_hyperparameters as gigamed_hps
import scripts.ixi_hyperparameters as ixi_hps
from scripts.train import run_train
from scripts.pretrain import run_pretrain
from scripts.pairwise_register_eval import run_eval
from scripts.groupwise_register_eval import run_group_eval, run_long_eval


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
        "--registration_model", type=str, required=True, help="Registration model"
    )

    parser.add_argument(
        "--num_keypoints", type=int, required=True, help="Number of keypoints"
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
        default=14,
        help="Number of channels to compute Dice loss, to save memory",
    )
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
        "--train_family_params",
        type=str,
        default="default",
        choices=["default", "mse_only", "tps0_only"],
        help="Type of training family parameters to use",
    )

    # Data
    parser.add_argument(
        "--train_dataset", help="<Required> Train datasets", required=True
    )
    parser.add_argument("--test_dataset", type=str, required=True, help="Test Dataset")

    parser.add_argument(
        "--mix_modalities",
        action="store_true",
        help="Whether or not to mix modalities amongst image pairs",
    )

    parser.add_argument("--num_workers", type=int, default=1, help="Num workers")

    parser.add_argument(
        "--seg_available",
        action="store_true",
        help="Whether or not segmentation maps are available for the dataset",
    )

    parser.add_argument(
        "--crop_dims_for_train", type=int, default=256, help="Num workers"
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

    parser.add_argument("--transform", type=str, default="none")

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
        dataset_str = args.test_dataset
    elif args.run_mode == "pretrain":
        prefix = "__pretrain__"
        dataset_str = args.train_dataset
    else:
        prefix = "__training__"
        dataset_str = args.train_dataset
    arguments = (
        prefix
        + args.job_name
        + "_dataset"
        + dataset_str
        + "_model"
        + str(args.registration_model)
        + "_keypoints"
        + str(args.num_keypoints)
        + "_batch"
        + str(args.batch_size)
        + "_normType"
        + str(args.norm_type)
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


def get_data(args):
    # Train dataset
    if args.train_dataset == "ixi":
        train_loader, _ = ixi.get_loaders()
    elif args.train_dataset == "gigamed":
        crop_dims = (args.crop_dims_for_train,) * args.dim
        transform = tio.Compose(
            [
                tio.CropOrPad(crop_dims, padding_mode=0, include=("img",)),
                tio.CropOrPad(crop_dims, padding_mode=0, include=("seg",)),
            ]
        )
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            transform=transform,
        )
        gigamed_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            transform=transform,
        )
        train_loader = gigamed_dataset_with_seg.get_train_loader()
        pretrain_loader = gigamed_dataset.get_pretrain_loader()
        ref_subject = gigamed_dataset.get_reference_subject()
    elif args.train_dataset == "gigamednb":
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            normal_brains_only=True,
        )
        gigamed_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            normal_brains_only=True,
        )
        train_loader = gigamed_dataset_with_seg.get_train_loader()
        pretrain_loader = gigamed_dataset.get_pretrain_loader()
        ref_subject = gigamed_dataset.get_reference_subject()
    elif args.train_dataset == "synthbrain":
        synthbrain_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            synthetic_brains_only=True,
        )
        synthbrain_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            synthetic_brains_only=True,
        )
        train_loader = synthbrain_dataset_with_seg.get_train_loader()
        pretrain_loader = synthbrain_dataset.get_pretrain_loader()
        ref_subject = synthbrain_dataset.get_reference_subject()
    elif args.train_dataset == "synthbraingenerated":
        synthbrain_dataset = synthbrain.SynthBrain(
            args.batch_size,
            args.num_workers,
        )
        train_loader = synthbrain_dataset.get_train_loader()
        pretrain_loader = synthbrain_dataset.get_pretrain_loader()

        # Use Gigamed's first subject as reference
        synthbrain_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            synthetic_brains_only=True,
        )
        ref_subject = synthbrain_dataset.get_reference_subject()
    elif args.train_dataset == "gigamed+synthbrain":
        gigamed_synthbrain_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            include_synthetic_brains=True,
        )
        gigamed_synthbrain_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            include_synthetic_brains=True,
        )
        train_loader = gigamed_synthbrain_dataset_with_seg.get_train_loader()
        pretrain_loader = gigamed_synthbrain_dataset.get_pretrain_loader()
        ref_subject = gigamed_synthbrain_dataset.get_reference_subject()
    elif args.train_dataset == "gigamed+synthbrain+randomanisotropy":
        transform = tio.Compose(
            [
                tio.RandomAnisotropy(downsampling=(1, 3)),
            ]
        )
        gigamed_synthbrain_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            transform=transform,
            include_synthetic_brains=True,
        )
        gigamed_synthbrain_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            transform=transform,
            include_synthetic_brains=True,
        )
        train_loader = gigamed_synthbrain_dataset_with_seg.get_train_loader()
        pretrain_loader = gigamed_synthbrain_dataset.get_pretrain_loader()
        ref_subject = gigamed_synthbrain_dataset.get_reference_subject()
    else:
        raise ValueError('Invalid train datasets "{}"'.format(args.train_dataset))

    # Eval dataset
    if args.test_dataset == "ixi":
        _, id_eval_loaders = ixi.get_loaders()

        return train_loader, {
            "id": id_eval_loaders,
        }
    elif args.test_dataset == "gigamed":
        eval_transform = tio.Compose(
            [
                tio.ToCanonical(),
                tio.Resample(1),
                tio.Resample("img"),
                tio.CropOrPad((256, 256, 256), padding_mode=0, include=("img",)),
                tio.CropOrPad((256, 256, 256), padding_mode=0, include=("seg",)),
            ],
            include=("img", "seg"),
        )
        eval_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            transform=eval_transform,
        )

        return {
            "pretrain": pretrain_loader,
            "train": train_loader,
            "eval": {
                "ss_unimodal": eval_dataset.get_eval_loaders(ss=True),
                "ss_multimodal": eval_dataset.get_eval_loaders(ss=True),
                "ss_lesion": eval_dataset.get_eval_lesion_loaders(ss=True),
                "ss_group": eval_dataset.get_eval_group_loaders(ss=True),
                "ss_long": eval_dataset.get_eval_longitudinal_loaders(ss=True),
                "nss_unimodal": eval_dataset.get_eval_loaders(ss=False),
                "nss_multimodal": eval_dataset.get_eval_loaders(ss=False),
                "nss_lesion": eval_dataset.get_eval_lesion_loaders(ss=False),
                "nss_group": eval_dataset.get_eval_group_loaders(ss=False),
                "nss_long": eval_dataset.get_eval_longitudinal_loaders(ss=False),
            },
            "ref_subject": ref_subject,
        }
    else:
        raise ValueError('Invalid test dataset "{}"'.format(args.test_dataset))


def get_model(args):
    if args.registration_model == "keymorph":
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
        elif args.backbone == "se3cnn":
            network = RXFM_Net(1, args.num_keypoints, norm_type=args.norm_type)
        elif args.backbone == "mednext":
            network = MedNeXt(
                1, args.num_keypoints, model_id="L", norm_type=args.norm_type
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
    elif args.registration_model == "itkelastix":
        from keymorph.baselines.itkelastix import ITKElastix

        registration_model = ITKElastix()
    elif args.registration_model == "synthmorph":

        from keymorph.baselines.voxelmorph import VoxelMorph

        registration_model = VoxelMorph(perform_preaffine_register=True)
    elif args.registration_model == "synthmorph-no-preaffine":

        from keymorph.baselines.voxelmorph import VoxelMorph

        registration_model = VoxelMorph(perform_preaffine_register=False)
    elif args.registration_model == "ants":
        from keymorph.baselines.ants import ANTs

        registration_model = ANTs()
    else:
        raise ValueError(
            'Invalid registration model "{}"'.format(args.registration_model)
        )
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
    loaders = get_data(args)

    # Model
    registration_model = get_model(args)

    # Optimizer
    if isinstance(registration_model, torch.nn.Module):
        optimizer = torch.optim.Adam(registration_model.parameters(), lr=args.lr)
    else:
        optimizer = None

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
            resume=args.resume,
            device=args.device,
        )

    if args.run_mode == "eval":
        if args.registration_model == "keymorph":
            assert args.load_path is not None, "Need to load a checkpoint for eval"
        assert args.batch_size == 1, ":("
        registration_model.eval()

        if args.test_dataset == "ixi":
            list_of_eval_unimodal_names = ixi_hps.EVAL_UNI_NAMES
            list_of_eval_multimodal_names = ixi_hps.EVAL_MULTI_NAMES
            list_of_eval_lesion_names = ixi_hps.EVAL_LESION_NAMES
            list_of_eval_group_names = ixi_hps.EVAL_GROUP_NAMES
            list_of_eval_long_names = ixi_hps.EVAL_LONG_NAMES
        elif args.test_dataset == "gigamed":
            list_of_eval_metrics = gigamed_hps.EVAL_METRICS
            list_of_eval_augs = gigamed_hps.EVAL_AUGS
            list_of_eval_aligns = gigamed_hps.MODEL_HPS[args.registration_model][
                "aligns"
            ]
            list_of_eval_group_aligns = gigamed_hps.MODEL_HPS[args.registration_model][
                "group_aligns"
            ]
            list_of_eval_long_aligns = gigamed_hps.MODEL_HPS[args.registration_model][
                "long_aligns"
            ]
            list_of_eval_unimodal_names = gigamed_hps.EVAL_UNI_NAMES
            list_of_eval_multimodal_names = gigamed_hps.EVAL_MULTI_NAMES
            list_of_eval_lesion_names = gigamed_hps.EVAL_LESION_NAMES
            list_of_eval_group_names = gigamed_hps.EVAL_GROUP_NAMES
            list_of_eval_long_names = gigamed_hps.EVAL_LONG_NAMES
            perform_groupwise_experiments = gigamed_hps.MODEL_HPS[
                args.registration_model
            ]["perform_groupwise_experiments"]
            if perform_groupwise_experiments:
                list_of_eval_group_sizes = gigamed_hps.MODEL_HPS[
                    args.registration_model
                ]["group_sizes"]

        print("\n\nStarting evaluation...")
        for dist in ["ss", "nss"]:
            print("Perform eval on", dist)

            # Pairwise Unimodal
            experiment_name = f"{dist}_unimodal"
            json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
            if not os.path.exists(json_path):
                eval_metrics = run_eval(
                    loaders["eval"][experiment_name],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_unimodal_names[dist],
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
            experiment_name = f"{dist}_multimodal"
            json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
            if not os.path.exists(json_path):
                eval_metrics = run_eval(
                    loaders["eval"][experiment_name],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_multimodal_names[dist],
                    list_of_eval_augs,
                    list_of_eval_aligns,
                    args,
                    save_dir_prefix=experiment_name,
                )
                if not args.debug_mode:
                    save_dict_as_json(eval_metrics, json_path)
            else:
                print("-> Skipping eval on", experiment_name)

            # Lesions
            experiment_name = f"{dist}_lesion"
            json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
            if not os.path.exists(json_path) and list_of_eval_lesion_names is not None:
                lesion_metrics = run_eval(
                    loaders["eval"][experiment_name],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_lesion_names[dist],
                    list_of_eval_augs,
                    list_of_eval_aligns,
                    args,
                    save_dir_prefix=experiment_name,
                )
                if not args.debug_mode:
                    save_dict_as_json(
                        lesion_metrics,
                        json_path,
                    )
            else:
                print("-> Skipping eval on", experiment_name)

        if perform_groupwise_experiments:
            for dist in ["ss", "nss"]:
                print("Perform groupwise eval on", dist)
                # Longitudinal
                experiment_name = f"{dist}_long"
                json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
                if (
                    not os.path.exists(json_path)
                    and list_of_eval_long_names is not None
                ):
                    long_metrics = run_long_eval(
                        loaders["eval"][experiment_name],
                        registration_model,
                        list_of_eval_metrics,
                        list_of_eval_long_names[dist],
                        list_of_eval_augs,
                        list_of_eval_long_aligns,
                        args,
                        save_dir_prefix=experiment_name,
                    )
                    if not args.debug_mode:
                        save_dict_as_json(long_metrics, json_path)
                else:
                    print("-> Skipping eval on", experiment_name)

                print("Perform groupwise eval on", dist)
                experiment_name = f"{dist}_group"
                json_path = args.model_eval_dir / f"summary_{experiment_name}.json"
                if (
                    not os.path.exists(json_path)
                    and list_of_eval_group_names is not None
                ):
                    group_metrics = run_group_eval(
                        loaders["eval"][experiment_name],
                        registration_model,
                        list_of_eval_metrics,
                        list_of_eval_group_names[dist],
                        list_of_eval_augs[:1],  # Ignore augmentation for groupwise
                        list_of_eval_group_aligns,
                        list_of_eval_group_sizes,
                        args,
                        save_dir_prefix=experiment_name,
                    )
                    if not args.debug_mode:
                        save_dict_as_json(group_metrics, json_path)
                else:
                    print("-> Skipping eval on", experiment_name)
        else:
            print("Skipping groupwise experiments!")
        print("\nEvaluation done!")

    elif args.run_mode == "pretrain":
        assert args.registration_model == "keymorph"

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
            # Extract random keypoints from reference subject
            ref_subject = loaders["ref_subject"]
            ref_img = ref_subject["img"][tio.DATA].float().unsqueeze(0)
            print("sampling random keypoints...")
            random_points = utils.sample_valid_coordinates(
                ref_img, args.num_keypoints, args.dim
            )
            random_points = random_points * 2 - 1
            random_points = random_points.repeat(args.batch_size, 1, 1)
            if args.visualize:
                show_warped_vol(
                    ref_img[0, 0].cpu().detach().numpy(),
                    ref_img[0, 0].cpu().detach().numpy(),
                    ref_img[0, 0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                )
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

        if args.train_dataset in ["gigamed", "synthbrain"]:
            if args.train_family_params == "mse_only":
                train_params = gigamed_hps.GIGAMED_FAMILY_TRAIN_PARAMS_MSE_ONLY
            elif args.train_family_params == "tps0_only":
                train_params = gigamed_hps.GIGAMED_FAMILY_TRAIN_PARAMS_TPS0_ONLY
            else:
                train_params = gigamed_hps.GIGAMED_FAMILY_TRAIN_PARAMS
        else:
            raise ValueError(
                'No train parameters found for "{}"'.format(args.train_dataset)
            )

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
                train_params,
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
