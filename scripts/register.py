import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from argparse import ArgumentParser
import torchio as tio
from pathlib import Path

from keymorph.utils import rescale_intensity
from keymorph.model import KeyMorph
from keymorph.unet3d.model import UNet2D, UNet3D, TruncatedUNet3D
from keymorph.net import ConvNet
from scripts.pairwise_register_eval import run_eval
from scripts.groupwise_register_eval import run_group_eval
from scripts.script_utils import summary, load_checkpoint


def parse_args():
    parser = ArgumentParser()

    # I/O
    parser.add_argument(
        "--gpus", type=str, default="0", help="Which GPUs to use? Index from 0"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        dest="save_dir",
        default="./register_output/",
        help="Path to the folder where outputs are saved",
    )
    parser.add_argument(
        "--load_path", type=str, default=None, help="Load checkpoint at .h5 path"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights/",
        help="Directory where keymorph model weights are saved",
    )
    parser.add_argument(
        "--save_eval_to_disk", action="store_true", help="Perform evaluation"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize images and points"
    )
    parser.add_argument("--debug_mode", action="store_true", help="Debug mode")

    # KeyMorph
    parser.add_argument(
        "--registration_model", type=str, default="keymorph", help="Registration model"
    )
    parser.add_argument(
        "--num_keypoints", type=int, required=True, help="Number of keypoints"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="truncatedunet",
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
        "--norm_type",
        type=str,
        default="instance",
        choices=["none", "instance", "batch", "group"],
        help="Normalization type",
    )

    parser.add_argument(
        "--weighted_kp_align",
        type=str,
        default="power",
        choices=[None, "variance", "power"],
        help="Type of weighting to use for keypoints",
    )

    parser.add_argument(
        "--list_of_aligns",
        nargs="*",
        default=("affine",),
        help="Alignments to use for KeyMorph",
    )

    parser.add_argument(
        "--list_of_metrics",
        nargs="*",
        default=("mse",),
        help="Metrics to report",
    )

    # Data
    parser.add_argument("--moving", type=str, required=True, help="Moving image path")

    parser.add_argument("--fixed", type=str, required=True, help="Fixed image path")

    parser.add_argument("--moving_seg", type=str, default=None, help="Moving seg path")

    parser.add_argument("--fixed_seg", type=str, default=None, help="Fixed seg path")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument(
        "--half_resolution",
        action="store_true",
        help="Evaluate on half-resolution models",
    )

    parser.add_argument(
        "--early_stop_eval_subjects",
        type=int,
        default=None,
        help="Early stop number of test subjects for fast eval",
    )

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=23,
        help="Random seed use to sort the training data",
    )

    parser.add_argument("--dim", type=int, default=3)

    parser.add_argument("--use_amp", action="store_true", help="Use AMP")

    parser.add_argument(
        "--groupwise", action="store_true", help="Perform groupwise registration"
    )

    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Use torch.utils.checkpoint",
    )

    parser.add_argument(
        "--num_resolutions_for_itkelastix", type=int, default=4, help="Num resolutions"
    )

    args = parser.parse_args()
    return args


def build_tio_subject(img_path, seg_path=None):
    _dict = {"img": tio.ScalarImage(img_path)}
    if seg_path is not None:
        _dict["seg"] = tio.LabelMap(seg_path)
    return tio.Subject(**_dict)


def get_loaders(args):
    args.seg_available = args.moving_seg is not None and args.fixed_seg is not None
    if os.path.isfile(args.moving) and os.path.isfile(args.fixed):
        if args.seg_available:
            moving = [build_tio_subject(args.moving, args.moving_seg)]
            fixed = [build_tio_subject(args.fixed, args.fixed_seg)]
        else:
            moving = [build_tio_subject(args.moving)]
            fixed = [build_tio_subject(args.fixed)]
    elif os.path.isdir(args.moving) and os.path.isdir(args.fixed):
        fixed, moving = [], []
        for moving_name in os.listdir(args.moving):
            moving_path = os.path.join(args.moving, moving_name)
            if args.seg_available:
                moving_seg_path = os.path.join(args.moving_seg, moving_name)
            else:
                moving_seg_path = None
            moving.append(build_tio_subject(moving_path, moving_seg_path))
        for fixed_name in os.listdir(args.fixed):
            fixed_path = os.path.join(args.fixed, fixed_name)
            if args.seg_available:
                fixed_seg_path = os.path.join(args.fixed_seg, fixed_name)
            else:
                fixed_seg_path = None
            fixed.append(build_tio_subject(fixed_path, fixed_seg_path))

    # Build dataset
    fixed_dataset = tio.SubjectsDataset(fixed, transform=transform)
    moving_dataset = tio.SubjectsDataset(moving, transform=transform)
    fixed_loader = DataLoader(
        fixed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    moving_loader = DataLoader(
        moving_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    loaders = {"fixed": fixed_loader, "moving": moving_loader}
    return loaders


def get_foundation_weights_path(weights_dir, num_keypoints, num_levels):
    template_name = "foundation-numkey{}-numlevels{}.pth.tar"
    return os.path.join(weights_dir, template_name.format(num_keypoints, num_levels))


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
            weight_keypoints=args.weighted_kp_align,
        )
        registration_model.to(args.device)
        summary(registration_model)
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


if __name__ == "__main__":
    args = parse_args()

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpus))
    else:
        args.device = torch.device("cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("Number of GPUs: {}".format(torch.cuda.device_count()))

    # Create save path
    save_path = Path(args.save_dir)
    if not os.path.exists(save_path) and args.save_eval_to_disk:
        os.makedirs(save_path)
    args.model_eval_dir = save_path

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    if args.half_resolution:
        transform = tio.Compose(
            [
                tio.Lambda(lambda x: x.permute(0, 1, 3, 2)),
                tio.Resize(128),
                tio.Lambda(rescale_intensity, include=("img",)),
            ]
        )
    else:
        transform = tio.Compose(
            [
                tio.ToCanonical(),
                tio.Resample(1),
                tio.Resample("img"),
                tio.CropOrPad((256, 256, 256), padding_mode=0, include=("img",)),
                tio.CropOrPad((256, 256, 256), padding_mode=0, include=("seg",)),
                tio.Lambda(rescale_intensity, include=("img",)),
            ],
            include=("img", "seg"),
        )

    # Loaders
    loaders = get_loaders(args)

    # Evaluation parameters
    list_of_eval_names = [("fixed", "moving")]
    list_of_eval_augs = ["rot0"]

    # Model
    registration_model = get_model(args)
    registration_model.eval()

    # Checkpoint loading
    if args.half_resolution and args.registration_model == "keymorph":
        assert (
            args.load_path is not None
        ), "Must specify path for weights for half resolution models"
    else:
        args.load_path = get_foundation_weights_path(
            args.weights_dir, args.num_keypoints, args.num_levels_for_unet
        )
    if args.load_path is not None:
        print(f"Loading checkpoint from {args.load_path}")
        ckpt_state, registration_model = load_checkpoint(
            args.load_path,
            registration_model,
            device=args.device,
        )

    if args.groupwise:
        print("running groupwise")
        run_group_eval(
            group_loader,
            registration_model,
            args.list_of_metrics,
            list_of_eval_names,
            list_of_eval_augs,
            args.list_of_aligns,
            list_of_group_sizes,
            args,
            save_dir_prefix="group_eval",
        )
    else:
        run_eval(
            loaders,
            registration_model,
            args.list_of_metrics,
            list_of_eval_names,
            list_of_eval_augs,
            args.list_of_aligns,
            args,
            save_dir_prefix="eval",
        )
