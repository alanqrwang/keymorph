import os
from pprint import pprint
import torch
import torch.nn.functional as F
import time
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path
import wandb
import torchio as tio
import json
from copy import deepcopy
import nibabel as nib

from itkelastix.register import ITKElastix
from keymorph import loss_ops
from keymorph.net import ConvNet, UNet, RXFM_Net, TruncatedUNet
from keymorph.model import KeyMorph
from keymorph import utils
from keymorph.utils import (
    ParseKwargs,
    initialize_wandb,
    align_img,
    save_summary_json,
)
from gigamed import ixi, gigamed, synthbrain
from keymorph.augmentation import (
    affine_augment,
    random_affine_augment,
)
from keymorph.cm_plotter import show_warped, show_warped_vol, plot_groupwise_register
import gigamed_eval_hyperparameters as gigamed_hps
import ixi_eval_hyperparameters as ixi_hps


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

    parser.add_argument("--save_preds", action="store_true", help="Perform evaluation")

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
        choices=["conv", "unet", "se3cnn", "se3cnn2", "truncatedunet"],
        help="Keypoint extractor module to use",
    )
    parser.add_argument(
        "--num_truncated_layers_for_truncatedunet",
        type=int,
        default=1,
        help="Number of truncated layers for truncated unet",
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
        "--group_size",
        type=int,
        default=4,
        help="Group size for groupwise registration evaluation",
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
        dataset_str = args.test_dataset
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

    if args.run_mode == "eval":
        args.model_result_dir = args.model_dir / "eval_results"
        if not os.path.exists(args.model_result_dir) and not args.debug_mode:
            os.makedirs(args.model_result_dir)
        args.model_eval_dir = args.model_dir / "eval_img"
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
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            group_size=args.group_size,
        )
        gigamed_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            group_size=args.group_size,
        )
        train_loader = gigamed_dataset_with_seg.get_train_loader()
        pretrain_loader = gigamed_dataset.get_pretrain_loader()
        ref_subject = gigamed_dataset.get_reference_subject()
    elif args.train_dataset == "gigamednb":
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=False,
            group_size=args.group_size,
            normal_brains_only=True,
        )
        gigamed_dataset_with_seg = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=True,
            group_size=args.group_size,
            normal_brains_only=True,
        )
        train_loader = gigamed_dataset_with_seg.get_train_loader()
        pretrain_loader = gigamed_dataset.get_pretrain_loader()
        ref_subject = gigamed_dataset.get_reference_subject()
    elif args.train_dataset == "synthbrain":
        synth_dataset = synthbrain.SynthBrain(args.batch_size, args.num_workers)
        train_loader = synth_dataset.get_train_loader()
    elif args.train_dataset == "gigamed+synthbrain":
        gigamed_synthbrain_dataset = gigamed.GigaMedSynthBrain(
            args.batch_size, args.num_workers, include_seg=args.seg_available
        )
        train_loader = gigamed_synthbrain_dataset.get_train_loader()
    elif args.train_dataset == "gigamed+synthbrain+randomanistropy":
        transform = tio.Compose(
            [
                tio.RandomAnisotropy(downsampling=(1, 4)),
            ]
        )
        gigamed_synthbrain_dataset = gigamed.GigaMedSynthBrain(
            args.batch_size,
            args.num_workers,
            include_seg=args.seg_available,
            transform=transform,
        )
        train_loader = gigamed_synthbrain_dataset.get_train_loader()
    else:
        raise ValueError('Invalid train datasets "{}"'.format(args.train_dataset))

    # Eval dataset
    if args.test_dataset == "ixi":
        _, id_eval_loaders = ixi.get_loaders()

        return train_loader, {
            "id": id_eval_loaders,
        }
    elif args.test_dataset == "gigamed":
        gigamed_proc_dataset = gigamed.GigaMed(
            args.batch_size, args.num_workers, include_seg=args.seg_available
        )
        gigamed_raw_dataset = gigamed.GigaMed(
            args.batch_size,
            args.num_workers,
            include_seg=args.seg_available,
            use_raw_data=True,
        )
        id_eval_loaders = gigamed_proc_dataset.get_eval_loaders(id=True)
        id_eval_lesion_loaders = gigamed_proc_dataset.get_eval_lesion_loaders(id=True)
        id_eval_group_loaders = gigamed_proc_dataset.get_eval_group_loaders(id=True)
        id_eval_long_loaders = gigamed_proc_dataset.get_eval_longitudinal_loaders(
            id=True
        )
        ood_eval_loaders = gigamed_proc_dataset.get_eval_loaders(id=False)
        ood_eval_lesion_loaders = gigamed_proc_dataset.get_eval_lesion_loaders(id=False)
        ood_eval_group_loaders = gigamed_proc_dataset.get_eval_group_loaders(id=False)
        ood_eval_long_loaders = gigamed_proc_dataset.get_eval_longitudinal_loaders(
            id=False
        )
        # raw_eval_loaders = gigamed_raw_dataset.get_eval_loaders(id=False)
        # raw_eval_lesion_loaders = gigamed_raw_dataset.get_eval_lesion_loaders(id=False)
        # raw_eval_group_loaders = gigamed_raw_dataset.get_eval_group_loaders(id=False)
        # raw_eval_long_loaders = gigamed_proc_dataset.get_eval_longitudinal_loaders(
        #     id=False
        # )

        return {
            "pretrain": pretrain_loader,
            "train": train_loader,
            "eval": {
                "id": id_eval_loaders,
                "id_lesion": id_eval_lesion_loaders,
                "id_group": id_eval_group_loaders,
                "id_long": id_eval_long_loaders,
                "ood": ood_eval_loaders,
                "ood_lesion": ood_eval_lesion_loaders,
                "ood_group": ood_eval_group_loaders,
                "ood_long": ood_eval_long_loaders,
                # "raw": raw_eval_loaders,
                # "raw_lesion": raw_eval_lesion_loaders,
                # "raw_group": raw_eval_group_loaders,
                # "raw_long": raw_eval_long_loaders,
            },
            "ref_subject": ref_subject,
        }
    else:
        raise ValueError('Invalid test dataset "{}"'.format(args.test_dataset))

    # if args.kpconsistency_coeff > 0:
    #     assert (
    #         len(moving_datasets) > 1
    #     ), "Need more than one modality to compute keypoint consistency loss"
    #     assert all(
    #         [len(fd) == len(md) for fd, md in zip(fixed_datasets, moving_datasets)]
    #     ), "Must have same number of subjects for fixed and moving datasets"


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
            network = UNet(
                args.dim,
                1,
                args.num_keypoints,
            )
        elif args.backbone == "truncatedunet":
            network = TruncatedUNet(
                args.dim,
                1,
                args.num_keypoints,
                args.num_truncated_layers_for_truncatedunet,
            )
        elif args.backbone == "se3cnn":
            network = RXFM_Net(1, args.num_keypoints, norm_type=args.norm_type)
        else:
            raise ValueError('Invalid keypoint extractor "{}"'.format(args.backbone))
        network = torch.nn.DataParallel(network)

        # Keypoint model
        registration_model = KeyMorph(
            network,
            args.num_keypoints,
            args.dim,
            use_amp=args.use_amp,
            max_train_keypoints=args.max_train_keypoints,
            weight_keypoints=args.weighted_kp_align,
        )
        registration_model.to(args.device)
        utils.summary(registration_model)
    elif args.registration_model == "itkelastix":
        registration_model = ITKElastix()
    else:
        raise ValueError(
            'Invalid registration model "{}"'.format(args.registration_model)
        )
    return registration_model


def run_train(train_loader, registration_model, optimizer, args):
    """Train for one epoch.

    Args:
        fixed_loaders: list of Dataloaders for fixed images
        moving_loaders: list of Dataloaders for moving images
        network: keypoint extractor
        optimizer: Pytorch optimizer
        args: Other script arguments
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    registration_model.train()

    res = []

    for step_idx, (subjects, family_name) in enumerate(train_loader):
        fixed, moving = subjects
        family_name = family_name[0]
        if step_idx == args.steps_per_epoch:
            break

        # Get training parameters given family name
        train_params = gigamed.GIGAMED_FAMILY_TRAIN_PARAMS
        transform_type = train_params[family_name]["transform_type"]
        loss_fn = train_params[family_name]["loss_fn"]
        max_random_params = train_params[family_name]["max_random_params"]
        transform_type = train_params[family_name]["transform_type"]

        # Get images and segmentations from TorchIO subject
        img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
        if args.seg_available:
            seg_f, seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]
            # One-hot encode segmentations
            if args.max_train_seg_channels is not None:
                seg_f, seg_m = utils.one_hot_subsampled_pair(
                    seg_f.long(), seg_m.long(), args.max_train_seg_channels
                )
            else:
                seg_f = utils.one_hot(seg_f.long())
                seg_m = utils.one_hot(seg_m.long())

        assert (
            img_f.shape == img_m.shape
        ), f"Fixed and moving images must have same shape:\n --> {fixed['img']['path']}: {img_f.shape}\n --> {moving['img']['path']}: {img_m.shape}"
        assert (
            img_f.shape[1] == 1
        ), f"Fixed image must have 1 channel:\n --> {fixed['img']['path']}: {img_f.shape}"
        assert (
            img_m.shape[1] == 1
        ), f"Moving image must have 1 channel:\n--> {moving['img']['path']}: {img_m.shape}"

        # Move to device
        img_f = img_f.float().to(args.device)
        img_m = img_m.float().to(args.device)
        if args.seg_available:
            seg_f = seg_f.float().to(args.device)
            seg_m = seg_m.float().to(args.device)

        # Explicitly augment moving image
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1
        if args.seg_available:
            img_m, seg_m = random_affine_augment(
                img_m,
                seg=seg_m,
                max_random_params=max_random_params,
                scale_params=scale_augment,
            )
        else:
            img_m = random_affine_augment(
                img_m, max_random_params=max_random_params, scale_params=scale_augment
            )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            registration_results = registration_model(
                img_f,
                img_m,
                transform_type=transform_type,
                return_aligned_points=args.visualize,
            )[0]
            grid = registration_results["grid"]
            align_type = registration_results["align_type"]
            tps_lmbda = registration_results["tps_lmbda"]
            points_m = registration_results["points_m"]
            points_f = registration_results["points_f"]
            if "points_a" in registration_results:
                points_a = registration_results["points_a"]
            points_weights = registration_results["points_weights"]

            img_a = align_img(grid, img_m)
            if args.seg_available:
                seg_a = align_img(
                    grid, seg_m
                )  # Note we use bilinear interpolation here so that backprop works

            # Compute metrics
            metrics = {}
            metrics["scale_augment"] = scale_augment
            metrics["mse"] = loss_ops.MSELoss()(img_f, img_a)
            if args.seg_available:
                metrics["softdiceloss"] = loss_ops.DiceLoss()(seg_a, seg_f)
                metrics["softdice"] = 1 - metrics["softdiceloss"]

            # Compute loss
            if loss_fn == "mse":
                loss = metrics["mse"]
            elif loss_fn == "dice":
                loss = metrics["softdiceloss"]
            else:
                raise ValueError('Invalid loss function "{}"'.format(loss_fn))
            metrics["loss"] = loss

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Keypoint consistency loss
        # if args.kpconsistency_coeff > 0:
        #     mods = np.random.choice(len(moving_loaders), size=2, replace=False)
        #     rand_subject = np.random.randint(0, len(moving_iter))
        #     sub1 = moving_loaders[mods[0]].dataset[rand_subject]
        #     sub2 = moving_loaders[mods[1]].dataset[rand_subject]

        #     sub1 = sub1["img"][tio.DATA].float().to(args.device).unsqueeze(0)
        #     sub2 = sub2["img"][tio.DATA].float().to(args.device).unsqueeze(0)
        #     sub1, sub2 = random_affine_augment_pair(
        #         sub1, sub2, scale_params=scale_augment
        #     )

        #     optimizer.zero_grad()
        #     with torch.set_grad_enabled(True):
        #         points1, points2 = registration_model.extract_keypoints_step(sub1, sub2)

        #     kploss = args.kpconsistency_coeff * loss_ops.MSELoss()(points1, points2)
        #     kploss.backward()
        #     optimizer.step()
        #     metrics["kploss"] = kploss

        end_time = time.time()
        metrics["epoch_time"] = end_time - start_time

        # Convert metrics to numpy
        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

        if args.debug_mode:
            print("\nDebugging info:")
            print(f"-> Family name: {family_name}")
            print(f"-> Alignment: {align_type} ")
            print(f"-> Max random params: {max_random_params} ")
            print(f"-> TPS lambda: {tps_lmbda} ")
            print(f"-> Loss: {loss_fn}")
            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
            print(
                f"-> Point shapes: {points_f.shape}, {points_m.shape}, {points_a.shape}"
            )
            print(f"-> Point weights: {points_weights}")
            print(f"-> Float16: {args.use_amp}")
            if args.seg_available:
                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

        if args.visualize:
            if args.dim == 2:
                show_warped(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                )
                if args.seg_available:
                    show_warped(
                        seg_m[0, 0].cpu().detach().numpy(),
                        seg_f[0, 0].cpu().detach().numpy(),
                        seg_a[0, 0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(),
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                    )
            else:
                show_warped_vol(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )
                if args.seg_available:
                    show_warped_vol(
                        seg_m.argmax(1)[0].cpu().detach().numpy(),
                        seg_f.argmax(1)[0].cpu().detach().numpy(),
                        seg_a.argmax(1)[0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(),
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                        save_path=(
                            None
                            if args.debug_mode
                            else os.path.join(
                                args.model_img_dir, f"seg_{args.curr_epoch}.png"
                            )
                        ),
                    )

    return utils.aggregate_dicts(res)


def run_pretrain(loader, random_points, keymorph_model, optimizer, args):
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    keymorph_model.train()

    res = []

    random_points = random_points.to(args.device)
    for step_idx, (subject, _) in enumerate(loader):
        if step_idx == args.steps_per_epoch:
            break
        x_fixed = subject["img"][tio.DATA].float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1
        x_moving, tgt_points = random_affine_augment(
            x_fixed, points=random_points, scale_params=scale_augment
        )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                pred_points = keymorph_model.get_keypoints(x_moving)
                pred_points = pred_points.view(-1, args.num_keypoints, args.dim)
                loss = F.mse_loss(tgt_points, pred_points)

        # Perform backward pass
        if args.use_amp:
            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = {}
        metrics["scale_augment"] = scale_augment
        metrics["loss"] = loss.cpu().detach().numpy()
        end_time = time.time()
        metrics["epoch_time"] = end_time - start_time
        res.append(metrics)

        if args.visualize and step_idx == 0:
            if args.dim == 2:
                show_warped(
                    x_moving[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    tgt_points[0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                    pred_points[0].cpu().detach().numpy(),
                )
            else:
                show_warped_vol(
                    x_moving[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    tgt_points[0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                    pred_points[0].cpu().detach().numpy(),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )

    return utils.aggregate_dicts(res)


def run_eval(
    loaders,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_aligns,
    args,
):

    def _build_metric_dict(names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            for a in list_of_eval_augs:
                for k in list_of_eval_aligns:
                    for n in names:
                        n1, n2 = n
                        list_of_all_test.append(f"{m}:{n1}:{n2}:{a}:{k}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    test_metrics = _build_metric_dict(list_of_eval_names)
    for dataset_name in list_of_eval_names:
        for aug in list_of_eval_augs:
            mod1, mod2 = utils.parse_test_mod(dataset_name)
            param = utils.parse_test_aug(aug)
            for i, fixed in enumerate(loaders[mod1]):
                if args.early_stop_eval_subjects and i == args.early_stop_eval_subjects:
                    break
                for j, moving in enumerate(loaders[mod2]):
                    if (
                        args.early_stop_eval_subjects
                        and j == args.early_stop_eval_subjects
                    ):
                        break
                    print(
                        f"Running test: subject id {i}->{j}, mod {mod1}->{mod2}, aug {aug}"
                    )
                    img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
                    if args.seg_available:
                        seg_f, seg_m = (
                            fixed["seg"][tio.DATA],
                            moving["seg"][tio.DATA],
                        )
                        # One-hot encode segmentations
                        seg_f = utils.one_hot_eval(seg_f)
                        seg_m = utils.one_hot_eval(seg_m)

                    # Move to device
                    img_f = img_f.float().to(args.device)
                    img_m = img_m.float().to(args.device)
                    if args.seg_available:
                        seg_f = seg_f.float().to(args.device)
                        seg_m = seg_m.float().to(args.device)

                    # Explicitly augment moving image
                    if args.seg_available:
                        img_m, seg_m = affine_augment(img_m, param, seg=seg_m)
                    else:
                        img_m = affine_augment(img_m, param)

                    with torch.set_grad_enabled(False):
                        registration_results = registration_model(
                            img_f,
                            img_m,
                            transform_type=list_of_eval_aligns,
                            return_aligned_points=True,
                        )

                    for res_dict in registration_results:
                        align_type_str = res_dict["align_type"]
                        if "img_a" in res_dict:
                            img_a = res_dict["img_a"]
                        elif "grid" in res_dict:
                            grid = res_dict["grid"]
                            img_a = align_img(grid, img_m)
                        else:
                            raise ValueError("No way to get aligned image")
                        if args.seg_available:
                            if "seg_a" in res_dict:
                                seg_a = res_dict["seg_a"]
                            elif "grid" in res_dict:
                                grid = res_dict["grid"]
                                seg_a = align_img(grid, seg_m)
                            else:
                                raise ValueError("No way to get aligned segmentation")

                        points_m = (
                            res_dict["points_m"] if "points_m" in res_dict else None
                        )
                        points_f = (
                            res_dict["points_f"] if "points_f" in res_dict else None
                        )
                        points_a = (
                            res_dict["points_a"] if "points_a" in res_dict else None
                        )
                        points_weights = (
                            res_dict["points_weights"]
                            if "points_weights" in res_dict
                            else None
                        )

                        if args.visualize:
                            if args.dim == 2:
                                show_warped(
                                    img_m[0, 0].cpu().detach().numpy(),
                                    img_f[0, 0].cpu().detach().numpy(),
                                    img_a[0, 0].cpu().detach().numpy(),
                                    points_m[0].cpu().detach().numpy(),
                                    points_f[0].cpu().detach().numpy(),
                                    points_a[0].cpu().detach().numpy(),
                                    weights=points_weights,
                                )
                                if args.seg_available:
                                    show_warped(
                                        seg_m[0, 0].cpu().detach().numpy(),
                                        seg_f[0, 0].cpu().detach().numpy(),
                                        seg_a[0, 0].cpu().detach().numpy(),
                                        points_m.cpu().detach().numpy(),
                                        points_f.cpu().detach().numpy(),
                                        points_a.cpu().detach().numpy(),
                                        weights=points_weights,
                                    )
                            else:
                                show_warped_vol(
                                    img_m[0, 0].cpu().detach().numpy(),
                                    img_f[0, 0].cpu().detach().numpy(),
                                    img_a[0, 0].cpu().detach().numpy(),
                                    points_m[0].cpu().detach().numpy(),
                                    points_f[0].cpu().detach().numpy(),
                                    points_a[0].cpu().detach().numpy(),
                                    weights=points_weights,
                                    save_path=None,
                                )
                                if args.seg_available:
                                    show_warped_vol(
                                        seg_m.argmax(1)[0].cpu().detach().numpy(),
                                        seg_f.argmax(1)[0].cpu().detach().numpy(),
                                        seg_a.argmax(1)[0].cpu().detach().numpy(),
                                        points_m[0].cpu().detach().numpy(),
                                        points_f[0].cpu().detach().numpy(),
                                        points_a[0].cpu().detach().numpy(),
                                        weights=points_weights,
                                        save_path=None,
                                    )

                        # Compute metrics
                        metrics = {}
                        if args.seg_available:
                            # Always compute hard dice once ahead of time
                            dice = loss_ops.DiceLoss(hard=True)(
                                seg_a, seg_f, ign_first_ch=True
                            )
                            dice_total = 1 - dice[0].item()
                            dice_roi = (1 - dice[1].cpu().detach().numpy()).tolist()
                        for m in list_of_eval_metrics:
                            if m == "mse":
                                metrics["mse"] = loss_ops.MSELoss()(img_f, img_a).item()
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["mse"])
                            elif m == "softdice":
                                assert args.seg_available
                                metrics["softdiceloss"] = loss_ops.DiceLoss()(
                                    seg_a, seg_f
                                ).item()
                                metrics["softdice"] = 1 - metrics["softdiceloss"]
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["softdice"])
                            elif m == "harddice":
                                assert args.seg_available
                                metrics["harddice"] = dice_total
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["harddice"])
                            elif m == "harddiceroi":
                                # Don't save roi into metric dict, since it's a list
                                assert args.seg_available
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(dice_roi)
                            elif m == "hausd":
                                assert args.seg_available and args.dim == 3
                                metrics["hausd"] = loss_ops.hausdorff_distance(
                                    seg_a, seg_f
                                )
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["hausd"])
                            elif m == "jdstd":
                                assert args.dim == 3
                                grid_permute = grid.permute(0, 4, 1, 2, 3)
                                metrics["jdstd"] = loss_ops.jdstd(grid_permute)
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["jdstd"])
                            elif m == "jdlessthan0":
                                assert args.dim == 3
                                grid_permute = grid.permute(0, 4, 1, 2, 3)
                                metrics["jdlessthan0"] = loss_ops.jdlessthan0(
                                    grid_permute, as_percentage=True
                                )
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["jdlessthan0"])
                            else:
                                raise ValueError('Invalid metric "{}"'.format(m))

                        if args.save_preds and not args.debug_mode:
                            assert args.batch_size == 1  # TODO: fix this
                            img_f_path = args.model_eval_dir / f"img_f_{i}-{mod1}.npy"
                            img_m_path = (
                                args.model_eval_dir / f"img_m_{j}-{mod2}-{aug}.npy"
                            )
                            img_a_path = (
                                args.model_eval_dir
                                / f"img_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                            )
                            points_f_path = (
                                args.model_eval_dir / f"points_f_{i}-{mod1}.npy"
                            )
                            points_m_path = (
                                args.model_eval_dir / f"points_m_{j}-{mod2}-{aug}.npy"
                            )
                            points_a_path = (
                                args.model_eval_dir
                                / f"points_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                            )
                            grid_path = (
                                args.model_eval_dir
                                / f"grid_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                            )
                            print(
                                "Saving:\n{}\n{}\n{}\n{}\n".format(
                                    img_f_path,
                                    img_m_path,
                                    img_a_path,
                                    grid_path,
                                )
                            )
                            np.save(img_f_path, img_f[0].cpu().detach().numpy())
                            np.save(img_m_path, img_m[0].cpu().detach().numpy())
                            np.save(img_a_path, img_a[0].cpu().detach().numpy())
                            np.save(grid_path, grid[0].cpu().detach().numpy())
                            if points_f is not None:
                                np.save(
                                    points_f_path,
                                    points_f[0].cpu().detach().numpy(),
                                )
                                np.save(
                                    points_m_path,
                                    points_m[0].cpu().detach().numpy(),
                                )
                                np.save(
                                    points_a_path,
                                    points_a[0].cpu().detach().numpy(),
                                )

                            if args.seg_available:
                                seg_f_path = (
                                    args.model_eval_dir / f"seg_f_{i}-{mod1}.npy"
                                )
                                seg_m_path = (
                                    args.model_eval_dir / f"seg_m_{j}-{mod2}-{aug}.npy"
                                )
                                seg_a_path = (
                                    args.model_eval_dir
                                    / f"seg_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                                )
                                np.save(
                                    seg_f_path,
                                    np.argmax(seg_f.cpu().detach().numpy(), axis=1),
                                )
                                np.save(
                                    seg_m_path,
                                    np.argmax(seg_m.cpu().detach().numpy(), axis=1),
                                )
                                np.save(
                                    seg_a_path,
                                    np.argmax(seg_a.cpu().detach().numpy(), axis=1),
                                )

                        if args.debug_mode:
                            print("\nDebugging info:")
                            print(f"-> Alignment: {align_type_str} ")
                            print(f"-> Max random params: {param} ")
                            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
                            if points_f is not None:
                                print(
                                    f"-> Point shapes: {points_f.shape}, {points_m.shape}"
                                )
                                print(f"-> Point weights: {points_weights}")
                            print(f"-> Float16: {args.use_amp}")
                            if args.seg_available:
                                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

                        print("\nMetrics:")
                        for metric_name, metric in metrics.items():
                            if not isinstance(metric, list):
                                print(f"-> {metric_name}: {metric:.5f}")
    return test_metrics


def run_groupwise_eval(
    group_loader,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_kp_aligns,
    args,
):
    def _build_metric_dict(dataset_names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            for a in list_of_eval_augs:
                for k in list_of_eval_kp_aligns:
                    for n in dataset_names:
                        list_of_all_test.append(f"{m}:{n}:{a}:{k}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    def _construct_template(list_of_imgs, aggr="mean"):
        if aggr == "mean":
            template = torch.mean(torch.stack(list_of_imgs), dim=0)
        elif aggr == "majority":
            template, _ = torch.mode(torch.stack(list_of_imgs), dim=0)
        else:
            raise ValueError('Invalid aggr "{}"'.format(aggr))
        return template

    def _repeat_to_N(batch_of_imgs, N=4):
        imgs_to_add = N - batch_of_imgs.shape[0]
        # Create additional volumes by duplicating the first volume
        first_img = batch_of_imgs[
            0:1
        ]  # Extract the first volume (keep its first axis to maintain the 4D shape)
        additional_imgs = first_img.repeat(
            imgs_to_add, 1, 1, 1, 1
        )  # Repeat the first image along the batch axis
        # Concatenate the original tensor with the additional volumes
        return torch.cat([batch_of_imgs, additional_imgs], dim=0)

    test_metrics = _build_metric_dict(list_of_eval_names)
    for dataset_name in list_of_eval_names:
        for aug in list_of_eval_augs:
            aug_params = utils.parse_test_aug(aug)
            for i, group in enumerate(group_loader[dataset_name]):
                if args.early_stop_eval_subjects and i == args.early_stop_eval_subjects:
                    break
                print(
                    f"Running groupwise test: group id {i}, dataset {dataset_name}, aug {aug}"
                )
                list_of_img_paths = [subject["img"]["path"][0] for subject in group]
                if args.seg_available:
                    list_of_seg_paths = [subject["seg"]["path"][0] for subject in group]

                # Load images
                group_imgs_m = []
                if args.seg_available:
                    group_segs_m = []

                for i in range(len(list_of_img_paths)):
                    img_m = torch.tensor(nib.load(list_of_img_paths[i]).get_fdata())[
                        None, None
                    ].float()
                    if args.seg_available:
                        seg_m = torch.tensor(nib.load(list_of_seg_paths[i]).get_fdata())
                        # One-hot encode segmentations
                        seg_m = utils.one_hot_eval(seg_m)

                    # For groupwise, we randomly affine augment all images
                    if aug_params is not None:
                        if args.seg_available:
                            img_m, seg_m = random_affine_augment(
                                img_m, max_random_params=aug_params, seg=seg_m
                            )
                        else:
                            img_m = random_affine_augment(
                                img_m, max_random_params=aug_params
                            )
                    group_imgs_m.append(img_m)
                    if args.seg_available:
                        group_segs_m.append(seg_m)

                group_imgs_m = torch.cat(group_imgs_m, dim=0)
                # If group size < 4, duplicate the first image to make 4.
                # This is because some groupwise registration packages require
                # at least 4 images.
                if group_imgs_m.shape[0] < 4:
                    group_imgs_m = _repeat_to_N(group_imgs_m, N=4)
                if args.seg_available:
                    group_segs_m = torch.cat(group_segs_m, dim=0)
                    if group_segs_m.shape[0] < 4:
                        group_segs_m = _repeat_to_N(group_segs_m, N=4)

                with torch.set_grad_enabled(False):
                    registration_results = registration_model.groupwise_register(
                        group_imgs_m,
                        transform_type=list_of_eval_kp_aligns,
                        device=args.device,
                    )

                for res_dict in registration_results:
                    align_type_str = res_dict["align_type"]
                    group_grids = res_dict["grids"]
                    group_imgs_a = []
                    if args.seg_available:
                        group_segs_a = []
                    # Align images and construct template image
                    for i in range(len(group_imgs_m)):
                        img_m = group_imgs_m[i : i + 1].to(args.device)
                        grid = group_grids[i : i + 1].to(args.device)
                        group_imgs_a.append(align_img(grid, img_m))
                        if args.seg_available:
                            seg_m = group_segs_m[i : i + 1].to(args.device)
                            group_segs_a.append(align_img(grid, seg_m))
                    group_imgs_a = torch.cat(group_imgs_a, dim=0)
                    if args.seg_available:
                        group_segs_a = torch.cat(group_segs_a, dim=0)

                    if args.visualize:
                        plot_groupwise_register(
                            [
                                img[0, 128].cpu().detach().numpy()
                                for img in group_imgs_m
                            ],
                            [
                                img[0, 128].cpu().detach().numpy()
                                for img in group_imgs_a
                            ],
                        )
                        if args.seg_available:
                            plot_groupwise_register(
                                [
                                    seg.argmax(0)[128].cpu().detach().numpy()
                                    for seg in group_segs_m
                                ],
                                [
                                    seg.argmax(0)[128].cpu().detach().numpy()
                                    for seg in group_segs_a
                                ],
                            )

                    metrics = {}
                    for m in list_of_eval_metrics:
                        if m == "mse":
                            metrics["mse"] = loss_ops.MSEPairwiseLoss()(
                                group_imgs_a
                            ).item()
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(metrics["mse"])
                        elif m == "softdice":
                            assert args.seg_available
                            metrics["softdiceloss"] = loss_ops.SoftDicePairwiseLoss()(
                                group_segs_a
                            ).item()
                            metrics["softdice"] = 1 - metrics["softdiceloss"]
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(metrics["softdice"])
                        elif m == "harddice":
                            assert args.seg_available
                            metrics["harddice"] = (
                                1 - loss_ops.HardDicePairwiseLoss()(group_segs_a).item()
                            )
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(metrics["harddice"])
                        # elif m == "harddiceroi":
                        #     # Don't save roi into metric dict, since it's a list
                        #     assert args.seg_available
                        #     test_metrics[
                        #         f"{m}:{dataset_name}:{aug}:{align_type_str}"
                        #     ].append(dice_roi)
                        elif m == "hausd":
                            assert args.seg_available and args.dim == 3
                            metrics["hausd"] = loss_ops.HausdorffPairwiseLoss()(
                                group_segs_a
                            )
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(metrics["hausd"])
                        elif m == "jdstd":
                            assert args.dim == 3
                            tot_jdstd = 0
                            for i in range(len(group_grids)):
                                grid = group_grids[i : i + 1]
                                grid_permute = grid.permute(0, 4, 1, 2, 3)
                                tot_jdstd += loss_ops.jdstd(grid_permute)
                            metrics["jdstd"] = tot_jdstd / len(group_grids)
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(metrics["jdstd"])
                        elif m == "jdlessthan0":
                            assert args.dim == 3
                            tot_jdlessthan0 = 0
                            for i in range(len(group_grids)):
                                grid = group_grids[i : i + 1]
                                grid_permute = grid.permute(0, 4, 1, 2, 3)
                                tot_jdlessthan0 += loss_ops.jdlessthan0(
                                    grid_permute, as_percentage=True
                                )
                            metrics["jdlessthan0"] = tot_jdlessthan0 / len(group_grids)
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(metrics["jdlessthan0"])
                        else:
                            raise ValueError('Invalid metric "{}"'.format(m))

                    if args.debug_mode:
                        print("\nDebugging info:")
                        print(f"-> Alignment: {align_type_str} ")
                        print(f"-> Max random params: {aug_params} ")
                        print(f"-> Group size: {len(group_grids)}")
                        print(f"-> Float16: {args.use_amp}")
                    # if args.save_preds and not args.debug_mode:
                    #     assert args.batch_size == 1  # TODO: fix this
                    #     img_f_path = args.model_eval_dir / f"img_f_{i}-{mod1}.npy"
                    #     img_m_path = args.model_eval_dir / f"img_m_{j}-{mod2}-{aug}.npy"
                    #     img_a_path = (
                    #         args.model_eval_dir
                    #         / f"img_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                    #     )
                    #     points_f_path = args.model_eval_dir / f"points_f_{i}-{mod1}.npy"
                    #     points_m_path = args.model_eval_dir / f"points_m_{j}-{mod2}-{aug}.npy"
                    #     points_a_path = (
                    #         args.model_eval_dir
                    #         / f"points_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                    #     )
                    #     grid_path = (
                    #         args.model_eval_dir
                    #         / f"grid_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                    #     )
                    #     print(
                    #         "Saving:\n{}\n{}\n{}\n{}\n".format(
                    #             img_f_path,
                    #             img_m_path,
                    #             img_a_path,
                    #             grid_path,
                    #         )
                    #     )
                    #     np.save(img_f_path, img_f[0].cpu().detach().numpy())
                    #     np.save(img_m_path, img_m[0].cpu().detach().numpy())
                    #     np.save(img_a_path, img_a[0].cpu().detach().numpy())
                    #     np.save(
                    #         points_f_path,
                    #         points_f[0].cpu().detach().numpy(),
                    #     )
                    #     np.save(
                    #         points_m_path,
                    #         points_m[0].cpu().detach().numpy(),
                    #     )
                    #     np.save(
                    #         points_a_path,
                    #         points_a[0].cpu().detach().numpy(),
                    #     )
                    #     np.save(grid_path, grid[0].cpu().detach().numpy())

                    #     if seg_available:
                    #         seg_f_path = args.model_eval_dir / f"seg_f_{i}-{mod1}.npy"
                    #         seg_m_path = args.model_eval_dir / f"seg_m_{j}-{mod2}-{aug}.npy"
                    #         seg_a_path = (
                    #             args.model_eval_dir
                    #             / f"seg_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                    #         )
                    #         np.save(
                    #             seg_f_path,
                    #             np.argmax(seg_f.cpu().detach().numpy(), axis=1),
                    #         )
                    #         np.save(
                    #             seg_m_path,
                    #             np.argmax(seg_m.cpu().detach().numpy(), axis=1),
                    #         )
                    #         np.save(
                    #             seg_a_path,
                    #             np.argmax(seg_a.cpu().detach().numpy(), axis=1),
                    #         )

                    print("\nMetrics:")
                    for metric_name, metric in metrics.items():
                        if not isinstance(metric, list):
                            print(f"-> {metric_name}: {metric:.5f}")

    return test_metrics


def main():
    args = parse_args()
    if args.debug_mode:
        args.steps_per_epoch = 3
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

        if args.save_preds:
            args.model_eval_dir = args.model_dir / "eval"
            if not os.path.exists(args.model_eval_dir) and not args.debug_mode:
                os.makedirs(args.model_eval_dir)

        if args.test_dataset == "ixi":
            list_of_eval_names = ixi_hps.EVAL_NAMES
            list_of_eval_lesion_names = ixi_hps.EVAL_LESION_NAMES
            list_of_eval_group_names = ixi_hps.EVAL_GROUP_NAMES
            list_of_eval_long_names = ixi_hps.EVAL_LONG_NAMES
        elif args.test_dataset == "gigamed":
            list_of_eval_metrics = gigamed_hps.EVAL_METRICS
            list_of_eval_augs = gigamed_hps.EVAL_AUGS
            if args.registration_model == "keymorph":
                list_of_eval_aligns = gigamed_hps.EVAL_KP_ALIGNS
                list_of_eval_group_aligns = gigamed_hps.EVAL_GROUP_KP_ALIGNS
                list_of_eval_long_aligns = gigamed_hps.EVAL_LONG_KP_ALIGNS
            elif args.registration_model == "itkelastix":
                list_of_eval_aligns = gigamed_hps.EVAL_ITKELASTIX_ALIGNS
                list_of_eval_group_aligns = gigamed_hps.EVAL_GROUP_ITKELASTIX_ALIGNS
                list_of_eval_long_aligns = gigamed_hps.EVAL_LONG_ITKELASTIX_ALIGNS

            list_of_eval_names = gigamed_hps.EVAL_NAMES
            list_of_eval_lesion_names = gigamed_hps.EVAL_LESION_NAMES
            list_of_eval_group_names = gigamed_hps.EVAL_GROUP_NAMES
            list_of_eval_long_names = gigamed_hps.EVAL_LONG_NAMES

        # for dist in ["id", "ood", "raw"]:
        for dist in ["id", "ood"]:
            print("Perform eval on", dist)
            json_path = args.model_result_dir / f"summary_{dist}.json"
            if not os.path.exists(json_path):
                eval_metrics = run_eval(
                    loaders["eval"][dist],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_names[dist],
                    list_of_eval_augs,
                    list_of_eval_aligns,
                    args,
                )
                if not args.debug_mode:
                    save_summary_json(eval_metrics, json_path)
            else:
                print("Skipping eval on", json_path)

            # Lesions
            json_path = args.model_result_dir / f"summary_lesion_{dist}.json"
            if not os.path.exists(json_path) and list_of_eval_lesion_names is not None:
                lesion_metrics = run_eval(
                    loaders["eval"][f"{dist}_lesion"],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_lesion_names[dist],
                    list_of_eval_augs,
                    list_of_eval_aligns,
                    args,
                )
                if not args.debug_mode:
                    save_summary_json(
                        lesion_metrics,
                        json_path,
                    )
            else:
                print("Skipping eval on", json_path)

            # Groupwise
            json_path = args.model_result_dir / f"summary_group_{dist}.json"
            if not os.path.exists(json_path) and list_of_eval_group_names is not None:
                group_metrics = run_groupwise_eval(
                    loaders["eval"][f"{dist}_group"],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_group_names[dist],
                    list_of_eval_augs,
                    list_of_eval_group_aligns,
                    args,
                )
                if not args.debug_mode:
                    save_summary_json(group_metrics, json_path)
            else:
                print("Skipping eval on", json_path)

            # Longitudinal
            json_path = args.model_result_dir / f"summary_long_{dist}.json"
            if not os.path.exists(json_path) and list_of_eval_long_names is not None:
                long_metrics = run_groupwise_eval(
                    loaders["eval"][f"{dist}_long"],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_long_names[dist],
                    list_of_eval_augs,
                    list_of_eval_long_aligns,
                    args,
                )
                if not args.debug_mode:
                    save_summary_json(long_metrics, json_path)
            else:
                print("Skipping eval on", json_path)

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
