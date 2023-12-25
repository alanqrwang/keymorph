import os
from pprint import pprint
import torch
import time
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path
import wandb
import torchio as tio
from scipy.stats import loguniform
import matplotlib.pyplot as plt
import json
from copy import deepcopy

from keymorph import loss_ops
from keymorph.net import ConvNet, UNet, RXFM_Net
from keymorph.model import KeyMorph
from keymorph import utils
from keymorph.utils import (
    ParseKwargs,
    initialize_wandb,
    str_or_float,
    align_img,
    save_summary_json,
)
from keymorph.data import ixi, gigamed, synthbrain
from keymorph.augmentation import (
    affine_augment,
    random_affine_augment,
    random_affine_augment_pair,
)
from keymorph.cm_plotter import show_warped, show_warped_vol
from keymorph.keypoint_aligners import RigidKeypointAligner, AffineKeypointAligner, TPS


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

    parser.add_argument("--save_preds", action="store_true", help="Perform evaluation")

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
    parser.add_argument(
        "--max_train_keypoints",
        type=int,
        default=64,
        help="Number of keypoints to subsample TPS, to save memory",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="conv_com",
        choices=["conv", "unet", "se3cnn", "se3cnn2"],
        help="Keypoint extractor module to use",
    )

    parser.add_argument(
        "--kp_layer",
        type=str,
        default="com",
        choices=["com", "linear"],
        help="Keypoint layer module to use",
    )

    parser.add_argument(
        "--tps_lmbda", type=str_or_float, default=None, help="TPS lambda value"
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

    parser.add_argument("--loss_fn", type=str, default="mse")

    parser.add_argument("--epochs", type=int, default=2000, help="Training Epochs")

    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=32,
        help="Number of gradient steps per epoch",
    )

    parser.add_argument("--eval", action="store_true", help="Perform evaluation")

    parser.add_argument(
        "--affine_slope",
        type=int,
        default=1,
        help="Constant to control how slow to increase augmentation. If negative, disabled.",
    )

    # Miscellaneous
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


def _get_tps_lmbda(num_samples, tps_lmbda):
    if tps_lmbda == "uniform":
        lmbda = torch.rand(num_samples) * 10
    elif tps_lmbda == "lognormal":
        lmbda = torch.tensor(np.random.lognormal(size=num_samples))
    elif tps_lmbda == "loguniform":
        a, b = 1e-6, 10
        lmbda = torch.tensor(loguniform.rvs(a, b, size=num_samples))
    else:
        lmbda = torch.tensor(tps_lmbda).repeat(num_samples)
    return lmbda


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

    for step_idx, (fixed, moving, task_type) in enumerate(train_loader):
        task_type = task_type[0]
        if step_idx == args.steps_per_epoch:
            break

        if task_type == "same_sub_same_mod":
            align_type = "rigid"
            args.loss_fn = "mse"
            max_random_params = (0, 0.15, 3.1416, 0)
        elif task_type == "diff_sub_same_mod":
            align_type = "tps"
            args.loss_fn = "mse"
            max_random_params = (0.2, 0.15, 3.1416, 0.1)
        elif task_type == "synthbrain":
            align_type = "tps"
            args.loss_fn = "dice"
            max_random_params = (0.2, 0.15, 3.1416, 0.1)
        elif task_type == "same_sub_diff_mod":
            raise NotImplementedError()
        elif task_type == "diff_sub_diff_mod":
            raise NotImplementedError()
        else:
            raise ValueError('Invalid task_type "{}"'.format(task_type))

        # Get images and segmentations from TorchIO subject
        img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
        if "seg" in fixed and "seg" in moving:
            seg_available = True
            seg_f, seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]
        else:
            seg_available = False

        # Move to device
        img_f = img_f.float().to(args.device)
        img_m = img_m.float().to(args.device)
        if seg_available:
            seg_f = seg_f.float().to(args.device)
            seg_m = seg_m.float().to(args.device)

        # Explicitly augment moving image
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1
        if seg_available:
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

        lmbda = _get_tps_lmbda(len(img_f), args.tps_lmbda).to(args.device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            (
                grid,
                points_f,
                points_m,
                points_a,
                weights,
            ) = registration_model(
                img_f,
                img_m,
                lmbda,
                align_type=align_type,
                return_aligned_points=True,
                return_weights=True,
            )
            img_a = align_img(grid, img_m)
            if seg_available:
                seg_a = align_img(grid, seg_m)

            # Compute metrics
            metrics = {}
            metrics["scale_augment"] = scale_augment
            metrics["mse"] = loss_ops.MSELoss()(img_f, img_a)
            # metrics["lesion_penalty"] = loss_ops.LesionPenalty(
            #     weights, points_f, points_m, lesion_mask_f, lesion_mask_m
            # )
            if seg_available:
                metrics["softdiceloss"] = loss_ops.DiceLoss()(seg_a, seg_f)
                metrics["softdice"] = 1 - metrics["softdiceloss"]

            # Compute loss
            if args.loss_fn == "mse":
                loss = (
                    metrics["mse"]
                    # + args.lesion_penalty_coeff * metrics["lesion_penalty"]
                )
            elif args.loss_fn == "dice":
                loss = (
                    metrics["softdiceloss"]
                    # + args.lesion_penalty_coeff * metrics["lesion_penalty"]
                )
            else:
                raise ValueError('Invalid loss function "{}"'.format(args.loss_fn))
            metrics["loss"] = loss

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

        # Keypoint consistency loss
        if args.kpconsistency_coeff > 0:
            mods = np.random.choice(len(moving_loaders), size=2, replace=False)
            rand_subject = np.random.randint(0, len(moving_iter))
            sub1 = moving_loaders[mods[0]].dataset[rand_subject]
            sub2 = moving_loaders[mods[1]].dataset[rand_subject]

            sub1 = sub1["img"][tio.DATA].float().to(args.device).unsqueeze(0)
            sub2 = sub2["img"][tio.DATA].float().to(args.device).unsqueeze(0)
            sub1, sub2 = random_affine_augment_pair(
                sub1, sub2, scale_params=scale_augment
            )

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                points1, points2 = registration_model.extract_keypoints_step(sub1, sub2)

            kploss = args.kpconsistency_coeff * loss_ops.MSELoss()(points1, points2)
            kploss.backward()
            optimizer.step()
            metrics["kploss"] = kploss

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
            print(f"-> Task type: {task_type}")
            print(f"-> Alignment: {align_type} ")
            print(f"-> Max random params: {max_random_params} ")
            print(f"-> TPS lambda: {lmbda} ")
            print(f"-> Loss: {args.loss_fn}")
            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
            print(f"-> Point shapes: {points_f.shape}, {points_m.shape}")
            print(f"-> Point weights: {weights}")
            print(f"-> Float16: {args.use_amp}")
            if seg_available:
                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

        if args.visualize and step_idx == 0:
            if args.dim == 2:
                show_warped(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                )
                if seg_available:
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
                    save_path=None
                    if args.debug_mode
                    else os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png"),
                )
                if seg_available:
                    show_warped_vol(
                        seg_m.argmax(1)[0].cpu().detach().numpy(),
                        seg_f.argmax(1)[0].cpu().detach().numpy(),
                        seg_a.argmax(1)[0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(),
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                        save_path=None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"seg_{args.curr_epoch}.png"
                        ),
                    )

    return utils.aggregate_dicts(res)


def run_eval(
    loaders,
    registration_model,
    mod1,
    mod2,
    param,
    aug,
    test_metrics,
    list_of_test_metrics,
    list_of_test_kp_aligns,
    args,
):
    for i, fixed in enumerate(loaders[mod1]):
        if args.early_stop_eval_subjects and i > args.early_stop_eval_subjects:
            break
        for j, moving in enumerate(loaders[mod2]):
            if args.early_stop_eval_subjects and j > args.early_stop_eval_subjects:
                break
            print(f"Running test: subject id {i}->{j}, mod {mod1}->{mod2}, aug {aug}")
            img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
            if "seg" in fixed and "seg" in moving:
                seg_available = True
                seg_f, seg_m = (
                    fixed["seg"][tio.DATA],
                    moving["seg"][tio.DATA],
                )
            else:
                seg_available = False
                assert not any(
                    "dice" in m for m in list_of_test_metrics
                ), "Can't compute Dice if no segmentations available"

            # Move to device
            img_f = img_f.float().to(args.device)
            img_m = img_m.float().to(args.device)
            if seg_available:
                seg_f = seg_f.float().to(args.device)
                seg_m = seg_m.float().to(args.device)

            # Explicitly augment moving image
            if seg_available:
                img_m, seg_m = affine_augment(img_m, param, seg=seg_m)
            else:
                img_m = affine_augment(img_m, param)

            with torch.set_grad_enabled(False):
                feat_f = registration_model.backbone(img_f)
                feat_m = registration_model.backbone(img_m)
                points_f = registration_model.keypoint_layer(feat_f)
                points_m = registration_model.keypoint_layer(feat_m)

                if args.weighted_kp_align == "variance":
                    weights = registration_model.weight_by_variance(feat_f, feat_m)
                elif args.weighted_kp_align == "power":
                    weights = registration_model.weight_by_power(feat_f, feat_m)
                else:
                    weights = None

                for _, align_type_str in enumerate(list_of_test_kp_aligns):
                    # Align via keypoints
                    if align_type_str == "rigid":
                        keypoint_aligner = RigidKeypointAligner(args.dim)
                        lmbda = None
                    elif align_type_str == "affine":
                        keypoint_aligner = AffineKeypointAligner(args.dim)
                        lmbda = None
                    elif "tps" in align_type_str:
                        _, tps_lmbda = align_type_str.split("_")
                        keypoint_aligner = TPS(args.dim, num_subgrids=args.num_subgrids)
                        lmbda = _get_tps_lmbda(len(img_f), float(tps_lmbda)).to(
                            args.device
                        )

                    grid = keypoint_aligner.grid_from_points(
                        points_m,
                        points_f,
                        img_f.shape,
                        lmbda=lmbda,
                        weights=weights,
                        compute_on_subgrids=True,
                    )
                    points_a = keypoint_aligner.points_from_points(
                        points_m,
                        points_f,
                        points_m,
                        lmbda=lmbda,
                        weights=weights,
                    )

                    img_a = align_img(grid, img_m)
                    if seg_available:
                        seg_a = align_img(grid, seg_m)

                    if args.visualize:
                        if args.dim == 2:
                            show_warped(
                                img_m[0, 0].cpu().detach().numpy(),
                                img_f[0, 0].cpu().detach().numpy(),
                                img_a[0, 0].cpu().detach().numpy(),
                                points_m[0].cpu().detach().numpy(),
                                points_f[0].cpu().detach().numpy(),
                                points_a[0].cpu().detach().numpy(),
                                weights=weights[0].cpu().detach().numpy(),
                            )
                            if seg_available:
                                show_warped(
                                    seg_m[0, 0].cpu().detach().numpy(),
                                    seg_f[0, 0].cpu().detach().numpy(),
                                    seg_a[0, 0].cpu().detach().numpy(),
                                    points_m[0].cpu().detach().numpy(),
                                    points_f[0].cpu().detach().numpy(),
                                    points_a[0].cpu().detach().numpy(),
                                    weights=weights[0].cpu().detach().numpy(),
                                )
                        else:
                            show_warped_vol(
                                img_m[0, 0].cpu().detach().numpy(),
                                img_f[0, 0].cpu().detach().numpy(),
                                img_a[0, 0].cpu().detach().numpy(),
                                points_m[0].cpu().detach().numpy(),
                                points_f[0].cpu().detach().numpy(),
                                points_a[0].cpu().detach().numpy(),
                                weights=weights[0].cpu().detach().numpy(),
                                save_path=None,
                            )
                            if seg_available:
                                show_warped_vol(
                                    seg_m.argmax(1)[0].cpu().detach().numpy(),
                                    seg_f.argmax(1)[0].cpu().detach().numpy(),
                                    seg_a.argmax(1)[0].cpu().detach().numpy(),
                                    points_m[0].cpu().detach().numpy(),
                                    points_f[0].cpu().detach().numpy(),
                                    points_a[0].cpu().detach().numpy(),
                                    weights=weights[0].cpu().detach().numpy(),
                                    save_path=None,
                                )

                    # Compute metrics
                    metrics = {}
                    metrics["mse"] = loss_ops.MSELoss()(img_f, img_a).item()
                    if seg_available:
                        metrics["softdiceloss"] = loss_ops.DiceLoss()(seg_a, seg_f)
                        metrics["softdice"] = 1 - metrics["softdiceloss"]
                        dice = loss_ops.DiceLoss(hard=True)(
                            seg_a, seg_f, ign_first_ch=True
                        )
                        dice_total = 1 - dice[0].item()
                        dice_roi = (1 - dice[1].cpu().detach().numpy()).tolist()
                        metrics["harddice"] = dice_total
                        metrics["harddice_roi"] = dice_roi
                        if args.dim == 3:  # TODO: Implement 2D metrics
                            metrics["hausd"] = loss_ops.hausdorff_distance(seg_a, seg_f)
                            grid = grid.permute(0, 4, 1, 2, 3)
                            metrics["jdstd"] = loss_ops.jdstd(grid)
                            metrics["jdlessthan0"] = loss_ops.jdlessthan0(
                                grid, as_percentage=True
                            )

                    for m in list_of_test_metrics:
                        if m == "mse:test":
                            test_metrics[
                                f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                            ].append(metrics["mse"])
                        elif m == "dice_total:test":
                            test_metrics[
                                f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                            ].append(dice_total)
                        elif m == "dice_roi:test":
                            test_metrics[
                                f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                            ].append(dice_roi)

                    if args.save_preds and not args.debug_mode:
                        assert args.batch_size == 1  # TODO: fix this
                        img_f_path = args.model_eval_dir / f"img_f_{i}-{mod1}.npy"
                        img_m_path = args.model_eval_dir / f"img_m_{j}-{mod2}-{aug}.npy"
                        img_a_path = (
                            args.model_eval_dir
                            / f"img_a_{i}-{mod1}_{j}-{mod2}-{aug}-{align_type_str}.npy"
                        )
                        points_f_path = args.model_eval_dir / f"points_f_{i}-{mod1}.npy"
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
                        np.save(grid_path, grid[0].cpu().detach().numpy())

                        if seg_available:
                            seg_f_path = args.model_eval_dir / f"seg_f_{i}-{mod1}.npy"
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

                    for name, metric in metrics.items():
                        if not isinstance(metric, list):
                            print(f"[Eval Stat] {name}: {metric:.5f}")
    return test_metrics


def groupwise_register(
    group_points,
    keypoint_aligner,
    lmbda,
    grid_shape,
    weights=None,
):
    """Perform groupwise registration.

    Args:
        group_points: list of tensors of shape (num_subjects, num_points, dim)
        keypoint_aligner: Keypoint aligner object
        lmbda: Lambda value for TPS
        grid_shape: Grid on which to resample

    Returns:
        grids: All grids for each subject in the group
        points: All transformed points for each subject in the group
    """

    # Compute mean of points, to be used as fixed points
    mean_points = torch.mean(group_points, dim=0, keepdim=True)

    # Register each point to the mean
    grids = []
    new_points = []
    for i in range(len(group_points)):
        points_m = group_points[i : i + 1]
        grid = keypoint_aligner.grid_from_points(
            points_m,
            mean_points,
            grid_shape,
            lmbda=lmbda,
            weights=weights,
            compute_on_subgrids=True,
        )
        points_a = keypoint_aligner.points_from_points(
            points_m,
            mean_points,
            points_m,
            lmbda=lmbda,
            weights=weights,
        )
        grids.append(grid)
        new_points.append(points_a)

    new_points = torch.cat(new_points, dim=0)
    return grids, new_points


def run_groupwise_eval(
    group_loader,
    registration_model,
    mod,
    param,
    aug,
    test_metrics,
    list_of_test_metrics,
    list_of_test_kp_aligns,
    args,
):
    for i, group in enumerate(group_loader[mod]):
        if args.early_stop_eval_subjects and i > args.early_stop_eval_subjects:
            break
        print(f"Running groupwise test: group id {i}, mod {mod}, aug {aug}")

        group_points = []
        for subject in group:
            img = subject["img"][tio.DATA]
            if "seg" in subject:
                seg_available = True
                seg = subject["seg"][tio.DATA]
            else:
                seg_available = False
                assert not any(
                    "dice" in m for m in list_of_test_metrics
                ), "Can't compute Dice if no segmentations available"

            # Move to device
            img = img.float().to(args.device)
            if seg_available:
                seg = seg.float().to(args.device)

            # Explicitly augment moving image
            # if seg_available:
            #     img_m, seg_m = affine_augment(img_m, param, seg=seg_m)
            # else:
            #     img_m = affine_augment(img_m, param)

            with torch.set_grad_enabled(False):
                feat = registration_model.backbone(img)
                points = registration_model.keypoint_layer(feat)

                # if args.weighted_kp_align == "variance":
                #     weights = registration_model.weight_by_variance(feat_f, feat_m)
                # elif args.weighted_kp_align == "power":
                #     weights = registration_model.weight_by_power(feat_f, feat_m)
                # else:
                #     weights = None
                weights = None  # TODO: support weighted groupwise registration??
                group_points.append(points)

        group_points = torch.cat(group_points, dim=0)
        for _, align_type_str in enumerate(list_of_test_kp_aligns):
            # Align via keypoints
            if align_type_str == "rigid":
                keypoint_aligner = RigidKeypointAligner(args.dim)
                lmbda = None
                num_iters = 1
            elif align_type_str == "affine":
                keypoint_aligner = AffineKeypointAligner(args.dim)
                lmbda = None
                num_iters = 1
            elif "tps" in align_type_str:
                _, tps_lmbda = align_type_str.split("_")
                keypoint_aligner = TPS(args.dim, num_subgrids=args.num_subgrids)
                lmbda = _get_tps_lmbda(len(img), float(tps_lmbda)).to(args.device)
                num_iters = 5

            curr_points = group_points.clone()
            for iternum in range(num_iters):
                grids, points_a = groupwise_register(
                    curr_points,
                    keypoint_aligner,
                    lmbda,
                    img.shape,
                    weights=weights,
                )

                tot = 0
                num = 0
                aligned_imgs = []
                for subject, grid in zip(group, grids):
                    img = subject["img"][tio.DATA].to(args.device)
                    img_a = align_img(grid, img)
                    aligned_imgs.append(img_a)
                for i, img1 in enumerate(aligned_imgs):
                    for j, img2 in enumerate(aligned_imgs):
                        if i != j:
                            tot += loss_ops.MSELoss()(img1, img2).item()
                            num += 1

                curr_points = points_a.clone()

                print(align_type_str, iternum, tot / num)
            # Compute metrics
            # metrics = {}
            # metrics["mse"] = loss_ops.MSELoss()(img_f, img_a).item()
            # if seg_available:
            #     metrics["softdiceloss"] = loss_ops.DiceLoss()(seg_a, seg_f)
            #     metrics["softdice"] = 1 - metrics["softdiceloss"]
            #     dice = loss_ops.DiceLoss(hard=True)(seg_a, seg_f, ign_first_ch=True)
            #     dice_total = 1 - dice[0].item()
            #     dice_roi = (1 - dice[1].cpu().detach().numpy()).tolist()
            #     metrics["harddice"] = dice_total
            #     metrics["harddice_roi"] = dice_roi
            #     if args.dim == 3:  # TODO: Implement 2D metrics
            #         metrics["hausd"] = loss_ops.hausdorff_distance(seg_a, seg_f)
            #         grid = grid.permute(0, 4, 1, 2, 3)
            #         metrics["jdstd"] = loss_ops.jdstd(grid)
            #         metrics["jdlessthan0"] = loss_ops.jdlessthan0(grid, as_percentage=True)

        # for m in list_of_test_metrics:
        #     if m == "mse:test":
        #         test_metrics[f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"].append(
        #             metrics["mse"]
        #         )
        #     elif m == "dice_total:test":
        #         test_metrics[f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"].append(
        #             dice_total
        #         )
        #     elif m == "dice_roi:test":
        #         test_metrics[f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"].append(
        #             dice_roi
        #         )

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

        # for name, metric in metrics.items():
        #     if not isinstance(metric, list):
        #         print(f"[Eval Stat] {name}: {metric:.5f}")
    return test_metrics


def main():
    args = parse_args()
    arg_dict = vars(deepcopy(args))
    if args.loss_fn == "mse":
        assert not args.mix_modalities, "MSE loss can't mix modalities"
    if args.debug_mode:
        args.steps_per_epoch = 3
    pprint(arg_dict)

    # Path to save outputs
    if args.eval:
        prefix = "__eval__"
        dataset_str = args.test_dataset
    else:
        prefix = "__training__"
        dataset_str = args.train_dataset
    arguments = (
        prefix
        + args.job_name
        + "_dataset"
        + dataset_str
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

    if args.eval:
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

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        print("WARNING! No GPU available, using the CPU instead...")
        args.device = torch.device("cpu")
    print("Number of GPUs: {}".format(torch.cuda.device_count()))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    if args.train_dataset == "ixi":
        train_loader, _ = ixi.get_loaders()
    elif args.train_dataset == "gigamed":
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size, args.num_workers, load_seg=False
        )
        train_loader = gigamed_dataset.get_train_loader()
    elif args.train_dataset == "synthbrain":
        synth_dataset = synthbrain.SynthBrain(args.batch_size, args.num_workers)
        train_loader = synth_dataset.get_train_loader()
    elif args.train_dataset == "gigamed+synthbrain":
        gigamed_synthbrain_dataset = gigamed.GigaMedSynthBrain(
            args.batch_size, args.num_workers, load_seg=False
        )
        train_loader = gigamed_synthbrain_dataset.get_train_loader()
    elif args.train_dataset == "gigamed+synthbrain+randomanistropy":
        transform = tio.Compose(
            [
                tio.Lambda(synthbrain.one_hot, include=("seg")),
                tio.RandomAnisotropy(downsampling=(1, 4)),
            ]
        )
        gigamed_synthbrain_dataset = gigamed.GigaMedSynthBrain(
            args.batch_size, args.num_workers, load_seg=False, transform=transform
        )
        train_loader = gigamed_synthbrain_dataset.get_train_loader()
    else:
        raise ValueError('Invalid train datasets "{}"'.format(args.train_dataset))

    if args.test_dataset == "ixi":
        _, test_loaders = ixi.get_loaders()
    elif args.test_dataset == "gigamed":
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size, args.num_workers, load_seg=False
        )
        test_loaders = gigamed_dataset.get_test_loaders()
        group_loaders = gigamed_dataset.get_group_loaders()
    else:
        raise ValueError('Invalid test dataset "{}"'.format(args.test_dataset))

    if args.kpconsistency_coeff > 0:
        assert (
            len(moving_datasets) > 1
        ), "Need more than one modality to compute keypoint consistency loss"
        assert all(
            [len(fd) == len(md) for fd, md in zip(fixed_datasets, moving_datasets)]
        ), "Must have same number of subjects for fixed and moving datasets"

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

    # Optimizer
    optimizer = torch.optim.Adam(registration_model.parameters(), lr=args.lr)

    # Checkpoint loading
    if args.load_path is not None:
        ckpt_state, registration_model, optimizer = utils.load_checkpoint(
            args.load_path, registration_model, optimizer, resume=args.resume
        )

    if args.eval:
        assert args.batch_size == 1, ":("
        registration_model.eval()

        if args.save_preds:
            args.model_eval_dir = args.model_dir / "eval"
            if not os.path.exists(args.model_eval_dir) and not args.debug_mode:
                os.makedirs(args.model_eval_dir)

        list_of_test_metrics = [
            "mse:test",
            # "dice_total:test",
            # "dice_roi:test",
        ]
        if args.test_dataset == "ixi":
            list_of_id_test_mods = [
                ("T1", "T1"),
                ("T2", "T2"),
                ("PD", "PD"),
                ("T1", "T2"),
                ("T1", "PD"),
                ("T2", "PD"),
            ]
            list_of_ood_test_mods = None
        elif args.test_dataset == "gigamed":
            list_of_id_test_mods = [
                # ('Dataset4999_IXIAllModalities', 'Dataset4999_IXIAllModalities')
                ("Dataset5083_IXIT1", "Dataset5083_IXIT1"),
                ("Dataset5084_IXIT2", "Dataset5084_IXIT2"),
                ("Dataset5085_IXIPD", "Dataset5085_IXIPD"),
            ]
            list_of_ood_test_mods = [
                ("Dataset6003_AIBL", "Dataset6003_AIBL"),
            ]
            list_of_group_test_mods = [
                "Dataset6003_AIBL",
            ]
        else:
            raise ValueError('Invalid dataset "{}"'.format(args.dataset))
        list_of_test_augs = [
            "rot0",
            "rot45",
            "rot90",
            "rot135",
            "rot180",
        ]

        list_of_test_kp_aligns = [
            "rigid",
            "affine",
            "tps_10",
            "tps_1",
            "tps_0",
        ]
        list_of_all_test = []
        for s1 in list_of_test_metrics:
            for s2 in list_of_id_test_mods:
                mod1, mod2 = s2
                for s3 in list_of_test_augs:
                    for s4 in list_of_test_kp_aligns:
                        list_of_all_test.append(f"{s1}:{mod1}:{mod2}:{s3}:{s4}")
        test_metrics = {}
        test_metrics.update({key: [] for key in list_of_all_test})

        # ID
        for mod in list_of_id_test_mods:
            for aug in list_of_test_augs:
                mod1, mod2 = utils.parse_test_mod(mod)
                param = utils.parse_test_aug(aug)
                test_metrics = run_eval(
                    test_loaders,
                    registration_model,
                    mod1,
                    mod2,
                    param,
                    aug,
                    test_metrics,
                    list_of_test_metrics,
                    list_of_test_kp_aligns,
                    args,
                )

        if not args.debug_mode:
            save_summary_json(test_metrics, args.model_result_dir / "summary.json")

        # OOD
        if list_of_ood_test_mods is not None:
            list_of_all_test = []
            for s1 in list_of_test_metrics:
                for s2 in list_of_ood_test_mods:
                    mod1, mod2 = s2
                    for s3 in list_of_test_augs:
                        for s4 in list_of_test_kp_aligns:
                            list_of_all_test.append(f"{s1}:{mod1}:{mod2}:{s3}:{s4}")
            test_metrics = {}
            test_metrics.update({key: [] for key in list_of_all_test})
            for mod in list_of_ood_test_mods:
                for aug in list_of_test_augs:
                    mod1, mod2 = utils.parse_test_mod(mod)
                    param = utils.parse_test_aug(aug)
                    test_metrics = run_eval(
                        test_loaders,
                        registration_model,
                        mod1,
                        mod2,
                        param,
                        aug,
                        test_metrics,
                        list_of_test_metrics,
                        list_of_test_kp_aligns,
                        args,
                    )

            if not args.debug_mode:
                save_summary_json(
                    test_metrics, args.model_result_dir / "summary_ood.json"
                )

        # Group
        if list_of_group_test_mods is not None:
            list_of_all_test = []
            for s1 in list_of_test_metrics:
                for s2 in list_of_group_test_mods:
                    for s3 in list_of_test_augs:
                        for s4 in list_of_test_kp_aligns:
                            list_of_all_test.append(f"{s1}:{s2}:{s3}:{s4}")
            test_metrics = {}
            test_metrics.update({key: [] for key in list_of_all_test})
            for mod in list_of_group_test_mods:
                for aug in list_of_test_augs:
                    param = utils.parse_test_aug(aug)
                    test_metrics = run_groupwise_eval(
                        group_loaders,
                        registration_model,
                        mod,
                        param,
                        aug,
                        test_metrics,
                        list_of_test_metrics,
                        list_of_test_kp_aligns,
                        args,
                    )

            if not args.debug_mode:
                save_summary_json(
                    test_metrics, args.model_result_dir / "summary_group.json"
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
                train_loader,
                registration_model,
                optimizer,
                args,
            )
            train_loss.append(epoch_stats["loss"])

            print(f"\nEpoch {epoch}/{args.epochs}")
            for name, metric in epoch_stats.items():
                print(f"[Train Stat] {name}: {metric:.5f}")

            if args.use_wandb and not args.debug_mode:
                wandb.log(epoch_stats)

            # Save model
            state = {
                "epoch": epoch,
                "args": args,
                "state_dict": registration_model.backbone.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            if epoch % args.log_interval == 0 and not args.debug_mode:
                torch.save(
                    state,
                    os.path.join(
                        args.model_ckpt_dir,
                        "epoch{}_trained_model.pth.tar".format(epoch),
                    ),
                )


if __name__ == "__main__":
    main()
