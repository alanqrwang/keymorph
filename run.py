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


def create_dirs(args):
    arg_dict = vars(deepcopy(args))

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_data(args):
    if args.train_dataset == "ixi":
        train_loader, _ = ixi.get_loaders()
    elif args.train_dataset == "gigamed":
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size, args.num_workers, load_seg=args.seg_available
        )
        train_loader = gigamed_dataset.get_train_loader()
    elif args.train_dataset == "synthbrain":
        synth_dataset = synthbrain.SynthBrain(args.batch_size, args.num_workers)
        train_loader = synth_dataset.get_train_loader()
    elif args.train_dataset == "gigamed+synthbrain":
        gigamed_synthbrain_dataset = gigamed.GigaMedSynthBrain(
            args.batch_size, args.num_workers, load_seg=args.seg_available
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
            args.batch_size,
            args.num_workers,
            load_seg=args.seg_available,
            transform=transform,
        )
        train_loader = gigamed_synthbrain_dataset.get_train_loader()
    else:
        raise ValueError('Invalid train datasets "{}"'.format(args.train_dataset))

    if args.test_dataset == "ixi":
        _, eval_loaders = ixi.get_loaders()
    elif args.test_dataset == "gigamed":
        gigamed_dataset = gigamed.GigaMed(
            args.batch_size, args.num_workers, load_seg=args.seg_available
        )
        id_eval_loaders = gigamed_dataset.get_eval_loaders(id=True)
        id_eval_group_loaders = gigamed_dataset.get_eval_group_loaders(id=True)
        id_eval_long_loaders = gigamed_dataset.get_eval_longitudinal_loaders(id=True)
        ood_eval_loaders = gigamed_dataset.get_eval_loaders(id=False)
        ood_eval_group_loaders = gigamed_dataset.get_eval_group_loaders(id=False)
        ood_eval_long_loaders = gigamed_dataset.get_eval_longitudinal_loaders(id=False)
    else:
        raise ValueError('Invalid test dataset "{}"'.format(args.test_dataset))

    # if args.kpconsistency_coeff > 0:
    #     assert (
    #         len(moving_datasets) > 1
    #     ), "Need more than one modality to compute keypoint consistency loss"
    #     assert all(
    #         [len(fd) == len(md) for fd, md in zip(fixed_datasets, moving_datasets)]
    #     ), "Must have same number of subjects for fixed and moving datasets"
    return train_loader, {
        "id": id_eval_loaders,
        "id_group": id_eval_group_loaders,
        "id_long": id_eval_long_loaders,
        "ood": ood_eval_loaders,
        "ood_group": ood_eval_group_loaders,
        "ood_long": ood_eval_long_loaders,
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

    for step_idx, (fixed, moving, task_type) in enumerate(train_loader):
        task_type = task_type[0]
        if step_idx == args.steps_per_epoch:
            break

        if task_type == "same_sub_same_mod":
            kp_align_type = "rigid"
            args.loss_fn = "mse"
            max_random_params = (0, 0.15, 3.1416, 0)
        elif task_type == "diff_sub_same_mod":
            kp_align_type = "tps_loguniform"
            args.loss_fn = "mse"
            max_random_params = (0.2, 0.15, 3.1416, 0.1)
        elif task_type == "synthbrain":
            kp_align_type = "tps_loguniform"
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
        if args.seg_available:
            seg_f, seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]

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
                kp_align_type=kp_align_type,
                return_aligned_points=args.visualize,
            )
            grid = registration_results["grid"][0]
            align_type = registration_results["align_type"][0]
            tps_lmbda = registration_results["tps_lmbda"][0]
            points_m = registration_results["points_m"][0]
            points_f = registration_results["points_f"][0]
            points_weights = registration_results["points_weights"][0]

            img_a = align_img(grid, img_m)
            if args.seg_available:
                seg_a = align_img(grid, seg_m)

            # Compute metrics
            metrics = {}
            metrics["scale_augment"] = scale_augment
            metrics["mse"] = loss_ops.MSELoss()(img_f, img_a)
            if args.seg_available:
                metrics["softdiceloss"] = loss_ops.DiceLoss()(seg_a, seg_f)
                metrics["softdice"] = 1 - metrics["softdiceloss"]

            # Compute loss
            if args.loss_fn == "mse":
                loss = metrics["mse"]
            elif args.loss_fn == "dice":
                loss = metrics["softdiceloss"]
            else:
                raise ValueError('Invalid loss function "{}"'.format(args.loss_fn))
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
            print(f"-> Task type: {task_type}")
            print(f"-> Alignment: {align_type} ")
            print(f"-> Max random params: {max_random_params} ")
            print(f"-> TPS lambda: {tps_lmbda} ")
            print(f"-> Loss: {args.loss_fn}")
            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
            print(f"-> Point shapes: {points_f.shape}, {points_m.shape}")
            print(f"-> Point weights: {points_weights}")
            print(f"-> Float16: {args.use_amp}")
            if args.seg_available:
                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

        if args.visualize and step_idx == 0:
            points_a = registration_results["points_a"][0]
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
                    save_path=None
                    if args.debug_mode
                    else os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png"),
                )
                if args.seg_available:
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
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_kp_aligns,
    args,
):
    def build_metric_dict(names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            for a in list_of_eval_augs:
                for k in list_of_eval_kp_aligns:
                    for n in names:
                        n1, n2 = n
                        list_of_all_test.append(f"{m}:{n1}:{n2}:{a}:{k}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    test_metrics = build_metric_dict(list_of_eval_names)
    for name in list_of_eval_names:
        for aug in list_of_eval_augs:
            mod1, mod2 = utils.parse_test_mod(name)
            param = utils.parse_test_aug(aug)
            for i, fixed in enumerate(loaders[mod1]):
                if args.early_stop_eval_subjects and i > args.early_stop_eval_subjects:
                    break
                for j, moving in enumerate(loaders[mod2]):
                    if (
                        args.early_stop_eval_subjects
                        and j > args.early_stop_eval_subjects
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
                            kp_align_type=list_of_eval_kp_aligns,
                            return_aligned_points=True,
                        )
                        points_m = registration_results["points_m"]
                        points_f = registration_results["points_f"]
                        points_weights = registration_results["points_weights"]

                        for align_type_str, grid, points_a in zip(
                            list_of_eval_kp_aligns,
                            registration_results["grid"],
                            registration_results["points_a"],
                        ):
                            img_a = align_img(grid, img_m)
                            if args.seg_available:
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
                                        weights=points_weights[0]
                                        .cpu()
                                        .detach()
                                        .numpy(),
                                    )
                                    if args.seg_available:
                                        show_warped(
                                            seg_m[0, 0].cpu().detach().numpy(),
                                            seg_f[0, 0].cpu().detach().numpy(),
                                            seg_a[0, 0].cpu().detach().numpy(),
                                            points_m[0].cpu().detach().numpy(),
                                            points_f[0].cpu().detach().numpy(),
                                            points_a[0].cpu().detach().numpy(),
                                            weights=points_weights[0]
                                            .cpu()
                                            .detach()
                                            .numpy(),
                                        )
                                else:
                                    show_warped_vol(
                                        img_m[0, 0].cpu().detach().numpy(),
                                        img_f[0, 0].cpu().detach().numpy(),
                                        img_a[0, 0].cpu().detach().numpy(),
                                        points_m[0].cpu().detach().numpy(),
                                        points_f[0].cpu().detach().numpy(),
                                        points_a[0].cpu().detach().numpy(),
                                        weights=points_weights[0]
                                        .cpu()
                                        .detach()
                                        .numpy(),
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
                                            weights=points_weights[0]
                                            .cpu()
                                            .detach()
                                            .numpy(),
                                            save_path=None,
                                        )

                            # Compute metrics
                            metrics = {}
                            if args.seg_available:
                                # Always compute hard dice once ahead of time
                                softdiceloss = loss_ops.DiceLoss()(seg_a, seg_f)
                                dice = loss_ops.DiceLoss(hard=True)(
                                    seg_a, seg_f, ign_first_ch=True
                                )
                                dice_total = 1 - dice[0].item()
                                dice_roi = (1 - dice[1].cpu().detach().numpy()).tolist()
                            for m in list_of_eval_metrics:
                                if m == "mse:test":
                                    metrics["mse"] = loss_ops.MSELoss()(
                                        img_f, img_a
                                    ).item()
                                    test_metrics[
                                        f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                    ].append(metrics["mse"])
                                elif m == "softdice:test":
                                    assert args.seg_available
                                    metrics["softdiceloss"] = softdiceloss
                                    metrics["softdice"] = 1 - softdiceloss
                                    test_metrics[
                                        f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                    ].append(metrics["softdice"])
                                elif m == "harddice:test":
                                    assert args.seg_available
                                    metrics["harddice"] = dice_total
                                    test_metrics[
                                        f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                    ].append(metrics["harddice"])
                                elif m == "harddiceroi:test":
                                    # Don't save roi into metric dict, since it's a list
                                    assert args.seg_available
                                    test_metrics[
                                        f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                    ].append(dice_roi)
                                elif m == "hausd:test":
                                    assert args.seg_available and args.dim == 3
                                    metrics["hausd"] = loss_ops.hausdorff_distance(
                                        seg_a, seg_f
                                    )
                                    test_metrics[
                                        f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                    ].append(metrics["hausd"])
                                elif m == "jdstd:test":
                                    assert args.seg_available and args.dim == 3
                                    grid_permute = grid.permute(0, 4, 1, 2, 3)
                                    metrics["jdstd"] = loss_ops.jdstd(grid_permute)
                                    test_metrics[
                                        f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                    ].append(metrics["jdstd"])
                                elif m == "jdlessthan0:test":
                                    assert args.seg_available and args.dim == 3
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
                                img_f_path = (
                                    args.model_eval_dir / f"img_f_{i}-{mod1}.npy"
                                )
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
                                    args.model_eval_dir
                                    / f"points_m_{j}-{mod2}-{aug}.npy"
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

                                if args.seg_available:
                                    seg_f_path = (
                                        args.model_eval_dir / f"seg_f_{i}-{mod1}.npy"
                                    )
                                    seg_m_path = (
                                        args.model_eval_dir
                                        / f"seg_m_{j}-{mod2}-{aug}.npy"
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


def run_groupwise_eval(
    group_loader,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_kp_aligns,
    args,
):
    def build_metric_dict(names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            for a in list_of_eval_augs:
                for k in list_of_eval_kp_aligns:
                    for n in names:
                        list_of_all_test.append(f"{m}:{n}:{a}:{k}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    test_metrics = build_metric_dict(list_of_eval_names)
    for name in list_of_eval_names:
        for aug in list_of_eval_augs:
            param = utils.parse_test_aug(aug)
            # For memory efficiency, we save preprocessed images in a group to a temp folder
            temp_dir = Path("./temp_group/")
            for i, group in enumerate(group_loader[name]):
                if args.early_stop_eval_subjects and i > args.early_stop_eval_subjects:
                    break
                print(
                    f"Running groupwise test: group id {i}, dataset {name}, aug {aug}"
                )

                # For groupwise, we randomly affine augment all images
                for subject in group:
                    img = subject["img"][tio.DATA]
                    if args.seg_available:
                        seg = subject["seg"][tio.DATA]
                        img, seg = random_affine_augment(img, param, seg=seg)
                    else:
                        img = random_affine_augment(img, param)
                    transformed_subject = tio.Subject(img=img, seg=seg)
                    transformed_subject.img.save(temp_dir / f"img_{i}.nii.gz")
                    transformed_subject.seg.save(temp_dir / f"seg_{i}.nii.gz")

                with torch.set_grad_enabled(False):
                    grids, points_as = registration_model.groupwise_register(
                        temp_dir, list_of_eval_kp_aligns
                    )

                tot_mse = 0
                tot_softdice = 0
                num = 0
                aligned_imgs = []
                for subject, grid in zip(group, grids):
                    img = subject["img"][tio.DATA].to(args.device)
                    img_a = align_img(grid, img)
                    aligned_imgs.append(img_a)
                for i, img1 in enumerate(aligned_imgs):
                    for j, img2 in enumerate(aligned_imgs):
                        if i != j:
                            tot_mse += loss_ops.MSELoss()(img1, img2).item()
                            tot_softdice += loss_ops.DiceLoss()(img1, img2).item()
                            num += 1

                # Compute metrics
                metrics = {}
                metrics["mse"] = tot_mse / num
                if args.seg_available:
                    metrics["softdiceloss"] = tot_softdice / num
                    metrics["softdice"] = 1 - metrics["softdiceloss"]
                    # dice = loss_ops.DiceLoss(hard=True)(seg_a, seg_f, ign_first_ch=True)
                    # dice_total = 1 - dice[0].item()
                    # dice_roi = (1 - dice[1].cpu().detach().numpy()).tolist()
                    # metrics["harddice"] = dice_total
                    # metrics["harddice_roi"] = dice_roi
                    # if args.dim == 3:  # TODO: Implement 2D metrics
                    #     metrics["hausd"] = loss_ops.hausdorff_distance(seg_a, seg_f)
                    #     grid = grid.permute(0, 4, 1, 2, 3)
                    #     metrics["jdstd"] = loss_ops.jdstd(grid)
                    #     metrics["jdlessthan0"] = loss_ops.jdlessthan0(grid, as_percentage=True)

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

                for name, metric in metrics.items():
                    if not isinstance(metric, list):
                        print(f"[Eval Stat] {name}: {metric:.5f}")

                # Clean up
                for f in os.listdir(temp_dir):
                    os.remove(temp_dir / f)
    return test_metrics


def main():
    args = parse_args()
    if args.loss_fn == "mse":
        assert not args.mix_modalities, "MSE loss can't mix modalities"
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
    train_loader, eval_loaders = get_data(args)

    # Model
    registration_model = get_model(args)

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

        list_of_eval_metrics = [
            "mse:test",
            "softdice:test",
            "harddice:test",
            "hausd:test",
            "jdstd:test",
            "jdlessthan0:test",
        ]
        if args.test_dataset == "ixi":
            list_of_eval_names = {
                "id": [
                    ("T1", "T1"),
                    ("T2", "T2"),
                    ("PD", "PD"),
                    ("T1", "T2"),
                    ("T1", "PD"),
                    ("T2", "PD"),
                ]
            }
            list_of_group_names = None
            list_of_long_names = None
        elif args.test_dataset == "gigamed":
            list_of_eval_names = {
                "id": [
                    # ('Dataset4999_IXIAllModalities', 'Dataset4999_IXIAllModalities'),
                    # ("Dataset5083_IXIT1", "Dataset5083_IXIT1"),
                    # ("Dataset5084_IXIT2", "Dataset5084_IXIT2"),
                    # ("Dataset5085_IXIPD", "Dataset5085_IXIPD"),
                    # ("Dataset5000_BraTS-GLI_2023", "Dataset5000_BraTS-GLI_2023"),
                ],
                "ood": [
                    ("Dataset6003_AIBL", "Dataset6003_AIBL"),
                ],
            }
            list_of_group_names = {
                "id": [
                    "Dataset5083_IXIT1",
                ],
                "ood": [
                    "Dataset6003_AIBL",
                ],
            }
            list_of_long_names = {
                "id": [
                    "Dataset6000_PPMI-T1-3T-PreProc",
                    "Dataset6001_ADNI-group-T1-3T-PreProc",
                    "Dataset6002_OASIS3",
                ],
                "ood": [
                    "Dataset6003_AIBL",
                ],
            }
        else:
            raise ValueError('Invalid dataset "{}"'.format(args.dataset))

        list_of_eval_augs = [
            "rot0",
            "rot45",
            "rot90",
            "rot135",
            "rot180",
        ]

        list_of_eval_kp_aligns = [
            "rigid",
            "affine",
            "tps_10",
            "tps_1",
            "tps_0.1",
            "tps_0.01",
            "tps_0",
        ]

        for dist in ["id", "ood"]:
            eval_metrics = run_eval(
                eval_loaders[dist],
                registration_model,
                list_of_eval_metrics,
                list_of_eval_names[dist],
                list_of_eval_augs,
                list_of_eval_kp_aligns,
                args,
            )
            if not args.debug_mode:
                save_summary_json(
                    eval_metrics, args.model_result_dir / f"summary_{dist}.json"
                )

            # Groupwise
            if list_of_group_names is not None:
                group_metrics = run_groupwise_eval(
                    eval_loaders[f"{dist}_group"],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_names[dist],
                    list_of_eval_augs,
                    list_of_eval_kp_aligns,
                    args,
                )

                if not args.debug_mode:
                    save_summary_json(
                        group_metrics,
                        args.model_result_dir / f"summary_group_{dist}.json",
                    )

            # Longitudinal
            if list_of_long_names is not None:
                long_metrics = run_groupwise_eval(
                    eval_loaders[f"{dist}_long"],
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_names[dist],
                    list_of_eval_augs,
                    list_of_eval_kp_aligns,
                    args,
                )

                if not args.debug_mode:
                    save_summary_json(
                        long_metrics,
                        args.model_result_dir / f"summary_long_{dist}.json",
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
