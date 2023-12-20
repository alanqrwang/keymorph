import random
import os
import time
import torch
import numpy as np
import torchio as tio
import torch.nn.functional as F
from argparse import ArgumentParser
from pathlib import Path
import wandb

from keymorph.net import ConvNet, UNet, RXFM_Net
from keymorph.cm_plotter import show_warped, show_warped_vol
from keymorph.data import ixi, gigamed
from keymorph import utils
from keymorph.utils import ParseKwargs, initialize_wandb
from keymorph.augmentation import random_affine_augment
from keymorph.layers import (
    LinearRegressor2d,
    LinearRegressor3d,
    CenterOfMass2d,
    CenterOfMass3d,
)

from keymorph.se3_3Dcnn import SE3CNN


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
        "--gpus", type=str, default="0", help="Which GPUs to use? Index from 0"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./pretraining_output/",
        help="Path to the folder where data is saved",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/centered_IXI/",
        help="Path to the training data",
    )

    parser.add_argument(
        "--load_path", type=str, default=None, help="Load checkpoint at .h5 path"
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from prior checkpoint"
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Visualize images and points"
    )

    parser.add_argument(
        "--log_interval", type=int, default=25, help="Frequency of logs"
    )

    # KeyMorph
    parser.add_argument(
        "--num_keypoints", type=int, default=128, help="Number of keypoints"
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

    # Data
    parser.add_argument("--dataset", type=str, default="ixi", help="Dataset")

    parser.add_argument(
        "--num_test_subjects", type=int, default=100, help="Number of test subjects"
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

    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate, default: 1e-4"
    )

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
        help="Constant to control how slow to increase augmentation",
    )

    # Miscellaneous
    parser.add_argument("--debug_mode", action="store_true", help="Debug mode")

    parser.add_argument(
        "--seed", type=int, default=23, help="Random seed use to sort the training data"
    )

    parser.add_argument("--dim", type=int, default=3)

    parser.add_argument("--use_amp", action="store_true", help="Use AMP")

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


def run_train(loader, random_points, network, keypoint_extractor, optimizer, args):
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    network.train()

    res = []

    random_points = random_points.to(args.device)
    for step_idx, subject in enumerate(loader):
        if step_idx == args.steps_per_epoch:
            break
        # Choose modality at random
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
                pred_feats = network(x_moving)
                pred_points = keypoint_extractor(pred_feats)
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
                    save_path=None
                    if args.debug_mode
                    else os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png"),
                )

    return utils.aggregate_dicts(res)


def main():
    args = parse_args()

    # Path to save outputs
    arguments = (
        "__pretraining__"
        + args.job_name
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
        os.makedirs(args.model_dir)
    args.model_ckpt_dir = args.model_dir / "checkpoints"
    if not os.path.exists(args.model_ckpt_dir) and not args.debug_mode:
        os.makedirs(args.model_ckpt_dir)
    args.model_result_dir = args.model_dir / "results"
    if not os.path.exists(args.model_result_dir) and not args.debug_mode:
        os.makedirs(args.model_result_dir)
    args.model_img_dir = args.model_dir / "img"
    if not os.path.exists(args.model_img_dir) and not args.debug_mode:
        os.makedirs(args.model_img_dir)

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        print("WARNING! No GPU available, using the CPU instead...")
        args.device = torch.device("cpu")
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("Number of GPUs: {}".format(torch.cuda.device_count()))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    if args.dataset == "ixi":
        train_loader, _ = ixi.get_pretraining_loaders()
        ref_subject = ixi.get_random_subject(train_loader)
    elif args.dataset == "gigamed":
        gmed = gigamed.GigaMed(
            args.data_dir, args.batch_size, args.num_workers, load_seg=False
        )
        train_loader, _ = gmed.get_pretraining_loaders()
        ref_subject = gmed.get_reference_subject()
    else:
        raise NotImplementedError

    # Extract random keypoints from reference subject
    ref_img = ref_subject["img"][tio.DATA].float().unsqueeze(0)
    print("sampling random keypoints...")
    random_points = utils.sample_valid_coordinates(
        ref_img, args.num_keypoints, args.dim
    )
    random_points = random_points * 2 - 1
    random_points = random_points.repeat(args.batch_size, 1, 1)
    show_warped_vol(
        ref_img[0, 0].cpu().detach().numpy(),
        ref_img[0, 0].cpu().detach().numpy(),
        ref_img[0, 0].cpu().detach().numpy(),
        random_points[0].cpu().detach().numpy(),
        random_points[0].cpu().detach().numpy(),
        random_points[0].cpu().detach().numpy(),
    )
    del ref_subject

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
    elif args.backbone == "se3cnn2":
        network = SE3CNN()
    else:
        raise ValueError('Invalid keypoint extractor "{}"'.format(args.backbone))
    network = torch.nn.DataParallel(network)
    network.to(args.device)
    if args.load_path:
        state_dict = torch.load(args.load_path)["state_dict"]
        network.load_state_dict(state_dict)
    utils.summary(network)

    if args.kp_layer == "com":
        if args.dim == 2:
            keypoint_layer = CenterOfMass2d()
        else:
            keypoint_layer = CenterOfMass3d()
    else:
        if args.dim == 2:
            keypoint_layer = LinearRegressor2d()
        else:
            keypoint_layer = LinearRegressor3d()

    # Optimizer
    params = list(network.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    train_loss = []

    # Train
    network.train()

    if args.use_wandb and not args.debug_mode:
        initialize_wandb(args)

    if args.resume:
        state = torch.load(args.load_path)
        network.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        random_points = state["random_points"]
    else:
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        args.curr_epoch = epoch
        epoch_stats = run_train(
            train_loader,
            random_points,
            network,
            keypoint_layer,
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
        state = {
            "epoch": epoch,
            "args": args,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "random_points": random_points,
        }

        if epoch % args.log_interval == 0 and not args.debug_mode:
            torch.save(
                state,
                os.path.join(
                    args.model_ckpt_dir,
                    "pretrained_epoch{}_model.pth.tar".format(epoch),
                ),
            )
        del state


if __name__ == "__main__":
    main()
