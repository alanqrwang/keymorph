import random
import os
import time
import torch
import numpy as np
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
import wandb

from keymorph.net import ConvNetFC, ConvNetCoM
from keymorph.cm_plotter import show_warped, show_warped_vol
from keymorph.data import ixi, gigamed
from keymorph import utils
from keymorph.utils import ParseKwargs, initialize_wandb
from keymorph.augmentation import random_affine_augment


def parse_args():
    parser = ArgumentParser()

    # I/O
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
        "--kp_extractor",
        type=str,
        default="conv_com",
        choices=["conv_fc", "conv_com"],
        help="Keypoint extractor module to use",
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
        default=100,
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


def run_train(loaders, random_points, network, optimizer, epoch, args):
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    network.train()

    res = []

    iters = [iter(loader) for loader in loaders]
    random_points = random_points.to(args.device)
    for _ in range(args.steps_per_epoch):
        # Choose modality at random
        subject = next(random.choice(iters))
        x_fixed = subject["img"][tio.DATA].float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = None
        x_moving, tgt_points = random_affine_augment(
            x_fixed, points=random_points, scale_params=scale_augment
        )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                pred_points = network(x_moving)
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

        if args.visualize:
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
                )

    return utils.aggregate_dicts(res)


def main():
    args = parse_args()

    # Path to save outputs
    arguments = (
        "[pretraining]keypoints"
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
        dataset_names = ["T1", "T2", "PD"]

        transform = tio.Compose(
            [
                #                 RandomBiasField(),
                #                 RandomNoise(),
                tio.Lambda(lambda x: x.permute(0, 1, 3, 2)),
                tio.Mask(masking_method="mask"),
                tio.Resize(128),
                tio.Lambda(ixi.one_hot, include=("seg")),
            ]
        )

        train_datasets = {}
        test_datasets = {}
        for mod in dataset_names:
            train_subjects = ixi.read_subjects_from_disk(args.data_dir, (0, 427), mod)
            train_datasets[mod] = tio.data.SubjectsDataset(
                train_subjects, transform=transform
            )
            test_subjects = ixi.read_subjects_from_disk(
                args.data_dir, (428, 428 + args.num_test_subjects), mod
            )
            test_datasets[mod] = tio.data.SubjectsDataset(
                test_subjects, transform=transform
            )
        ref_subject = train_datasets[dataset_names[0]][0]
    else:
        raise NotImplementedError

    fixed_loaders = {
        k: DataLoader(
            v, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        for k, v in train_datasets.items()
    }

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

    # CNN, i.e. keypoint extractor
    if args.kp_extractor == "conv_fc":
        network = ConvNetFC(
            args.dim, 1, args.num_keypoints * args.dim, norm_type=args.norm_type
        )
        network = torch.nn.DataParallel(network)
    elif args.kp_extractor == "conv_com":
        network = ConvNetCoM(args.dim, 1, args.num_keypoints, norm_type=args.norm_type)
    network = torch.nn.DataParallel(network)
    network.to(args.device)
    utils.summary(network)

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
    else:
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        args.curr_epoch = epoch
        epoch_stats = run_train(
            list(fixed_loaders.values()),
            random_points,
            network,
            optimizer,
            epoch,
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
                    args.model_dir, "pretrained_epoch{}_model.pth.tar".format(epoch)
                ),
            )
        del state


if __name__ == "__main__":
    main()
