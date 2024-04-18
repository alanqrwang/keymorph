import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from argparse import ArgumentParser
import torchio as tio
from scipy.stats import loguniform
from pathlib import Path

from keymorph.keypoint_aligners import ClosedFormAffine, TPS
from keymorph.net import ConvNetFC, ConvNetCoM
from keymorph.model import KeyMorph
from keymorph.utils import align_img
from keymorph import loss_ops
from gigamed import ixi, gigamed, synthbrain


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

    parser.add_argument("--save_preds", action="store_true", help="Perform evaluation")

    parser.add_argument(
        "--visualize", action="store_true", help="Visualize images and points"
    )

    # KeyMorph
    parser.add_argument(
        "--num_keypoints", type=int, required=True, help="Number of keypoints"
    )

    parser.add_argument(
        "--kp_extractor",
        type=str,
        default="conv_com",
        choices=["conv_fc", "conv_com"],
        help="Keypoint extractor module to use",
    )

    parser.add_argument(
        "--kp_align_method",
        type=str,
        default="affine",
        choices=["affine", "tps"],
        help="Keypoint alignment module to use",
    )

    parser.add_argument("--tps_lmbda", default=None, help="TPS lambda value")

    parser.add_argument(
        "--norm_type",
        type=str,
        default="instance",
        choices=["none", "instance", "batch", "group"],
        help="Normalization type",
    )

    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mse",
        choices=["mse", "dice"],
        help="Loss function",
    )

    # Data
    parser.add_argument("--moving", type=str, required=True, help="Moving image path")

    parser.add_argument("--fixed", type=str, required=True, help="Fixed image path")

    parser.add_argument("--moving_seg", type=str, default=None, help="Moving seg path")

    parser.add_argument("--fixed_seg", type=str, default=None, help="Fixed seg path")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=23,
        help="Random seed use to sort the training data",
    )

    parser.add_argument("--dim", type=int, default=3)

    args = parser.parse_args()
    return args


def _get_tps_lmbda(num_samples, args):
    if args.tps_lmbda is None:
        assert args.kp_align_method != "tps", "Need to implement this"
        lmbda = None
    else:
        lmbda = torch.tensor(float(args.tps_lmbda)).repeat(num_samples).to(args.device)
    return lmbda


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
    if not os.path.exists(save_path) and args.save_preds:
        os.makedirs(save_path)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    if args.half_resolution:
        transform = tio.Compose(
            [
                #                 RandomBiasField(),
                #                 RandomNoise(),
                tio.Lambda(lambda x: x.permute(0, 1, 3, 2)),
                tio.Resize(128),
                tio.Lambda(ixi.one_hot, include=("seg")),
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
            ],
            include=("img", "seg"),
        )

    # Build subject
    moving_dict = {"img": tio.ScalarImage(args.moving)}
    fixed_dict = {"img": tio.ScalarImage(args.fixed)}
    if args.moving_seg is not None:
        moving_dict["seg"] = tio.LabelMap(args.moving_seg)
    if args.fixed_seg is not None:
        fixed_dict["seg"] = tio.LabelMap(args.fixed_seg)
    fixed = [tio.Subject(**fixed_dict)]
    moving = [tio.Subject(**moving_dict)]

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

    if args.load_path:
        state_dict = torch.load(args.load_path)["state_dict"]
        network.load_state_dict(state_dict)

    # Keypoint alignment module
    if args.kp_align_method == "affine":
        kp_aligner = ClosedFormAffine(args.dim)
    elif args.kp_align_method == "tps":
        kp_aligner = TPS(args.dim)
    else:
        raise NotImplementedError

    # Keypoint model
    registration_model = KeyMorph(network, kp_aligner, args.num_keypoints, args.dim)
    registration_model.eval()

    for i, fixed in enumerate(fixed_loader):
        for j, moving in enumerate(moving_loader):

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
            lmbda = _get_tps_lmbda(len(img_f), args)
            grid, points_f, points_m = registration_model(img_f, img_m, lmbda)
            img_a = align_img(grid, img_m)
            if seg_available:
                seg_a = align_img(grid, seg_m)
            points_a = kp_aligner.points_from_points(
                points_m, points_f, points_m, lmbda=lmbda
            )

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            # axes[0].imshow(img_f[0,0,64].cpu().numpy(), cmap='gray')
            # axes[1].imshow(img_m[0,0,64].cpu().numpy(), cmap='gray')
            # axes[2].imshow(img_a[0,0,64].cpu().numpy(), cmap='gray')
            # plt.show()

            # Compute metrics
            metrics = {}
            metrics["mse"] = loss_ops.MSELoss()(img_f, img_a)
            if seg_available:
                metrics["softdiceloss"] = loss_ops.DiceLoss()(seg_a, seg_f)
                metrics["softdice"] = 1 - metrics["softdiceloss"]
                metrics["harddice"] = (
                    1 - loss_ops.DiceLoss(hard=True)(seg_a, seg_f, ign_first_ch=True)[0]
                )
                if args.dim == 3:  # TODO: Implement 2D metrics
                    metrics["hausd"] = loss_ops.hausdorff_distance(seg_a, seg_f)
                    grid = grid.permute(0, 4, 1, 2, 3)
                    metrics["jdstd"] = loss_ops.jdstd(grid)
                    metrics["jdlessthan0"] = loss_ops.jdlessthan0(
                        grid, as_percentage=True
                    )

            if args.save_preds:
                assert args.batch_size == 1  # TODO: fix this
                img_a_path = save_path / f"img_a_{i}_{j}.npy"
                seg_a_path = save_path / f"seg_a_{i}_{j}.npy"
                points_f_path = save_path / f"points_f_{i}_{j}.npy"
                points_m_path = save_path / f"points_m_{i}_{j}.npy"
                points_a_path = save_path / f"points_a_{i}_{j}.npy"
                grid_path = save_path / f"grid_{i}_{j}.npy"
                print(
                    "Saving:\n{}\n{}\n{}\n{}\n{}\n{}".format(
                        img_a_path,
                        seg_a_path,
                        points_f_path,
                        points_m_path,
                        points_a_path,
                        grid_path,
                    )
                )
                np.save(img_a_path, img_a[0].cpu().detach().numpy())
                np.save(seg_a_path, np.argmax(seg_a.cpu().detach().numpy(), axis=1))
                np.save(points_f_path, points_f[0].cpu().detach().numpy())
                np.save(points_m_path, points_m[0].cpu().detach().numpy())
                np.save(points_a_path, points_a[0].cpu().detach().numpy())
                np.save(grid_path, grid[0].cpu().detach().numpy())

            for name, metric in metrics.items():
                print(f"[Eval Stat] {name}: {metric:.5f}")
