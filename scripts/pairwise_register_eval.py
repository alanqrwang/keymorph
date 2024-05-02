import os
import torch
import numpy as np
import torchio as tio

from keymorph import utils
from keymorph.utils import (
    align_img,
    save_dict_as_json,
)
from keymorph.cm_plotter import imshow_registration_2d, imshow_registration_3d
from keymorph.augmentation import affine_augment
import keymorph.loss_ops as loss_ops


def run_eval(
    loaders,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_aligns,
    args,
    save_dir_prefix="eval",
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
                        f"\n\n\nRunning test: subject id {i}->{j}, mod {mod1}->{mod2}, aug {aug}"
                    )

                    if args.save_eval_to_disk and not args.debug_mode:
                        # Create directory to save images, segs, points, metrics
                        mod1_str = "-".join(mod1.split("/")[-2:])
                        mod2_str = "-".join(mod2.split("/")[-2:])
                        save_dir = (
                            args.model_eval_dir
                            / save_dir_prefix
                            / f"{i}_{j}_{mod1_str}_{mod2_str}"
                        )
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                    else:
                        save_dir = None
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
                            seg_f=seg_f if args.seg_available else None,
                            seg_m=seg_m if args.seg_available else None,
                            transform_type=list_of_eval_aligns,
                            return_aligned_points=True,
                            save_dir=save_dir,
                            num_resolutions_for_itkelastix=args.num_resolutions_for_itkelastix,
                        )

                    for align_type_str, res_dict in registration_results.items():
                        if "img_m" in res_dict:
                            img_m = res_dict["img_m"]
                        if "img_f" in res_dict:
                            img_f = res_dict["img_f"]
                        if "img_a" in res_dict:
                            img_a = res_dict["img_a"]
                        elif "grid" in res_dict:
                            grid = res_dict["grid"]
                            img_a = align_img(grid, img_m)
                        else:
                            raise ValueError("No way to get aligned image")
                        if "grid" in res_dict:
                            grid = res_dict["grid"]
                        else:
                            assert (
                                "jdstd" in res_dict and "jdlessthan0" in res_dict
                            )  # If no grid, then must have jdstd and jdlessthan0
                            grid = None
                        if args.seg_available:
                            if "seg_m" in res_dict:
                                seg_m = res_dict["seg_m"]
                            if "seg_f" in res_dict:
                                seg_f = res_dict["seg_f"]
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
                                imshow_registration_2d(
                                    img_m[0, 0].cpu().detach().numpy(),
                                    img_f[0, 0].cpu().detach().numpy(),
                                    img_a[0, 0].cpu().detach().numpy(),
                                    (
                                        points_m[0].cpu().detach().numpy()
                                        if points_m is not None
                                        else None
                                    ),
                                    (
                                        points_f[0].cpu().detach().numpy()
                                        if points_f is not None
                                        else None
                                    ),
                                    (
                                        points_a[0].cpu().detach().numpy()
                                        if points_a is not None
                                        else None
                                    ),
                                    weights=points_weights,
                                )
                                if args.seg_available:
                                    imshow_registration_2d(
                                        seg_m[0, 0].cpu().detach().numpy(),
                                        seg_f[0, 0].cpu().detach().numpy(),
                                        seg_a[0, 0].cpu().detach().numpy(),
                                        (
                                            points_m[0].cpu().detach().numpy()
                                            if points_m is not None
                                            else None
                                        ),
                                        (
                                            points_f[0].cpu().detach().numpy()
                                            if points_f is not None
                                            else None
                                        ),
                                        (
                                            points_a[0].cpu().detach().numpy()
                                            if points_a is not None
                                            else None
                                        ),
                                        weights=points_weights,
                                    )
                            else:
                                imshow_registration_3d(
                                    img_m[0, 0].cpu().detach().numpy(),
                                    img_f[0, 0].cpu().detach().numpy(),
                                    img_a[0, 0].cpu().detach().numpy(),
                                    (
                                        points_m[0].cpu().detach().numpy()
                                        if points_m is not None
                                        else None
                                    ),
                                    (
                                        points_f[0].cpu().detach().numpy()
                                        if points_f is not None
                                        else None
                                    ),
                                    (
                                        points_a[0].cpu().detach().numpy()
                                        if points_a is not None
                                        else None
                                    ),
                                    weights=(
                                        points_weights[0].cpu().detach().numpy()
                                        if points_weights is not None
                                        else None
                                    ),
                                    save_path=None,
                                )
                                if args.seg_available:
                                    imshow_registration_3d(
                                        seg_m.argmax(1)[0].cpu().detach().numpy(),
                                        seg_f.argmax(1)[0].cpu().detach().numpy(),
                                        seg_a.argmax(1)[0].cpu().detach().numpy(),
                                        (
                                            points_m[0].cpu().detach().numpy()
                                            if points_m is not None
                                            else None
                                        ),
                                        (
                                            points_f[0].cpu().detach().numpy()
                                            if points_f is not None
                                            else None
                                        ),
                                        (
                                            points_a[0].cpu().detach().numpy()
                                            if points_a is not None
                                            else None
                                        ),
                                        weights=(
                                            points_weights[0].cpu().detach().numpy()
                                            if points_weights is not None
                                            else None
                                        ),
                                        save_path=None,
                                    )

                        # Compute metrics
                        metrics = {}
                        if args.seg_available:
                            # Always compute hard dice once ahead of time
                            dice_total = loss_ops.DiceLoss(hard=True)(
                                seg_a, seg_f, ign_first_ch=True
                            )
                            dice_roi = loss_ops.DiceLoss(
                                hard=True, return_regions=True
                            )(seg_a, seg_f, ign_first_ch=True)
                            dice_total = 1 - dice_total.item()
                            dice_roi = (1 - dice_roi.cpu().detach().numpy()).tolist()
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
                                assert args.seg_available
                                metrics["harddiceroi"] = dice_roi
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["harddiceroi"])
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
                                if grid is None:
                                    metrics["jdstd"] = res_dict["jdstd"]
                                else:
                                    grid_permute = grid.permute(0, 4, 1, 2, 3)
                                    metrics["jdstd"] = loss_ops.jdstd(grid_permute)
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["jdstd"])
                            elif m == "jdlessthan0":
                                assert args.dim == 3
                                if grid is None:
                                    metrics["jdlessthan0"] = res_dict["jdlessthan0"]
                                else:
                                    grid_permute = grid.permute(0, 4, 1, 2, 3)
                                    metrics["jdlessthan0"] = loss_ops.jdstd(
                                        grid_permute
                                    )
                                test_metrics[
                                    f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"
                                ].append(metrics["jdlessthan0"])
                            else:
                                raise ValueError('Invalid metric "{}"'.format(m))

                        if args.save_eval_to_disk and not args.debug_mode:
                            assert args.batch_size == 1  # TODO: fix this

                            # Save metrics
                            metrics_path = save_dir / f"metrics-{align_type_str}.json"
                            print("Saving:", metrics_path)
                            save_dict_as_json(metrics, metrics_path)

                            # Save images and grid
                            img_f_path = save_dir / f"img_f_{i}-{mod1_str}.npy"
                            img_m_path = save_dir / f"img_m_{j}-{mod2_str}-{aug}.npy"
                            img_a_path = (
                                save_dir
                                / f"img_a_{i}-{mod1_str}_{j}-{mod2_str}-{aug}-{align_type_str}.npy"
                            )
                            grid_path = (
                                save_dir
                                / f"grid_{i}-{mod1_str}_{j}-{mod2_str}-{aug}-{align_type_str}.npy"
                            )
                            if not os.path.exists(img_f_path):
                                print("Saving:", img_f_path)
                                np.save(img_f_path, img_f[0].cpu().detach().numpy())
                            if not os.path.exists(img_m_path):
                                print("Saving:", img_m_path)
                                np.save(img_m_path, img_m[0].cpu().detach().numpy())
                            print("Saving:", img_a_path)
                            np.save(img_a_path, img_a[0].cpu().detach().numpy())
                            if grid is not None:
                                print("Saving:", grid_path)
                                np.save(grid_path, grid[0].cpu().detach().numpy())
                            else:
                                print("Grid is None, not saving!")

                            # Save segmentations
                            if args.seg_available:
                                seg_f_path = save_dir / f"seg_f_{i}-{mod1_str}.npy"
                                seg_m_path = (
                                    save_dir / f"seg_m_{j}-{mod2_str}-{aug}.npy"
                                )
                                seg_a_path = (
                                    save_dir
                                    / f"seg_a_{i}-{mod1_str}_{j}-{mod2_str}-{aug}-{align_type_str}.npy"
                                )
                                if not os.path.exists(seg_f_path):
                                    print("Saving:", seg_f_path)
                                    np.save(
                                        seg_f_path,
                                        np.argmax(seg_f.cpu().detach().numpy(), axis=1),
                                    )
                                if not os.path.exists(seg_m_path):
                                    print("Saving:", seg_m_path)
                                    np.save(
                                        seg_m_path,
                                        np.argmax(seg_m.cpu().detach().numpy(), axis=1),
                                    )
                                print("Saving:", seg_a_path)
                                np.save(
                                    seg_a_path,
                                    np.argmax(seg_a.cpu().detach().numpy(), axis=1),
                                )

                            # Save points
                            if points_f is not None:
                                points_f_path = (
                                    save_dir / f"points_f_{i}-{mod1_str}.npy"
                                )
                                points_m_path = (
                                    save_dir / f"points_m_{j}-{mod2_str}-{aug}.npy"
                                )
                                points_a_path = (
                                    save_dir
                                    / f"points_a_{i}-{mod1_str}_{j}-{mod2_str}-{aug}-{align_type_str}.npy"
                                )
                                if not os.path.exists(points_f_path):
                                    print("Saving:", points_f_path)
                                    np.save(
                                        points_f_path,
                                        points_f[0].cpu().detach().numpy(),
                                    )
                                if not os.path.exists(points_m_path):
                                    print("Saving:", points_m_path)
                                    np.save(
                                        points_m_path,
                                        points_m[0].cpu().detach().numpy(),
                                    )
                                print("Saving:", points_a_path)
                                np.save(
                                    points_a_path,
                                    points_a[0].cpu().detach().numpy(),
                                )
                                if points_weights is not None:
                                    points_weights_path = (
                                        save_dir
                                        / f"points_weights_{i}-{mod1_str}_{j}-{mod2_str}-{aug}-{align_type_str}.npy"
                                    )
                                    print("Saving:", points_weights_path)
                                    np.save(
                                        points_weights_path,
                                        points_weights[0].cpu().detach().numpy(),
                                    )

                        # Print some stats
                        print("\nDebugging info:")
                        print(f'-> Time: {res_dict["time"]}')
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
                        # print(f"-> Full Results: {res_dict}")

                        print("\nMetrics:")
                        for metric_name, metric in metrics.items():
                            print(f"-> {metric_name}: {metric}")
    return test_metrics
