import os
import torch
import numpy as np
import torchio as tio
import shutil
import matplotlib.pyplot as plt

from keymorph.utils import align_img, one_hot
from keymorph.augmentation import random_affine_augment
import keymorph.loss_ops as loss_ops

from scripts.script_utils import (
    save_dict_as_json,
    parse_test_aug,
)


def run_long_eval(
    group_loader,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_kp_aligns,
    args,
    save_dir_prefix="long_eval",
    duplicate_files=False,
):
    """Longitudinal evaluation."""

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

    test_metrics = _build_metric_dict(list_of_eval_names)
    for dataset_name in list_of_eval_names:
        for aug in list_of_eval_augs:
            if not args.debug_mode:
                # Create directory to save images, segs, points, metrics
                dataset_name_str = "-".join(dataset_name.split("/")[-2:])
                group_dir = (
                    args.model_eval_dir / save_dir_prefix / f"{dataset_name_str}_{aug}"
                )
                groupimg_m_dir = os.path.join(group_dir, "img_m")
                groupseg_m_dir = os.path.join(group_dir, "seg_m")
                if not os.path.exists(groupimg_m_dir):
                    os.makedirs(groupimg_m_dir)
                if not os.path.exists(groupseg_m_dir):
                    os.makedirs(groupseg_m_dir)

            aug_params = parse_test_aug(aug)
            for i, group in enumerate(group_loader[dataset_name]):
                if args.early_stop_eval_subjects and i == args.early_stop_eval_subjects:
                    break
                print(
                    f"Running longitudinal test: group id {i}, dataset {dataset_name}, aug {aug}"
                )
                print("Number of longitudinal images:", len(group))
                for sub_id, subject in enumerate(group):
                    img_m = subject["img"][tio.DATA].float().unsqueeze(0)
                    aff_m = subject["img"]["affine"]
                    if args.seg_available:
                        seg_m = subject["seg"][tio.DATA].float().unsqueeze(0)
                        # One-hot encode segmentations
                        seg_m = one_hot(seg_m.long()).float()

                    # Randomly affine augment all images
                    if aug_params is not None:
                        if args.seg_available:
                            img_m, seg_m = random_affine_augment(
                                img_m, max_random_params=aug_params, seg=seg_m
                            )
                        else:
                            img_m = random_affine_augment(
                                img_m, max_random_params=aug_params
                            )

                    # FIXME: change aff_m matrix accordingly

                    # Save subject to group directory
                    if not args.debug_mode:
                        img_m_path = os.path.join(
                            groupimg_m_dir, f"img_m_{sub_id:03}.npz"
                        )
                        np.savez(img_m_path, img=img_m, aff=aff_m)
                        print("saving:", img_m_path)
                        if args.seg_available:
                            seg_m_path = os.path.join(
                                groupseg_m_dir, f"seg_m_{sub_id:03}.npz"
                            )
                            np.savez(seg_m_path, seg=seg_m, aff=aff_m)
                            print("saving:", seg_m_path)

                # Run groupwise registration on group_dir
                registration_results = _run_group_eval_dir(
                    group_dir,
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_kp_aligns,
                    aug,
                    args,
                    duplicate_files=duplicate_files,
                )

                for align_type_str, res_dict in registration_results.items():
                    for m in list_of_eval_metrics:
                        if m == "mse":
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["mse"])
                        elif m == "softdice":
                            assert args.seg_available
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["softdice"])
                        elif m == "harddice":
                            assert args.seg_available
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["harddice"])
                        elif m == "harddiceroi":
                            assert args.seg_available
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["harddiceroi"])
                        elif m == "hausd":
                            assert args.seg_available and args.dim == 3
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["hausd"])
                        elif m == "jdstd":
                            assert args.dim == 3
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["jdstd"])
                        elif m == "jdlessthan0":
                            assert args.dim == 3
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}"
                            ].append(res_dict["metrics"]["jdlessthan0"])
                        else:
                            raise ValueError('Invalid metric "{}"'.format(m))
    return test_metrics


def run_group_eval(
    group_loader,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_kp_aligns,
    list_of_group_sizes,
    args,
    save_dir_prefix="group_eval",
    duplicate_files=False,
):
    """Group evaluation. Since group size can be large, we cannot load
    all images at once. Instead, we load images one by one, augment
    them as necessary, and save them into a separate directory.
    Every registration model has a groupwise_register method that
    takes in the directory."""

    def _build_metric_dict(dataset_names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            for a in list_of_eval_augs:
                for k in list_of_eval_kp_aligns:
                    for n in dataset_names:
                        for g in list_of_group_sizes:
                            list_of_all_test.append(f"{m}:{n}:{a}:{k}:{g}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    test_metrics = _build_metric_dict(list_of_eval_names)
    assert args.save_eval_to_disk and args.batch_size == 1  # Must save to disk
    for dataset_name in list_of_eval_names:
        for aug in list_of_eval_augs:
            for group_size in list_of_group_sizes:
                if not args.debug_mode:
                    # Create directory to save images, segs, points, metrics
                    dataset_name_str = "-".join(dataset_name.split("/")[-2:])
                    group_dir = (
                        args.model_eval_dir
                        / save_dir_prefix
                        / f"{dataset_name_str}_{aug}_{group_size}"
                    )
                    groupimg_m_dir = os.path.join(group_dir, "img_m")
                    groupseg_m_dir = os.path.join(group_dir, "seg_m")
                    if not os.path.exists(groupimg_m_dir):
                        os.makedirs(groupimg_m_dir)
                    if not os.path.exists(groupseg_m_dir):
                        os.makedirs(groupseg_m_dir)

                aug_params = parse_test_aug(aug)
                print(
                    f"Running groupwise test: dataset {dataset_name}, aug {aug}, group size {group_size}"
                )
                for i, subject in enumerate(group_loader[dataset_name]):
                    if i == group_size:
                        break

                    img_m = subject["img"][tio.DATA].float()
                    aff_m = subject["img"]["affine"]
                    if args.seg_available:
                        seg_m = subject["seg"][tio.DATA].float()
                        # One-hot encode segmentations
                        seg_m = one_hot(seg_m.long()).float()

                    # Randomly affine augment all images
                    if aug_params is not None:
                        if args.seg_available:
                            img_m, seg_m = random_affine_augment(
                                img_m, max_random_params=aug_params, seg=seg_m
                            )
                        else:
                            img_m = random_affine_augment(
                                img_m, max_random_params=aug_params
                            )

                    # FIXME: change aff_m matrix accordingly

                    # Save subject to group directory
                    if not args.debug_mode:
                        img_m_path = os.path.join(groupimg_m_dir, f"img_m_{i:03}.npz")
                        np.savez(img_m_path, img=img_m, aff=aff_m)
                        print("saving:", img_m_path)
                        if args.seg_available:
                            seg_m_path = os.path.join(
                                groupseg_m_dir, f"seg_m_{i:03}.npz"
                            )
                            np.savez(seg_m_path, seg=seg_m, aff=aff_m)
                            print("saving:", seg_m_path)

                # Run groupwise registration on group_dir
                registration_results = _run_group_eval_dir(
                    group_dir,
                    registration_model,
                    list_of_eval_metrics,
                    list_of_eval_kp_aligns,
                    aug,
                    args,
                    duplicate_files=duplicate_files,
                )

                for align_type_str, res_dict in registration_results.items():
                    for m in list_of_eval_metrics:
                        if m == "mse":
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["mse"])
                        elif m == "softdice":
                            assert args.seg_available
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["softdice"])
                        elif m == "harddice":
                            assert args.seg_available
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["harddice"])
                        elif m == "harddiceroi":
                            assert args.seg_available
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["harddiceroi"])
                        elif m == "hausd":
                            assert args.seg_available and args.dim == 3
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["hausd"])
                        elif m == "jdstd":
                            assert args.dim == 3
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["jdstd"])
                        elif m == "jdlessthan0":
                            assert args.dim == 3
                            test_metrics[
                                f"{m}:{dataset_name}:{aug}:{align_type_str}:{group_size}"
                            ].append(res_dict["metrics"]["jdlessthan0"])
                        else:
                            raise ValueError('Invalid metric "{}"'.format(m))
    return test_metrics


def _run_group_eval_dir(
    group_dir,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_kp_aligns,
    aug,
    args,
    duplicate_files=False,
):
    """Takes a directory of images and segmentations, and runs groupwise registration on them.
    In group_dir, assumes img_m/ and seg_m/. Creates img_a/, seg_a/, and registration_results/.
    """

    def _construct_template(list_of_imgs, aggr="mean"):
        if aggr == "mean":
            template = torch.mean(torch.stack(list_of_imgs), dim=0)
        elif aggr == "majority":
            template, _ = torch.mode(torch.stack(list_of_imgs), dim=0)
        else:
            raise ValueError('Invalid aggr "{}"'.format(aggr))
        return template

    def _duplicate_files_to_N(directory, N=4):
        # Get a list of files in the directory
        files = sorted(
            [
                f
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
        )

        # Check if the number of files is less than 4
        if len(files) < 4:
            # Determine the path of the first file
            first_file_path = os.path.join(directory, files[0])

            # Duplicate the first file until there are 4 files in total
            while len(files) < 4:
                new_file_path = os.path.join(
                    directory, f"{files[0][:3]}_m_{len(files):03}.npz"
                )
                shutil.copy(first_file_path, new_file_path)
                files.append(new_file_path)  # Update the files list

                print(
                    f"Created: {new_file_path}"
                )  # Optional: print the name of the created file

    groupimg_m_dir = os.path.join(group_dir, "img_m")
    groupseg_m_dir = os.path.join(group_dir, "seg_m")
    groupimg_a_dir = {
        align_type_str: os.path.join(group_dir, f"img_a_{align_type_str}")
        for align_type_str in list_of_eval_kp_aligns
    }
    groupseg_a_dir = {
        align_type_str: os.path.join(group_dir, f"seg_a_{align_type_str}")
        for align_type_str in list_of_eval_kp_aligns
    }
    registration_results_dir = os.path.join(group_dir, "registration_results")
    for subdir in groupimg_a_dir.values():
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    for subdir in groupseg_a_dir.values():
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    if not os.path.exists(registration_results_dir):
        os.makedirs(registration_results_dir)
    # If number of files in group directory is less than 4, duplicate the first image to make 4.
    # This is because some groupwise registration packages require
    # at least 4 images.
    if not args.debug_mode and duplicate_files:
        _duplicate_files_to_N(groupimg_m_dir, N=4)
        if args.seg_available:
            _duplicate_files_to_N(groupseg_m_dir, N=4)

    groupimg_m_paths = sorted(
        [os.path.join(groupimg_m_dir, f) for f in os.listdir(groupimg_m_dir)]
    )
    groupseg_m_paths = sorted(
        [os.path.join(groupseg_m_dir, f) for f in os.listdir(groupseg_m_dir)]
    )

    with torch.set_grad_enabled(False):
        registration_results = registration_model.groupwise_register(
            groupimg_m_dir,
            transform_type=list_of_eval_kp_aligns,
            device=args.device,
            save_results_to_disk=True,
            save_dir=registration_results_dir,
            plot=args.visualize,
            num_iters=5,
            log_to_console=True,
            num_resolutions_for_itkelastix=args.num_resolutions_for_itkelastix,
        )
        # registration_results = {"bspline": {}}

    print("\nComputing metrics and saving results...")
    for align_type_str, res_dict in registration_results.items():
        print(f"\n...for {align_type_str}")
        groupimg_a_paths = []
        groupseg_a_paths = []
        # Get all grid paths stored in registration_results directory
        groupgrids_paths = sorted(
            [
                os.path.join(registration_results_dir, f)
                for f in os.listdir(registration_results_dir)
                if f.startswith(align_type_str)
            ]
        )

        # Align images and construct template image
        for i in range(len(groupimg_m_paths)):
            img_m = torch.tensor(np.load(groupimg_m_paths[i])["img"]).to(args.device)
            grid = torch.tensor(np.load(groupgrids_paths[i])).to(args.device)
            img_a = align_img(grid, img_m)
            img_a_path = os.path.join(
                groupimg_a_dir[align_type_str], f"img_a_{align_type_str}_{i:03}.npy"
            )
            np.save(
                img_a_path,
                img_a.cpu().detach().numpy(),
            )
            groupimg_a_paths.append(img_a_path)
            if args.seg_available:
                seg_m = torch.tensor(np.load(groupseg_m_paths[i])["seg"]).to(
                    args.device
                )
                seg_a = align_img(grid, seg_m)
                seg_a_path = os.path.join(
                    groupseg_a_dir[align_type_str], f"seg_a_{align_type_str}_{i:03}.npy"
                )
                np.save(
                    seg_a_path,
                    seg_a.cpu().detach().numpy(),
                )
                groupseg_a_paths.append(seg_a_path)

            if args.visualize:
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                [ax.axis("off") for ax in axes]
                axes[0].imshow(img_m[0, 0, 128].cpu().detach().numpy())
                axes[1].imshow(img_a[0, 0, 128].cpu().detach().numpy())
                if args.seg_available:
                    axes[2].imshow(seg_m.argmax(1)[0, 128].cpu().detach().numpy())
                    axes[3].imshow(seg_a.argmax(1)[0, 128].cpu().detach().numpy())
                axes[0].set_title("Moving")
                axes[1].set_title("Aligned")
                if args.seg_available:
                    axes[2].set_title("Moving Seg")
                    axes[3].set_title("Aligned Seg")
                fig.show()
                plt.show()

        grouppoints_m = (
            res_dict["grouppoints_m"] if "grouppoints_m" in res_dict else None
        )
        grouppoints_a = (
            res_dict["grouppoints_a"] if "grouppoints_a" in res_dict else None
        )
        # grouppoints_weights = (
        #     res_dict["points_weights"]
        #     if "points_weights" in res_dict
        #     else None
        # )

        # if args.visualize:
        #     plot_groupwise_register(
        #         [img[0, 128].cpu().detach().numpy() for img in groupimgs_m],
        #         [img[0, 128].cpu().detach().numpy() for img in groupimgs_a],
        #     )
        #     if args.seg_available:
        #         plot_groupwise_register(
        #             [
        #                 seg.argmax(0)[128].cpu().detach().numpy()
        #                 for seg in groupsegs_m
        #             ],
        #             [
        #                 seg.argmax(0)[128].cpu().detach().numpy()
        #                 for seg in groupsegs_a
        #             ],
        #         )

        metrics = {}
        img_metric_names, grid_metric_names = [], []
        for m in list_of_eval_metrics:
            if m == "mse":
                metrics["mse"] = loss_ops.MSEPairwiseLoss()(groupimg_a_paths).item()
            elif m == "softdice":
                assert args.seg_available
                img_metric_names.append("softdice")
            elif m == "harddice":
                assert args.seg_available
                img_metric_names.append("harddice")
            elif m == "harddiceroi":
                assert args.seg_available
                img_metric_names.append("harddiceroi")
            elif m == "hausd":
                assert args.seg_available and args.dim == 3
                img_metric_names.append("hausd")
            elif m == "jdstd":
                assert args.dim == 3
                grid_metric_names.append("jdstd")
            elif m == "jdlessthan0":
                assert args.dim == 3
                grid_metric_names.append("jdlessthan0")
            else:
                raise ValueError('Invalid metric "{}"'.format(m))

        seg_metrics = loss_ops.MultipleAvgSegPairwiseMetric()(
            groupseg_a_paths, img_metric_names
        )
        if "harddice" in seg_metrics:
            seg_metrics["harddice"] = (1 - seg_metrics["harddice"]).item()
        if "harddiceroi" in seg_metrics:
            seg_metrics["harddiceroi"] = (1 - seg_metrics["harddiceroi"]).tolist()
        if "softdice" in seg_metrics:
            seg_metrics["softdice"] = (1 - seg_metrics["softdice"]).item()
        grid_metrics = loss_ops.MultipleAvgGridMetric()(
            groupgrids_paths, grid_metric_names
        )

        metrics = metrics | seg_metrics | grid_metrics
        res_dict["metrics"] = metrics

        # Save registration results and metrics
        results_path = group_dir / "registration_results.pt"
        metrics_path = group_dir / f"metrics-{align_type_str}.json"
        if not os.path.exists(results_path):
            print("Saving:", results_path)
            torch.save(registration_results, results_path)
        print("Saving:", metrics_path)
        save_dict_as_json(metrics, metrics_path)

        # Save points
        if grouppoints_m is not None:
            grouppoints_m_path = group_dir / f"points_m-{aug}.npy"
            grouppoints_a_path = group_dir / f"points_a-{aug}-{align_type_str}.npy"
            if not os.path.exists(grouppoints_m_path):
                print("Saving:", grouppoints_m_path)
                np.save(
                    grouppoints_m_path,
                    grouppoints_m[0].cpu().detach().numpy(),
                )
            print("Saving:", grouppoints_a_path)
            np.save(
                grouppoints_a_path,
                grouppoints_a[0].cpu().detach().numpy(),
            )

        print("\nDebugging info:")
        print(f"-> Alignment: {align_type_str} ")
        # print(f"-> Max random params: {aug_params} ")
        print(f"-> Group size: {len(groupimg_m_paths)}")
        print(f"-> Float16: {args.use_amp}")

        print("\nMetrics:")
        for metric_name, metric in metrics.items():
            print(f"-> {metric_name}: {metric}")

    return registration_results
