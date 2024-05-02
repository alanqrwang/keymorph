import os
import torch
import numpy as np
import torchio as tio
import time
from torch.profiler import profile, record_function, ProfilerActivity

import keymorph.utils as utils
from keymorph.utils import align_img
from keymorph.cm_plotter import show_warped, show_warped_vol
from keymorph.augmentation import random_affine_augment
import keymorph.loss_ops as loss_ops


def run_train(train_loader, registration_model, optimizer, train_params, args):
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
            if args.use_profiler:
                with profile(
                    enabled=args.use_profiler,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_stack=True,
                    profile_memory=True,
                    experimental_config=torch._C._profiler._ExperimentalConfig(
                        verbose=True
                    ),
                ) as prof:
                    with record_function("model_inference"):
                        registration_results = registration_model(
                            img_f,
                            img_m,
                            transform_type=transform_type,
                            return_aligned_points=args.visualize,
                        )[transform_type]
                print(
                    prof.key_averages(group_by_stack_n=5).table(
                        sort_by="self_cuda_memory_usage"
                    )
                )
            else:
                registration_results = registration_model(
                    img_f,
                    img_m,
                    transform_type=transform_type,
                    return_aligned_points=args.visualize,
                )[transform_type]
            grid = registration_results["grid"]
            align_type = transform_type
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
