import os
import torch
import numpy as np
import torchio as tio
import time
from torch.profiler import profile, record_function, ProfilerActivity

from keymorph.utils import align_img, one_hot, one_hot_subsampled_pair
from keymorph.viz_tools import imshow_registration_2d, imshow_registration_3d
from keymorph.augmentation import random_affine_augment
import keymorph.loss_ops as loss_ops

from scripts.script_utils import aggregate_dicts


def run_train(train_loader, registration_model, optimizer, args):
    """Train for one epoch.

    Args:
        train_loader: Dataloader which returns pair of TorchIO subjects per iteration
        registration_model: Registration model
        optimizer: Pytorch optimizer
        args: Other script arguments
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    registration_model.train()

    res = []

    transform_type = args.transform_type
    loss_fn = args.loss_fn
    max_random_params = args.max_random_affine_augment_params

    for step_idx, subjects in enumerate(train_loader):
        fixed, moving = subjects
        if step_idx == args.steps_per_epoch:
            break

        # Get images and segmentations from TorchIO subject
        img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
        aff_f, aff_m = fixed["img"]["affine"], moving["img"]["affine"]
        if np.prod(img_f.shape) >= 77594624:
            print("Skipping large image")
            continue
        if np.prod(img_m.shape) >= 77594624:
            print("Skipping large image")
            continue
        if args.seg_available:
            seg_f, seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]
            # One-hot encode segmentations
            if args.max_train_seg_channels is not None:
                seg_f, seg_m = one_hot_subsampled_pair(
                    seg_f.long(), seg_m.long(), args.max_train_seg_channels
                )
            else:
                seg_f = one_hot(seg_f.long())
                seg_m = one_hot(seg_m.long())

        assert (
            img_f.shape[1] == 1
        ), f"Fixed image must have 1 channel:\n --> {fixed['img']['path']}: {img_f.shape}"
        assert (
            img_m.shape[1] == 1
        ), f"Moving image must have 1 channel:\n--> {moving['img']['path']}: {img_m.shape}"

        # Move to device
        img_f = img_f.float().to(args.device)
        img_m = img_m.float().to(args.device)
        aff_f = aff_f.float().to(args.device)
        aff_m = aff_m.float().to(args.device)
        if args.seg_available:
            seg_f = seg_f.float().to(args.device)
            seg_m = seg_m.float().to(args.device)

        # Explicitly augment moving image
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1
        if args.seg_available:
            img_m, seg_m, aug_affine = random_affine_augment(
                img_m,
                seg=seg_m,
                max_random_params=max_random_params,
                scale_params=scale_augment,
                return_affine_matrix=True,
            )
        else:
            img_m, aug_affine = random_affine_augment(
                img_m,
                max_random_params=max_random_params,
                scale_params=scale_augment,
                return_affine_matrix=True,
            )
        # New moving affine matrix is the composition of the original affine matrix and the augmentation matrix
        aff_m = torch.bmm(aff_m, aug_affine)

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
                            aff_f=aff_f,
                            aff_m=aff_m,
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
                    aff_f=aff_f,
                    aff_m=aff_m,
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
            print(f"-> Alignment: {align_type} ")
            print(f"-> Max random params: {max_random_params} ")
            print(f"-> TPS lambda: {tps_lmbda} ")
            print(f"-> Loss: {loss_fn}")
            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
            print(f"-> Point shapes: {points_f.shape}, {points_m.shape}")
            # print(f"-> Point weights: {points_weights}")
            print(f"-> Float16: {args.use_amp}")
            if args.seg_available:
                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

        if args.visualize:
            if args.dim == 2:
                imshow_registration_2d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                )
                if args.seg_available:
                    imshow_registration_2d(
                        seg_m[0, 0].cpu().detach().numpy(),
                        seg_f[0, 0].cpu().detach().numpy(),
                        seg_a[0, 0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(),
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                    )
            else:
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    projection=True,
                    resize=(256, 256, 256),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_256x256x256_{args.curr_epoch}.png"
                        )
                    ),
                )
                if args.seg_available:
                    imshow_registration_3d(
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

    return aggregate_dicts(res)


# Custom function to resize images while ensuring correct type
def resize_subject(subject, target_shape):
    resized_subject = {}
    for key, image in subject.items():
        if isinstance(image, tio.ScalarImage):
            # Ensure type is set correctly before copying
            image.type = tio.INTENSITY
        elif isinstance(image, tio.LabelMap):
            # Ensure type is set correctly before copying
            image.type = tio.LABEL
        else:
            continue  # Skip any non-image items

        # Create a new image with the resized data
        resized_image = tio.Resize(target_shape)(image)
        resized_subject[key] = resized_image

    return tio.Subject(resized_subject)


def run_train_sameres(train_loader, registration_model, optimizer, args):
    import torch.nn.functional as F

    def resize_tensor(tensor, target_shape, mode="trilinear"):
        return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False)

    """Train for one epoch. Images are reshaped to be of same resolution. Metrics are computed w.r.t. original fixed resolution.

    Args:
        train_loader: Dataloader which returns pair of TorchIO subjects per iteration
        registration_model: Registration model
        optimizer: Pytorch optimizer
        args: Other script arguments
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    registration_model.train()

    res = []

    transform_type = args.transform_type
    loss_fn = args.loss_fn
    max_random_params = args.max_random_affine_augment_params

    for step_idx, subjects in enumerate(train_loader):
        fixed, moving = subjects
        if step_idx == args.steps_per_epoch:
            break

        # Save original fixed shape
        orig_fixed_shape = fixed["img"][tio.DATA].shape[2:]

        # Get images and segmentations from TorchIO subject
        orig_img_f, orig_img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
        aff_f, aff_m = fixed["img"]["affine"], moving["img"]["affine"]
        if np.prod(orig_img_f.shape) >= 77594624:
            print("Skipping large image")
            continue
        if np.prod(orig_img_m.shape) >= 77594624:
            print("Skipping large image")
            continue
        if args.seg_available:
            orig_seg_f, orig_seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]
            # One-hot encode segmentations
            if args.max_train_seg_channels is not None:
                orig_seg_f, orig_seg_m = one_hot_subsampled_pair(
                    orig_seg_f.long(), orig_seg_m.long(), args.max_train_seg_channels
                )
            else:
                orig_seg_f = one_hot(orig_seg_f.long())
                orig_seg_m = one_hot(orig_seg_m.long())

        assert (
            orig_img_f.shape[1] == 1
        ), f"Fixed image must have 1 channel:\n --> {fixed['img']['path']}: {img_f.shape}"
        assert (
            orig_img_m.shape[1] == 1
        ), f"Moving image must have 1 channel:\n--> {moving['img']['path']}: {img_m.shape}"

        # Move to device
        orig_img_f = orig_img_f.float().to(args.device)
        orig_img_m = orig_img_m.float().to(args.device)
        aff_f = aff_f.float().to(args.device)
        aff_m = aff_m.float().to(args.device)
        if args.seg_available:
            orig_seg_f = orig_seg_f.float().to(args.device)
            orig_seg_m = orig_seg_m.float().to(args.device)

        # Reshape images to be of same resolution
        img_f = resize_tensor(orig_img_f, (256, 256, 256))
        img_m = resize_tensor(orig_img_m, (256, 256, 256))
        if args.seg_available:
            seg_f = resize_tensor(orig_seg_f, (256, 256, 256), mode="nearest")
            seg_m = resize_tensor(orig_seg_m, (256, 256, 256), mode="nearest")

        # Explicitly augment moving image
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1
        if args.seg_available:
            img_m, seg_m, aug_affine = random_affine_augment(
                img_m,
                seg=seg_m,
                max_random_params=max_random_params,
                scale_params=scale_augment,
                return_affine_matrix=True,
            )
        else:
            img_m, aug_affine = random_affine_augment(
                img_m,
                max_random_params=max_random_params,
                scale_params=scale_augment,
                return_affine_matrix=True,
            )
        # New moving affine matrix is the composition of the original affine matrix and the augmentation matrix
        aff_m = torch.bmm(aff_m, aug_affine)

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
                            aff_f=aff_f,
                            aff_m=aff_m,
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
                    aff_f=aff_f,
                    aff_m=aff_m,
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

            # Reshape back to original resolution
            orig_img_a = resize_tensor(img_a, orig_fixed_shape)
            if args.seg_available:
                orig_seg_a = resize_tensor(seg_a, orig_fixed_shape, mode="nearest")

            # Compute metrics
            metrics = {}
            metrics["scale_augment"] = scale_augment
            metrics["mse"] = loss_ops.MSELoss()(orig_img_f, orig_img_a)
            if args.seg_available:
                metrics["softdiceloss"] = loss_ops.DiceLoss()(orig_seg_a, orig_seg_f)
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
            print(f"-> Alignment: {align_type} ")
            print(f"-> Max random params: {max_random_params} ")
            print(f"-> TPS lambda: {tps_lmbda} ")
            print(f"-> Loss: {loss_fn}")
            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
            print(f"-> Point shapes: {points_f.shape}, {points_m.shape}")
            # print(f"-> Point weights: {points_weights}")
            print(f"-> Float16: {args.use_amp}")
            if args.seg_available:
                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

        if args.visualize:
            if args.dim == 2:
                imshow_registration_2d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                )
                if args.seg_available:
                    imshow_registration_2d(
                        seg_m[0, 0].cpu().detach().numpy(),
                        seg_f[0, 0].cpu().detach().numpy(),
                        seg_a[0, 0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(),
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                    )
            else:
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(),
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    projection=True,
                    resize=(256, 256, 256),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_256x256x256_{args.curr_epoch}.png"
                        )
                    ),
                )
                if args.seg_available:
                    imshow_registration_3d(
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

    return aggregate_dicts(res)
