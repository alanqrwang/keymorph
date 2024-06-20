import os
import torch
import numpy as np
import torchio as tio
import time
import torch.nn.functional as F

from keymorph import utils
from keymorph.augmentation import random_affine_augment
from keymorph.viz_tools import (
    imshow_registration_2d,
    imshow_registration_3d,
)


def run_pretrain(loader, ref_subject, keymorph_model, optimizer, args):
    """Run pretraining loop for a single epoch.

    Given reference points, pretraining samples a random subject in the training set and applies an
    affine augmentation to the subject's image and the reference points. The model then takes as input
    the augmented image and tries to predict the corresponding augmented reference points.
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    keymorph_model.train()

    max_random_params = args.max_random_affine_augment_params
    res = []

    ref_points = ref_subject["points"].to(args.device)
    for step_idx, subject in enumerate(loader):
        if step_idx == args.steps_per_epoch:
            break
        img_f = subject["img"][tio.DATA].float().to(args.device)
        aff_f = subject["img"]["affine"].float().to(args.device)
        shape_f = torch.tensor(img_f.shape[2:]).to(img_f)
        if np.prod(img_f.shape) >= 77594624:
            print("Skipping large image")
            continue

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1

        # Affine augment fixed image to get moving image and points
        augmented_img, tgt_points, aug_affine = random_affine_augment(
            img_f,
            points=ref_points,
            max_random_params=max_random_params,
            scale_params=scale_augment,
            return_affine_matrix=True,
        )
        # New target affine matrix is the composition of the original affine matrix and the augmentation matrix
        tgt_affine = torch.bmm(aff_f, aug_affine)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                points_a = keymorph_model.get_keypoints(augmented_img)
                points_a = points_a.view(-1, args.num_keypoints, args.dim)
                if args.align_keypoints_in_real_world_coords:
                    points_a = utils.convert_points_norm2real(points_a, aff_f, shape_f)
                loss = F.mse_loss(tgt_points, points_a)

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
            # Convert points back to normalized space
            tgt_voxel_shapes = (
                torch.stack([torch.tensor(im.shape[1:]) for im in augmented_img])
                .float()
                .to(args.device)
            )
            ref_points_viz = utils.convert_points_real2norm(
                ref_points, aff_f, tgt_voxel_shapes
            )
            tgt_points_viz = utils.convert_points_real2norm(
                tgt_points, tgt_affine, tgt_voxel_shapes
            )
            pred_points_viz = utils.convert_points_real2norm(
                points_a, tgt_affine, tgt_voxel_shapes
            )
            if args.dim == 2:
                imshow_registration_2d(
                    augmented_img[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    tgt_points_viz[0].cpu().detach().numpy(),
                    ref_points_viz[0].cpu().detach().numpy(),
                    pred_points_viz[0].cpu().detach().numpy(),
                )
            else:
                imshow_registration_3d(
                    img_f[0, 0].cpu().detach().numpy(),
                    augmented_img[0, 0].cpu().detach().numpy(),
                    augmented_img[0, 0].cpu().detach().numpy(),
                    ref_points_viz[0].cpu().detach().numpy(),
                    tgt_points_viz[0].cpu().detach().numpy(),
                    pred_points_viz[0].cpu().detach().numpy(),
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )

    return utils.aggregate_dicts(res)
