import os
import torch
import numpy as np
import torchio as tio
import time
import torch.nn.functional as F

from keymorph import utils
from keymorph.augmentation import random_affine_augment
from keymorph.viz_tools import imshow_registration_2d, imshow_registration_3d


def run_pretrain(loader, random_points, keymorph_model, optimizer, args):
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    keymorph_model.train()

    res = []

    random_points = random_points.to(args.device)
    for step_idx, (subject, _) in enumerate(loader):
        if step_idx == args.steps_per_epoch:
            break
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
                pred_points = keymorph_model.get_keypoints(x_moving)
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
                imshow_registration_2d(
                    x_moving[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    tgt_points[0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                    pred_points[0].cpu().detach().numpy(),
                )
            else:
                imshow_registration_3d(
                    x_moving[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    x_fixed[0, 0].cpu().detach().numpy(),
                    tgt_points[0].cpu().detach().numpy(),
                    random_points[0].cpu().detach().numpy(),
                    pred_points[0].cpu().detach().numpy(),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )

    return utils.aggregate_dicts(res)
