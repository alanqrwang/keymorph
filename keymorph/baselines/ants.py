import subprocess
import torch
import ants
import time
import torch.nn.functional as F
import numpy as np


class ANTs:
    def __init__(self):
        super().__init__()

    def eval(self):
        pass

    def __call__(self, fixed, moving, transform_type="rigid", **kwargs):
        return self.pairwise_register(fixed, moving, transform_type, **kwargs)

    def pairwise_register(self, img_f, img_m, transform_type="rigid", **kwargs):
        original_device = img_f.device
        assert len(img_f) == 1, "Fixed image should be a single image"
        assert len(img_m) == 1, "Moving image should be a single image"

        seg_m = kwargs["seg_m"].argmax(1).float()

        # Dictionary of results
        result_dict = {}

        for ttype in transform_type:
            print(ttype)
            if ttype == "rigid":
                type_of_transform = "Rigid"
            elif ttype == "affine":
                type_of_transform = "AffineFast"
            elif ttype == "syn":
                type_of_transform = "SyNRA"
            else:
                raise ValueError("Invalid transform type")
            start_time = time.time()
            moving = ants.from_numpy(img_m[0, 0].cpu().detach().numpy())
            fixed = ants.from_numpy(img_f[0, 0].cpu().detach().numpy())
            seg_moving = ants.from_numpy(seg_m[0].cpu().detach().numpy())

            mytx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform=type_of_transform,
                verbose=False,
            )

            moved = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                interpolator="linear",
                transformlist=mytx["fwdtransforms"],
            )
            seg_moved = ants.apply_transforms(
                fixed=fixed,
                moving=seg_moving,
                interpolator="nearestNeighbor",
                transformlist=mytx["fwdtransforms"],
            )

            if ttype == "syn":
                jac = ants.create_jacobian_determinant_image(
                    moved, mytx["fwdtransforms"][0]
                )
                jdstd = jac.std().item()
                jdlessthan0 = np.count_nonzero(jac <= 0) / len(jac.flatten())
            else:
                jdstd = 0.0
                jdlessthan0 = 0.0

            img_a = torch.tensor(moved.numpy())[None, None].float().to(original_device)
            seg_a = torch.tensor(seg_moved.numpy()).long()
            seg_a = (
                F.one_hot(seg_a).permute(3, 0, 1, 2)[None].float().to(original_device)
            )

            register_time = time.time() - start_time

            res = {
                "align_type": ttype,
                "img_a": img_a,
                "seg_a": seg_a,
                "time": register_time,
                "jdstd": jdstd,
                "jdlessthan0": jdlessthan0,
            }
            result_dict[ttype] = res

        return result_dict

    def groupwise_register(self, inputs, transform_type="rigid", **kwargs):
        raise NotImplementedError


def shell_command(command):
    print("RUNNING", command)
    subprocess.run(command, shell=True)
