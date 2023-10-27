import os
import subprocess
from pathlib import Path
import torchio as tio
from keymorph import utils
import numpy as np


def _reorient(src_img_dir, src_seg_dir, tgt_img_dir, tgt_seg_dir):
    """Reorient to MNI space using fslreorient2std"""
    print(
        f"Reorienting:\n {src_img_dir}, {src_seg_dir} -> {tgt_img_dir}, {tgt_seg_dir}"
    )

    # First reorient
    for i, basename in enumerate(sorted(os.listdir(src_img_dir))):
        name = basename.split(".")[0]
        s = os.path.join(src_img_dir, basename)
        t = os.path.join(tgt_img_dir, name)
        reorient_command = f"fslreorient2std {s} {t}"
        print("running command:\n", reorient_command)
        subprocess.run(reorient_command, shell=True)

        if i > 3:
            break

    for i, basename in enumerate(sorted(os.listdir(src_seg_dir))):
        name = basename.split(".")[0]
        s = os.path.join(src_seg_dir, basename)
        t = os.path.join(tgt_seg_dir, name)
        reorient_command = f"fslreorient2std {s} {t}"
        print("running command:\n", reorient_command)
        subprocess.run(reorient_command, shell=True)

        if i > 3:
            break
    print("finished")


def _hdbet(src_img_dir, src_seg_dir, tgt_img_dir, tgt_seg_dir, need_skullstrip=True):
    """Run HD-Bet brain extraction"""
    print(f"HD-BET:\n {src_img_dir}, {src_seg_dir} -> {tgt_img_dir}, {tgt_seg_dir}")

    if need_skullstrip:
        bet_command = f"hd-bet -i {src_img_dir} -o {tgt_img_dir}"
        subprocess.run(bet_command, shell=True)
    else:
        cp_command = (
            f"cp -avr {src_img_dir} {'/'.join(str(tgt_img_dir).split('/')[:-1])}"
        )
        subprocess.run(cp_command, shell=True)

    # Since segmentations don't need to be extracted, just copy it over
    cp_command = f"cp -avr {src_seg_dir} {'/'.join(str(tgt_seg_dir).split('/')[:-1])}"
    subprocess.run(cp_command, shell=True)
    print("finished")


def _torchio(src_img_dir, src_seg_dir, tgt_dir):
    """To canonical, resample to 1mm isotropic, crop/pad to 256^3, and intensity normalize to [0, 1]."""
    print(f"Running torchio transforms:\n {src_img_dir}, {src_seg_dir} -> {tgt_dir}")

    tio_transform = tio.Compose(
        [
            tio.ToCanonical(),
            tio.Resample(1),
            tio.CropOrPad((256, 256, 256), padding_mode=0, include=("img")),
            tio.CropOrPad((256, 256, 256), padding_mode=0, include=("seg")),
            tio.Lambda(utils.rescale_intensity, include=("img")),
        ]
    )

    img_data_paths = [os.path.join(src_img_dir, f) for f in os.listdir(src_img_dir)]
    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        path_name = basename.split(".")[0]
        extension = ".".join(basename.split(".")[1:])
        seg_path = os.path.join(src_seg_dir, path_name[:-5] + "." + extension)
        print("\nhere...\n")
        subject_kwargs = {
            "img": tio.ScalarImage(img_path),
            "seg": tio.LabelMap(seg_path),
        }
        try:
            subject = tio.Subject(**subject_kwargs)
            subject = tio_transform(subject)
            np.savez(
                os.path.join(tgt_dir, path_name + ".npz"),
                img=subject["img"]["data"],
                seg=subject["seg"]["data"],
            )
        except Exception as e:
            print("Error processing:", img_path)
    print("finished")


def main():
    base_dir = Path("/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base")
    reorient_dir = Path("/midtier/sablab/scratch/alw4013/nnUNet_MNI_preprocessed")
    bet_dir = Path("/midtier/sablab/scratch/alw4013/nnUNet_MNI_HD-BET_preprocessed")
    torchio_dir = Path(
        "/midtier/sablab/scratch/alw4013/nnUNet_MNI_HD-BET_canonical_1mmiso_256x256x256_preprocessed"
    )
    dataset_names = [
        "Dataset5000_BraTS-GLI_2023",
        "Dataset5001_BraTS-SSA_2023",
        "Dataset5002_BraTS-MEN_2023",
        "Dataset5003_BraTS-MET_2023",
        "Dataset5004_BraTS-MET-NYU_2023",
        "Dataset5005_BraTS-PED_2023",
        "Dataset5006_BraTS-MET-UCSF_2023",
        "Dataset5007_UCSF-BMSR",
        "Dataset5010_ATLASR2",
        "Dataset5011_BONBID-HIE_2023",
        "Dataset5012_ShiftsBest",
        "Dataset5013_ShiftsLjubljana",
        "Dataset5018_TopCoWMRAwholeBIN",
        "Dataset5024_TopCoWcrownMRAwholeBIN",
        "Dataset5038_BrainTumour",
        "Dataset5041_BRATS",
        "Dataset5042_BRATS2016",
        "Dataset5043_BrainDevelopment",
        "Dataset5044_EPISURG",
        "Dataset5046_FeTA",
        "Dataset5066_WMH",
        # "Dataset5083_IXIT1",
        # "Dataset5084_IXIT2",
        # "Dataset5085_IXIPD",
        "Dataset5090_ISLES2022",
        "Dataset5094_crossmoda2022",
        "Dataset5095_MSSEG",
        "Dataset5096_MSSEG2",
        "Dataset5111_UCSF-ALPTDG-time1",
        "Dataset5112_UCSF-ALPTDG-time2",
        "Dataset5113_StanfordMETShare",
    ]
    need_skullstrip = [
        "Dataset5010_ATLASR2",
        "Dataset5018_TopCoWMRAwholeBIN",
        "Dataset5024_TopCoWcrownMRAwholeBIN",
        "Dataset5043_BrainDevelopment",
        "Dataset5044_EPISURG",
        "Dataset5066_WMH",
        "Dataset5094_crossmoda2022",
        "Dataset5096_MSSEG2",
    ]

    for ds in dataset_names:
        # fslreorient2std
        # Train
        ds_src_img_dir = base_dir / ds / "imagesTr"
        ds_src_seg_dir = base_dir / ds / "labelsTr"
        ds_tgt_img_dir = reorient_dir / ds / "imagesTr"
        ds_tgt_seg_dir = reorient_dir / ds / "labelsTr"

        if not os.path.exists(ds_tgt_img_dir):
            os.makedirs(ds_tgt_img_dir)
        if not os.path.exists(ds_tgt_seg_dir):
            os.makedirs(ds_tgt_seg_dir)
        _reorient(
            ds_src_img_dir,
            ds_src_seg_dir,
            ds_tgt_img_dir,
            ds_tgt_seg_dir,
        )

        # Test
        # ds_src_img_dir = base_dir / ds / "imagesTs"
        # ds_src_seg_dir = base_dir / ds / "labelsTs"
        # ds_tgt_img_dir = reorient_dir / ds / "imagesTs"
        # ds_tgt_seg_dir = reorient_dir / ds / "labelsTs"

        # if not os.path.exists(ds_tgt_img_dir):
        #     os.makedirs(ds_tgt_img_dir)
        # if not os.path.exists(ds_tgt_seg_dir):
        #     os.makedirs(ds_tgt_seg_dir)
        # _reorient(
        #     ds_src_img_dir,
        #     ds_src_seg_dir,
        #     ds_tgt_img_dir,
        #     ds_tgt_seg_dir,
        # )

        # HD-BET
        ds_src_img_dir = reorient_dir / ds / "imagesTr"
        ds_src_seg_dir = reorient_dir / ds / "labelsTr"
        ds_tgt_img_dir = bet_dir / ds / "imagesTr"
        ds_tgt_seg_dir = bet_dir / ds / "labelsTr"

        if not os.path.exists(ds_tgt_img_dir):
            os.makedirs(ds_tgt_img_dir)
        if not os.path.exists(ds_tgt_seg_dir):
            os.makedirs(ds_tgt_seg_dir)
        _hdbet(
            ds_src_img_dir,
            ds_src_seg_dir,
            ds_tgt_img_dir,
            ds_tgt_seg_dir,
            need_skullstrip=ds in need_skullstrip,
        )

        # Test
        # ds_src_img_dir = reorient_dir / ds / "imagesTs"
        # ds_src_seg_dir = reorient_dir / ds / "labelsTs"
        # ds_tgt_img_dir = bet_dir / ds / "imagesTs"
        # ds_tgt_seg_dir = bet_dir / ds / "labelsTs"

        # if not os.path.exists(ds_tgt_img_dir):
        #     os.makedirs(ds_tgt_img_dir)
        # if not os.path.exists(ds_tgt_seg_dir):
        #     os.makedirs(ds_tgt_seg_dir)
        # _hdbet(
        #     ds_src_img_dir,
        #     ds_src_seg_dir,
        #     ds_tgt_img_dir,
        #     ds_tgt_seg_dir,
        #     need_skullstrip=ds in need_skullstrip,
        # )

        # TorchIO
        # Train
        ds_src_img_dir = bet_dir / ds / "imagesTr"
        ds_src_seg_dir = bet_dir / ds / "labelsTr"
        ds_tgt_dir = torchio_dir / ds

        if not os.path.exists(ds_tgt_dir):
            os.makedirs(ds_tgt_dir)
        _torchio(
            ds_src_img_dir,
            ds_src_seg_dir,
            ds_tgt_dir,
        )

        # Test
        # ds_src_img_dir = bet_dir / ds / "imagesTs"
        # ds_src_seg_dir = bet_dir / ds / "labelsTs"
        # ds_tgt_dir = torchio_dir / ds

        # if not os.path.exists(ds_tgt_dir):
        #     os.makedirs(ds_tgt_dir)
        # _torchio(
        #     ds_src_img_dir,
        #     ds_src_seg_dir,
        #     ds_tgt_dir,
        # )


if __name__ == "__main__":
    main()
