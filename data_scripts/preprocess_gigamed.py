import os
import subprocess
from pathlib import Path
import torchio as tio
from keymorph import utils
import numpy as np
from pprint import pprint
from argparse import ArgumentParser

# WARNING!!!!
# On AI cluster, load FSL and bc modules.
# Activate conda environment alw4013-hdbet.

need_skullstrip = [
    "Dataset4999_IXIAllModalities",
    "Dataset5010_ATLASR2",
    "Dataset5043_BrainDevelopment",
    "Dataset5044_EPISURG",
    "Dataset5066_WMH",
    "Dataset5083_IXIT1",
    "Dataset5084_IXIT2",
    "Dataset5085_IXIPD",
    "Dataset5096_MSSEG2",
    "NYU_METS",
    "Dataset7000_openneuro-ds004791",
    "Dataset7001_openneuro-ds004848",
]


def shell_command(command):
    print("RUNNING", command)
    subprocess.run(command, shell=True)


def _resample_pad_intensity_normalize(
    src_img_dir,
    src_seg_dir,
    tgt_img_dir,
    tgt_seg_dir,
    seg_available=True,
    min_max_norm=True,
):
    """Resample to 1mm isotropic and crop/pad to 256^3.
    If specified, also intensity normalize to [0, 1]."""
    failed = []

    transforms = [
        tio.ToCanonical(),
        tio.Resample(1),
        tio.Resample("img"),
        tio.CropOrPad((256, 256, 256), padding_mode=0, include=("img")),
        tio.CropOrPad((256, 256, 256), padding_mode=0, include=("seg")),
    ]
    if min_max_norm:
        transforms.append(tio.Lambda(utils.rescale_intensity, include=("img")))

    tio_transform = tio.Compose(transforms)

    img_data_paths = [
        os.path.join(src_img_dir, f) for f in os.listdir(src_img_dir) if "mask" not in f
    ]
    for img_path in img_data_paths:
        subject_kwargs = {
            "img": tio.ScalarImage(img_path),
        }
        basename = os.path.basename(img_path)
        path_name = basename.split(".")[0]
        if seg_available:
            extension = ".".join(basename.split(".")[1:])
            seg_path = os.path.join(src_seg_dir, path_name[:-5] + "." + extension)
            subject_kwargs["seg"] = tio.LabelMap(seg_path)
        try:
            subject = tio.Subject(**subject_kwargs)
            subject = tio_transform(subject)
            subject["img"].save(
                os.path.join(tgt_img_dir, path_name + ".nii.gz"),
            )
            if seg_available:
                subject["seg"].save(
                    os.path.join(tgt_seg_dir, path_name + ".nii.gz"),
                )
        except Exception as e:
            print("Error torchioing:", img_path)
            print(e)
            failed.append(img_path)
    return failed


def _reorient(src_dir, tgt_dir, skip=False, swapdim=None):
    """Reorient to MNI space using fslreorient2std"""
    failed = []
    if skip:  # Just copy over
        try:
            bet_command = f"cp -av {src_dir} {'/'.join(str(tgt_dir).split('/')[:-1])}"
            shell_command(bet_command)
        except Exception as e:
            print("Error reorienting:", src_dir)
            failed.append(src_dir)
    if swapdim is not None:
        for basename in sorted(os.listdir(src_dir)):
            name = basename.split(".")[0]
            s = os.path.join(src_dir, basename)
            t = os.path.join(tgt_dir, name)
            try:
                bet_command = f"fslswapdim {s} {swapdim} {t}"
                shell_command(bet_command)
            except Exception as e:
                print("Error reorienting:", src_dir)
                failed.append(src_dir)
    else:
        for basename in sorted(os.listdir(src_dir)):
            try:
                name = basename.split(".")[0]
                s = os.path.join(src_dir, basename)
                t = os.path.join(tgt_dir, name)
                reorient_command = f"fslreorient2std {s} {t}"
                shell_command(reorient_command)

            except Exception as e:
                print("Error reorienting:", src_dir)
                failed.append(src_dir)

    return failed


def _hdbet(src_img_dir, tgt_img_dir, need_skullstrip=True):
    """Run HD-Bet brain extraction"""
    failed = []

    try:
        if need_skullstrip:
            bet_command = f"hd-bet -i {src_img_dir} -o {tgt_img_dir}"
        else:
            bet_command = (
                f"cp -av {src_img_dir} {'/'.join(str(tgt_img_dir).split('/')[:-1])}"
            )
        shell_command(bet_command)
    except Exception as e:
        print("Error brain extracting:", src_img_dir)
        failed.append(src_img_dir)

    return failed


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--min_max_norm",
        action="store_true",
        help="If added, min-max normalize to [0, 1].",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If added, preprocess on test data. Else, train data.",
    )
    args = parser.parse_args()

    if args.test:
        print("Preprocessing test data.")
        image_dir, label_dir = "imagesTs", "labelsTs"
    else:
        print("Preprocessing train data.")
        image_dir, label_dir = "imagesTr", "labelsTr"

    all_failed = []
    base_dir = Path("/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base")

    if args.min_max_norm:
        torchio_dir = Path(
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_preprocessed"
        )
        reorient_dir = Path(
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_preprocessed"
        )
        bet_dir = Path(
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed"
        )
    else:
        torchio_dir = Path(
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_unscaled_preprocessed"
        )
        reorient_dir = Path(
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_unscaled_MNI_preprocessed"
        )
        bet_dir = Path(
            "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_unscaled_MNI_HD-BET_preprocessed"
        )
    dataset_names = [
        "Dataset4999_IXIAllModalities",
        "Dataset5000_BraTS-GLI_2023",
        "Dataset5001_BraTS-SSA_2023",
        "Dataset5002_BraTS-MEN_2023",
        "Dataset5003_BraTS-MET_2023",
        "Dataset5004_BraTS-MET-NYU_2023",
        "Dataset5005_BraTS-PED_2023",
        "Dataset5006_BraTS-MET-UCSF_2023",
        "Dataset5007_UCSF-BMSR",
        "Dataset5010_ATLASR2",
        "Dataset5012_ShiftsBest",
        "Dataset5013_ShiftsLjubljana",
        "Dataset5038_BrainTumour",
        "Dataset5041_BRATS",
        "Dataset5042_BRATS2016",
        "Dataset5043_BrainDevelopment",
        "Dataset5044_EPISURG",
        "Dataset5046_FeTA",
        "Dataset5066_WMH",
        "Dataset5083_IXIT1",
        "Dataset5084_IXIT2",
        "Dataset5085_IXIPD",
        "Dataset5090_ISLES2022",
        "Dataset5095_MSSEG",
        "Dataset5096_MSSEG2",
        "Dataset5111_UCSF-ALPTDG-time1",
        "Dataset5112_UCSF-ALPTDG-time2",
        "Dataset5113_StanfordMETShare",
        "Dataset5114_UCSF-ALPTDG",
        "Dataset6000_PPMI-T1-3T-PreProc",
        "Dataset6001_ADNI-group-T1-3T-PreProc",
        "Dataset6002_OASIS3",
        "Dataset6003_AIBL",
        "Dataset7000_openneuro-ds004791",
        "Dataset7001_openneuro-ds004848",
    ]

    for ds in dataset_names:
        # TorchIO
        ds_src_img_dir = base_dir / ds / image_dir
        ds_src_seg_dir = base_dir / ds / label_dir
        ds_tgt_img_dir = torchio_dir / ds / image_dir
        ds_tgt_seg_dir = torchio_dir / ds / label_dir
        if os.path.exists(ds_src_seg_dir) and len(os.listdir(ds_src_seg_dir)) > 0:
            seg_available = True
        else:
            seg_available = False

        # If image directory doesn't exist, skip
        if not os.path.exists(ds_src_img_dir):
            continue

        if not os.path.exists(ds_tgt_img_dir):
            os.makedirs(ds_tgt_img_dir)
        if not os.path.exists(ds_tgt_seg_dir) and seg_available:
            os.makedirs(ds_tgt_seg_dir)
        failed = _resample_pad_intensity_normalize(
            ds_src_img_dir,
            ds_src_seg_dir,
            ds_tgt_img_dir,
            ds_tgt_seg_dir,
            seg_available=seg_available,
            min_max_norm=args.min_max_norm,
        )
        all_failed += failed

        # fslreorient2std
        ds_src_img_dir = torchio_dir / ds / image_dir
        ds_src_seg_dir = torchio_dir / ds / label_dir
        ds_tgt_img_dir = reorient_dir / ds / image_dir
        ds_tgt_seg_dir = reorient_dir / ds / label_dir

        if not os.path.exists(ds_tgt_img_dir):
            os.makedirs(ds_tgt_img_dir)
        if not os.path.exists(ds_tgt_seg_dir) and seg_available:
            os.makedirs(ds_tgt_seg_dir)
        failed = _reorient(
            ds_src_img_dir,
            ds_tgt_img_dir,
            skip=(ds == "Dataset5012_ShiftsBest"),
            swapdim="x -y z"
            if ds == "Dataset5113_StanfordMETShare"
            else None,  # for some reason, this dataset needs to swap dimensions
        )
        all_failed += failed
        if seg_available:
            failed = _reorient(
                ds_src_seg_dir,
                ds_tgt_seg_dir,
                skip=(ds == "Dataset5012_ShiftsBest"),
                swapdim="x -y z"
                if ds == "Dataset5113_StanfordMETShare"
                else None,  # for some reason, this dataset needs to swap dimensions
            )
            all_failed += failed

        # HD-BET
        ds_src_img_dir = reorient_dir / ds / image_dir
        ds_src_seg_dir = reorient_dir / ds / label_dir
        ds_tgt_img_dir = bet_dir / ds / image_dir
        ds_tgt_seg_dir = bet_dir / ds / label_dir

        if not os.path.exists(ds_tgt_img_dir):
            os.makedirs(ds_tgt_img_dir)
        if not os.path.exists(ds_tgt_seg_dir) and seg_available:
            os.makedirs(ds_tgt_seg_dir)
        failed = _hdbet(
            ds_src_img_dir,
            ds_tgt_img_dir,
            need_skullstrip=ds in need_skullstrip,
        )
        all_failed += failed

        # Since segmentations don't need to be extracted, just copy it over
        if seg_available:
            failed = _hdbet(
                ds_src_seg_dir,
                ds_tgt_seg_dir,
                need_skullstrip=False,
            )
            all_failed += failed

    if len(all_failed) == 0:
        print("All successful!")
    else:
        print("Failed on:\n")
        pprint(all_failed)


if __name__ == "__main__":
    main()
