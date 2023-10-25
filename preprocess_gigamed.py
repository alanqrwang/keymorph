import os
import subprocess
from pathlib import Path


def preprocess_dir(
    src_img_dir, src_seg_dir, tgt_img_dir, tgt_seg_dir, skullstrip=False
):
    print(f"processing: {src_img_dir}, {src_seg_dir} -> {tgt_img_dir}, {tgt_seg_dir}")

    img_data_paths = [os.path.join(src_img_dir, i) for i in os.listdir(src_img_dir)]
    seg_data_paths = [os.path.join(src_seg_dir, i) for i in os.listdir(src_seg_dir)]

    # First reorient
    for img_path in img_data_paths:
        reorient_command = f"fslreorient2std {img_path}"
        subprocess.run(reorient_command)

    if skullstrip:
        # Run HD Bet brain extraction
        bet_command = f"hd-bet -i {src_dir} -o {tgt_dir}"
        subprocess.run(bet_command)
        print(f"finished processing {ds} with {len(loaded_subjects)} subjects")

    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        path_name = basename.split(".")[0]
        extension = ".".join(basename.split(".")[1:])
        seg_path = os.path.join(seg_data_folder, path_name[:-5] + "." + extension)
        subject_kwargs = {
            "img": tio.ScalarImage(img_path),
            "seg": tio.LabelMap(seg_path),
        }
        subject = tio.Subject(**subject_kwargs)
        loaded_subjects.append(subject)
    print(dataset_name, len(loaded_subjects))


def main():
    src_dir = Path(
        "/share/sablab/nfs04/users/rs2492/data/nnUNet_preprocessed_DATA/nnUNet_raw_data_base"
    )
    tgt_dir = Path(
        "/share/sablab/nfs04/users/rs2492/data/nnUNet_preprocessed_DATA/nnUNet_MNI_HD-BET"
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
        "Dataset5044_EPISURG",
        "Dataset5046_FeTA",
        "Dataset5066_WMH",
        "Dataset5085_IXIPD",
    ]
    need_skullstrip = [""]

    for ds in dataset_names:
        if ds in need_skullstrip:
            skullstrip = True
        else:
            skullstrip = False

        ds_src_img_dir = src_dir / ds / "imagesTr"
        ds_src_seg_dir = src_dir / ds / "labelsTr"
        ds_tgt_img_dir = tgt_dir / ds / "imagesTr"
        ds_tgt_seg_dir = tgt_dir / ds / "labelsTr"

        if not os.exists(ds_tgt_img_dir):
            os.makedirs(ds_tgt_img_dir)
        if not os.exists(ds_tgt_seg_dir):
            os.makedirs(ds_tgt_seg_dir)
        preprocess_dir(
            ds_src_img_dir,
            ds_src_seg_dir,
            ds_tgt_img_dir,
            ds_tgt_seg_dir,
            skullstrip=skullstrip,
        )


if __name__ == "__main__":
    main()
