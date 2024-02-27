import os
import subprocess
from pathlib import Path
from pprint import pprint


def shell_command(command):
    print("RUNNING", command)
    subprocess.run(command, shell=True)


def _synthseg(src_img_dir, tgt_dir):
    """Segment with SynthSeg"""
    failed = []

    img_data_paths = [
        os.path.join(src_img_dir, f) for f in os.listdir(src_img_dir) if "mask" not in f
    ]
    out_data_paths = []
    indices_to_process = []
    for i, img_path in enumerate(img_data_paths):
        basename = os.path.basename(img_path)
        out_path = os.path.join(tgt_dir, basename)
        if not os.path.exists(
            out_path
        ):  # Only perform synthseg if the segmentation file doesn't exist
            indices_to_process.append(i)
        out_data_paths.append(out_path)

    img_data_paths = [img_data_paths[i] for i in indices_to_process]
    out_data_paths = [out_data_paths[i] for i in indices_to_process]

    print(len(img_data_paths))
    print(len(out_data_paths))

    in_path_txt = os.path.join(src_img_dir, "in_paths.txt")
    out_path_txt = os.path.join(src_img_dir, "out_paths.txt")
    with open(in_path_txt, "w") as fp:
        for item in img_data_paths:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open(out_path_txt, "w") as fp:
        for item in out_data_paths:
            # write each item on a new line
            fp.write("%s\n" % item)

    synthseg_cmd = f"python /home/alw4013/SynthSeg/scripts/commands/SynthSeg_predict.py --i {in_path_txt} --o {out_path_txt}"
    try:
        shell_command(synthseg_cmd)
    except Exception as e:
        print("Error synthseging:", src_img_dir)
        print(e)
        failed.append(src_img_dir)

    # Clean up after ourselves
    os.remove(in_path_txt)
    os.remove(out_path_txt)
    return failed


def main():
    all_failed = []
    # root_dir = Path(
    #     "/midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed"
    # )
    # root_dir = Path("/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base")
    # dataset_names = [
    # "Dataset4999_IXIAllModalities",
    # "Dataset5000_BraTS-GLI_2023",
    # "Dataset5001_BraTS-SSA_2023",
    # "Dataset5002_BraTS-MEN_2023",
    # "Dataset5003_BraTS-MET_2023",
    # "Dataset5004_BraTS-MET-NYU_2023",
    # "Dataset5005_BraTS-PED_2023",
    # "Dataset5006_BraTS-MET-UCSF_2023",
    # "Dataset5007_UCSF-BMSR",
    # "Dataset5010_ATLASR2",
    # "Dataset5012_ShiftsBest",
    # "Dataset5013_ShiftsLjubljana",
    # "Dataset5038_BrainTumour",
    # "Dataset5041_BRATS",
    # "Dataset5042_BRATS2016",
    # "Dataset5043_BrainDevelopment",
    # "Dataset5044_EPISURG",
    # "Dataset5046_FeTA",
    # "Dataset5066_WMH",
    # "Dataset5083_IXIT1",
    # "Dataset5084_IXIT2",
    # "Dataset5085_IXIPD",
    # "Dataset5090_ISLES2022",
    # "Dataset5095_MSSEG",
    # "Dataset5096_MSSEG2",
    # "Dataset5111_UCSF-ALPTDG-time1",
    # "Dataset5112_UCSF-ALPTDG-time2",
    # "Dataset5113_StanfordMETShare",
    # "Dataset5114_UCSF-ALPTDG",
    # "Dataset6000_PPMI-T1-3T-PreProc",
    # "Dataset6001_ADNI-group-T1-3T-PreProc",
    # "Dataset6002_OASIS3",
    # "Dataset6003_AIBL",
    # "Dataset7000_openneuro-ds004791",
    # "Dataset7001_openneuro-ds004848",
    # ]

    root_dir = Path(
        "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/"
    )
    dataset_names = [
        # "Dataset4999_IXIAllModalities",
        "Dataset1000_PPMI",
        "Dataset1001_PACS2019",
        "Dataset1002_AIBL",
        "Dataset1004_OASIS2",
        "Dataset1005_OASIS1",
        "Dataset1006_OASIS3",
        "Dataset1007_ADNI",
    ]

    for ds in dataset_names:
        # SynthSeg on train if it exists
        ds_src_img_dir = root_dir / ds / "imagesTr"
        if os.path.exists(ds_src_img_dir) and len(os.listdir(ds_src_img_dir)) > 0:
            ds_tgt_dir = root_dir / ds / "synthSeglabelsTr"

            if not os.path.exists(ds_tgt_dir):
                os.makedirs(ds_tgt_dir)
            failed = _synthseg(
                ds_src_img_dir,
                ds_tgt_dir,
            )
            all_failed += failed

        # SynthSeg on test if it exists
        ds_src_img_dir = root_dir / ds / "imagesTs"
        if os.path.exists(ds_src_img_dir) and len(os.listdir(ds_src_img_dir)) > 0:
            ds_tgt_dir = root_dir / ds / "synthSeglabelsTs"

            if not os.path.exists(ds_tgt_dir):
                os.makedirs(ds_tgt_dir)
            failed = _synthseg(
                ds_src_img_dir,
                ds_tgt_dir,
            )
            all_failed += failed

    if len(all_failed) == 0:
        print("All successful!")
    else:
        print("Failed on:\n")
        pprint(all_failed)


if __name__ == "__main__":
    main()
