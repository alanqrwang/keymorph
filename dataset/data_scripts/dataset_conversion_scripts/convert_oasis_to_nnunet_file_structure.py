from glob import glob
import os

## Multi-modal Images and Segs
# base_dir = "/midtier/sablab/scratch/alw4013/data/OASIS3/npp-preprocessed/image/"
# base_dir = "/midtier/sablab/scratch/alw4013/data/brain_nolesions/OASIS3/"
base_dir = "/midtier/sablab/scratch/data/OASIS3/"
ds_name = "OASIS3"
ds_num = "1006"

split_ratio = 0.8
tot_subjects = 2838
split_idx = int(tot_subjects * split_ratio)

train_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# /midtier/sablab/scratch/data/OASIS3/OAS30246_MR_d0746/anat3/sub-OAS30246_ses-d0746_run-02_T1w.json

# See https://www.oasis-brains.org/files/OASIS-3_Imaging_Data_Dictionary_v2.3.pdf
modality_map = {
    "T1w": 0,
    "T2w": 1,
    "acq-TSE_T2w": 2,
    "T2star": 3,
    "FLASH": 4,
    "FLAIR": 5,
    "acq-TOF_angio": 6,
    "angio": 6,
}

# Get paths for each subject
path_dict = {}
for sub_dir in sorted(os.listdir(base_dir)):
    subject_dir = os.path.join(base_dir, sub_dir)
    path_dict[sub_dir] = []
    for img_type_dir in sorted(os.listdir(subject_dir)):
        if "anat" in img_type_dir:
            paths = sorted(glob(os.path.join(subject_dir, img_type_dir, f"*.nii.gz")))
            path_dict[sub_dir] += paths

from pprint import pprint


def extract_target_substring(s):
    # Split the string by underscore
    parts = s.split("_")

    # Remove the file extension to handle cases where the target part is at the end
    parts[-1] = parts[-1].split(".")[0]

    # Check if the last part is a "run-XX" pattern, if so, use the second last part
    if parts[-2].startswith("run-"):
        target_part = parts[-1]
    elif parts[-2].startswith("acq") and "hippocampus" not in parts[-2]:
        target_part = "_".join([parts[-2], parts[-1]])
    else:
        target_part = parts[-1]

    return target_part


pprint(path_dict)
# pprint(len([k for k, v in path_dict.items() if len(v) > 1]))
# Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    split_dir = train_dir if sub_id < split_idx else test_dir
    for time_id, path in enumerate(paths):
        mod_name = extract_target_substring(os.path.basename(path))

        mod_id = modality_map[mod_name]
        dst_file = os.path.join(
            split_dir, f"{ds_name}-time{time_id+1}_{sub_id+1:06}_{mod_id:04}.nii.gz"
        )
        cmd = "cp -avr " + path + " " + str(dst_file)
        print("running:", cmd)
        os.system(cmd)
