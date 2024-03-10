from glob import glob
import os
import nibabel as nib

## Multi-modal Images and Segs
# base_dir = "/midtier/sablab/scratch/alw4013/data/OASIS3/npp-preprocessed/image/"
base_dir = "/midtier/sablab/scratch/data/OASIS1/"
ds_name = "OASIS1"
ds_num = "1005"

split_ratio = 0.8
tot_subjects = 436
split_idx = int(tot_subjects * split_ratio)

train_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get paths for each subject
path_dict = {}
for sub_dir in sorted(os.listdir(base_dir)):
    subject_dir = os.path.join(base_dir, sub_dir)
    paths = sorted(glob(os.path.join(subject_dir, "RAW", f"*.hdr")))
    path_dict[sub_dir] = paths

from pprint import pprint

pprint(path_dict)
# pprint(len([k for k, v in path_dict.items() if len(v) > 1]))
# Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    split_dir = train_dir if sub_id < split_idx else test_dir
    for time_id, path in enumerate(paths):
        print(time_id, path)
        mod_id = 0  # Only 1 mod, T1 MPRAGE
        dst_file = os.path.join(
            split_dir, f"{ds_name}-time{time_id+1}_{sub_id+1:06}_{mod_id:04}.nii.gz"
        )
        nifti_data = nib.load(path)
        print(f"saving {path} to {dst_file}")
        nib.save(nifti_data, dst_file)
