from glob2 import glob
from natsort import natsorted
import os

## Multi-modal Images and Segs
base_dir = "/midtier/sablab/scratch/alw4013/ADNI_group_T1_3T_PreProc/npp-preprocessed/"
images1 = natsorted(glob(os.path.join(base_dir, "*mni_norm.nii.gz")))
ds_name = "ADNI-group-T1-3T-PreProc"
ds_num = "6001"

split_idx = 540
train_dir = f"/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get all unique subjects
unique_subjects = set()
for path in images1:
    subject_name = os.path.basename(path).split("-")[0]
    unique_subjects.add(subject_name)

# Get paths for each subject
path_dict = {k: [] for k in unique_subjects}
for subject_name in unique_subjects:
    paths = natsorted(glob(os.path.join(base_dir, f"{subject_name}-*mni_norm.nii.gz")))
    path_dict[subject_name] = paths

print(len(path_dict))
## Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    if len(paths) > 1:  # Only use if subject has more than 1 timepoints
        split_dir = train_dir if sub_id < split_idx else test_dir
        for time_id, path in enumerate(paths):
            mod_id = 0  # Only 1 mod, T1
            dst_file = os.path.join(
                split_dir, f"{ds_name}-time{time_id+1}_{sub_id+1:06}_{mod_id:04}.nii.gz"
            )
            cmd = "cp -avr " + path + " " + str(dst_file)
            print("running:", cmd)
            os.system(cmd)
