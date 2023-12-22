from glob2 import glob
from natsort import natsorted
import os

## Multi-modal Images and Segs
base_dir = "/midtier/sablab/scratch/alw4013/data/OASIS3/npp-preprocessed/image/sub-OAS31172/ses-d1717/anat"
base_dir = "/midtier/sablab/scratch/alw4013/data/OASIS3/npp-preprocessed/image/"
ds_name = "OASIS3"
ds_num = "6002"

split_idx = 844
train_dir = f"/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get paths for each subject
path_dict = {}
for sub_dir in natsorted(os.listdir(base_dir)):
    subject_dir = os.path.join(base_dir, sub_dir)
    for ses_dir in natsorted(os.listdir(subject_dir)):
        ses_dir = os.path.join(subject_dir, ses_dir)
        paths = natsorted(
            glob(os.path.join(ses_dir, "anat", f"*talairach_norm_.nii.gz"))
        )
        path_dict[sub_dir] = paths

from pprint import pprint

pprint(len([k for k, v in path_dict.items() if len(v) > 1]))
# Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    if len(paths) > 1:  # Only use if subject has more than 1 timepoints
        split_dir = train_dir if sub_id < split_idx else test_dir
        for time_id, path in enumerate(paths):
            print(time_id, path)
            mod_id = 0  # Only 1 mod, T1
            dst_file = os.path.join(
                split_dir, f"{ds_name}-time{time_id+1}_{sub_id+1:06}_{mod_id:04}.nii.gz"
            )
            cmd = "cp -avr " + path + " " + str(dst_file)
            print("running:", cmd)
            os.system(cmd)
