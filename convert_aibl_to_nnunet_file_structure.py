from glob2 import glob
from natsort import natsorted
import os
import pandas as pd

## Multi-modal Images and Segs
base_dir = "/midtier/sablab/scratch/alw4013/data/AIBL/npp-preprocessed"
csv_path = (
    "/midtier/sablab/scratch/alw4013/data/AIBL/AIBL-MPRAGE_12_06_2023-MCI-LONG.csv"
)
csv = pd.read_csv(csv_path)
images1 = natsorted(glob(os.path.join(base_dir, "*mni_norm.nii.gz")))
ds_name = "AIBL"
ds_num = "6003"

split_idx = 0
train_dir = f"/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    print("creating dir")
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    print("creating dir 2")
    os.makedirs(test_dir)

# Get paths for each subject
path_dict = {}
for index, row in csv.iterrows():
    path = os.path.join(base_dir, f"{row['Image Data ID']}_mni_norm.nii.gz")
    if row["Subject"] in path_dict:
        path_dict[row["Subject"]].append(path)
    else:
        path_dict[row["Subject"]] = [path]

from pprint import pprint

# pprint(path_dict)
# Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    if len(paths) > 1:  # Only use if subject has more than 1 timepoints
        split_dir = train_dir if sub_id < split_idx else test_dir
        for time_id, path in enumerate(paths):
            mod_id = 0  # Only 1 mod, T1
            dst_file = os.path.join(
                split_dir, f"{ds_name}-time{time_id+1}_{sub_id+1:06}_{mod_id:04}.nii.gz"
            )
            cmd = "cp -avr " + path + " " + str(dst_file)
            os.system(cmd)
