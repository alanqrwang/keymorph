# Use for any directory where files are organized in a flat structure
from glob import glob
import os

## Multi-modal Images and Segs
base_dir = "/midtier/sablab/scratch/alw4013/data/brain_nolesions/AIBL/"
images1 = sorted(glob(os.path.join(base_dir, "*.nii.gz")))
ds_name = "AIBL"
ds_num = "1002"

split_ratio = 0.8
split_idx = int(len(images1) * split_ratio)
train_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Assume only one modality
mod_id = 0

## Saving in nnUNet Style Change the Dataset Number and name
for sub_id, path in enumerate(images1):
    split_dir = train_dir if sub_id < split_idx else test_dir
    dst_file = os.path.join(split_dir, f"{ds_name}_{sub_id+1:06}_{mod_id:04}.nii.gz")
    cmd = "cp -avr " + path + " " + str(dst_file)
    print("running:", cmd)
    os.system(cmd)
