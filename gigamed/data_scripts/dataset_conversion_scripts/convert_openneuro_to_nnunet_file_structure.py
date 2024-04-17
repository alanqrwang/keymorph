from glob2 import glob
from natsort import natsorted
import os

## Note: All images from openneuro are going to be test images
base_dir = "/midtier/sablab/scratch/alw4013/data/openneuro/"
openneuro_ds = "ds004848"

ds_num = "7001"
ds_name = f"openneuro-{openneuro_ds}"

train_dir = f"/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get paths for each subject
path_dict = {}
openneuro_dir = os.path.join(base_dir, openneuro_ds)
for subject_name in os.listdir(openneuro_dir):
    subject_dir = os.path.join(openneuro_dir, subject_name)
    if os.path.isdir(subject_dir):
        path_dict[subject_name] = []
        anat_dir = os.path.join(openneuro_dir, subject_dir, "anat")
        for mod_file in os.listdir(anat_dir):
            if mod_file.endswith("nii.gz"):
                path_dict[subject_name].append(os.path.join(anat_dir, mod_file))


print(path_dict)
## Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    # split_dir = train_dir if sub_id < split_idx else test_dir
    split_dir = test_dir
    for time_id, path in enumerate(paths):
        mod_id = 0  # Only 1 mod, T1
        dst_file = os.path.join(
            split_dir, f"{ds_name}_{sub_id+1:06}_{mod_id:04}.nii.gz"
        )
        cmd = "cp -avr " + path + " " + str(dst_file)
        print("running:", cmd)
        os.system(cmd)
