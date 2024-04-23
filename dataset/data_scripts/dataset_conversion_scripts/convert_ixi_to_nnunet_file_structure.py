from glob2 import glob
from natsort import natsorted
import os
from pathlib import Path

## Multi-modal Images and Segs
t1_base_dir = Path("/midtier/sablab/scratch/alw4013/IXI/T1/")
t2_base_dir = Path("/midtier/sablab/scratch/alw4013/IXI/T2/")
pd_base_dir = Path("/midtier/sablab/scratch/alw4013/IXI/PD/")
images1 = natsorted(glob(str(t1_base_dir / "*.nii.gz")))

split_idx = 427
train_dir = Path(
    "/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base/Dataset4999_IXIAllModalities/imagesTr"
)
test_dir = Path(
    "/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base/Dataset4999_IXIAllModalities/imagesTs"
)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

unique_subjects = set()
for path in images1:
    subject_name = "-".join(os.path.basename(path).split("-")[:3])
    unique_subjects.add(subject_name)

path_dict = {k: [] for k in unique_subjects}
for subject_name in unique_subjects:
    t1_path = t1_base_dir / (subject_name + "-T1.nii.gz")
    t2_path = t2_base_dir / (subject_name + "-T2.nii.gz")
    pd_path = pd_base_dir / (subject_name + "-PD.nii.gz")
    if os.path.exists(t1_path):
        path_dict[subject_name].append(t1_path)
    if os.path.exists(t2_path):
        path_dict[subject_name].append(t2_path)
    if os.path.exists(pd_path):
        path_dict[subject_name].append(pd_path)

## Saving in nnUNet Style Change the Dataset Number and name
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    if len(paths) == 3:  # Only use if subject has all 3 modalities
        split_dir = train_dir if sub_id < split_idx else test_dir
        for path in paths:
            path = str(path)
            if "T1" in path:
                mod_id = 0
            elif "T2" in path:
                mod_id = 1
            elif "PD" in path:
                mod_id = 2
            else:
                raise ValueError

            dst_file = split_dir / f"IXIAllModalities_{sub_id+1:06}_{mod_id:04}.nii.gz"
            cmd = "cp -avr " + path + " " + str(dst_file)
            print("running:", cmd)
            os.system(cmd)
