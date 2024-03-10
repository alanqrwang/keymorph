import os
import dicom2nifti
from glob import glob

## Multi-modal Images and Segs
base_dir = "/midtier/sablab/scratch/data/ADNI-MRI-original-3D/ADNI/"
ds_name = "ADNI"
ds_num = "1007"

tot_subjects = 2578
split_idx = int(tot_subjects * 0.8)

train_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTr"
test_dir = f"/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset{ds_num}_{ds_name}/imagesTs"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

strings_to_ignore = [
    "cor_3d_frm",
    "ax",
    "cor",
    "cor_mpr",
    "smartbrain",
    "mip_images_sw_",
]
substrings_to_ignore = [
    "pasl",
    "asl",
    "dti",
    "pet",
    "swi",
    "fmri",
    "mra",
    "mrs",
    "dti",
    "dti_30dir",
    "dti_64",
    "calibration",
    "pasl",
    "localizer",
    "field_mapping",
    "scout",
    "cal_head",
    "cal_",
    "_cal",
    "cor_mpr",
    "mri_note",
    "perfusion_weighted",
    "cor_3d_frm",
    "blood_flow",
]
# First, convert all dicom series to nifti
# for sub_dir in os.listdir(base_dir):
#     sub_dir = os.path.join(base_dir, sub_dir)
#     if os.path.isdir(sub_dir):
#         # print(sub_dir)
#         for modality in os.listdir(sub_dir):
#             modality_dir = os.path.join(sub_dir, modality)
#             if os.path.isdir(modality_dir):
# for timepoint in os.listdir(modality_dir):
#     timepoint_dir = os.path.join(modality_dir, timepoint)
#     if os.path.isdir(timepoint_dir):
#         print(timepoint_dir)
#         for series in os.listdir(timepoint_dir):
#             series_dir = os.path.join(timepoint_dir, series)
#             if os.path.isdir(series_dir):
#                 print(series_dir)
#                 dicom2nifti.convert_directory(
#                     series_dir, series_dir
#                 )


# Collect all paths per subject
m = set()
filtered_m = set()
path_dict = {}
for subject in os.listdir(base_dir):
    sub_dir = os.path.join(base_dir, subject)
    if os.path.isdir(sub_dir):
        path_dict[subject] = []
        for modality in os.listdir(sub_dir):
            modality_dir = os.path.join(sub_dir, modality)
            if os.path.isdir(modality_dir):
                m.add(modality)
                if not any(
                    string in modality.lower() for string in substrings_to_ignore
                ) and not any(string in modality for string in strings_to_ignore):
                    filtered_m.add(modality)
                    for timepoint in os.listdir(modality_dir):
                        timepoint_dir = os.path.join(modality_dir, timepoint)
                        if os.path.isdir(timepoint_dir):
                            for series in os.listdir(timepoint_dir):
                                series_dir = os.path.join(timepoint_dir, series)
                                if os.path.isdir(series_dir):
                                    for nii in glob(
                                        os.path.join(series_dir, "*.nii.gz")
                                    ):
                                        path_dict[subject].append(nii)

print(m)
print(filtered_m)

from pprint import pprint

# Saving in nnUNet Style Change the Dataset Number and name
# I really have no idea how ADNI is organized. So I just separate by
# subject and modality. Sorry, future me.
for sub_id, (subject_name, paths) in enumerate(path_dict.items()):
    split_dir = train_dir if sub_id < split_idx else test_dir
    for mod_id, path in enumerate(paths):
        dst_file = os.path.join(
            split_dir, f"{ds_name}_{sub_id+1:06}_{mod_id:04}.nii.gz"
        )
        cmd = "cp -avr " + path + " " + str(dst_file)
        print("running:", cmd)
        os.system(cmd)
