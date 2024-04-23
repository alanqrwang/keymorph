import os
import shutil


# /midtier/sablab/scratch/alw4013/data/openneuro/ds004791/sub-0011/anat/sub-0011_T1w.nii.gz
def convert_to_flat_structure(input_dir, output_dir):
    for sub in os.listdir(input_dir):
        sub_dir = os.path.join(input_dir, sub)
        if os.path.isdir(sub_dir):
            anat_dir = os.path.join(sub_dir, "anat")
            for file in os.listdir(anat_dir):
                if file.endswith(".nii.gz") or file.endswith(".nii"):
                    src_path = os.path.join(anat_dir, file)
                    dst_path = os.path.join(output_dir, file)
                    print(f"Copying {src_path} to {dst_path}")
                    shutil.copy2(src_path, dst_path)

    print("Conversion to flat structure complete.")


# Example usage:
input_dir = "/midtier/sablab/scratch/alw4013/data/openneuro/ds004791"
output_dir = "/midtier/sablab/scratch/alw4013/data/openneuro_imgs/ds004791"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
convert_to_flat_structure(input_dir, output_dir)
