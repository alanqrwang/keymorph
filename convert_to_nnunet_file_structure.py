from glob2 import glob
from natsort import natsorted
import os

## Multi-modal Images and Segs
images1 = natsorted(glob("/midtier/sablab/scratch/alw4013/IXI/PD/*.nii.gz"))
# segs = natsorted(
#     glob(
#         "/share/sablab/nfs04/users/rs2492/data/ALL_DATA/UCSF-ALPTDG/data/*/*time1_seg.nii.gz"
#     )
# )

split_idx = 427
train_dir = (
    "/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base/Dataset5085_IXIPD/imagesTr"
)
test_dir = (
    "/midtier/sablab/scratch/alw4013/nnUNet_raw_data_base/Dataset5085_IXIPD/imagesTs"
)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

## Saving in nnUNet Style Change the Dataset Number and name
for i in range(split_idx):
    dst_file = (
        train_dir
        + "/"
        + "IXIPD"
        + "_"
        + "{:06}".format(i + 1)
        + "_000"
        + str(0)
        + ".nii.gz"
    )
    os.system("cp -avr " + images1[i] + " " + dst_file)

for i in range(split_idx, len(images1)):
    dst_file = (
        test_dir
        + "/"
        + "IXIPD"
        + "_"
        + "{:06}".format(i + 1)
        + "_000"
        + str(0)
        + ".nii.gz"
    )
    os.system("cp -avr " + images1[i] + " " + dst_file)
