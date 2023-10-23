import os
import torchio as tio


def read_subjects_from_disk(directory: str, train: bool, dataset_name: str):
    if train:
        split_folder_img, split_folder_seg = "imagesTr", "labelsTr"
    else:
        split_folder_img, split_folder_seg = "imagesTs", "labelsTs"
    img_data_folder = os.path.join(directory, dataset_name, split_folder_img)
    seg_data_folder = os.path.join(directory, dataset_name, split_folder_seg)

    img_data_paths = sorted(
        [os.path.join(img_data_folder, name) for name in os.listdir(img_data_folder)]
    )

    # First, run through and get all unique modalities
    all_modalities = set()
    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        case_id, modality_id = (
            basename.split(".")[0][:-5],
            basename.split(".")[0][-5:],
        )
        all_modalities.add(modality_id)

    # Now, load all subjects, separated by modality
    subject_dict = {mod: [] for mod in all_modalities}
    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        case_id, modality_id = (
            basename.split(".")[0][:-5],
            basename.split(".")[0][-5:],
        )
        extension = ".".join(basename.split(".")[1:])
        seg_path = os.path.join(seg_data_folder, case_id + "." + extension)
        subject_kwargs = {
            "img": tio.ScalarImage(img_path),
            "seg": tio.LabelMap(seg_path),
        }
        subject = tio.Subject(**subject_kwargs)
        subject_dict[modality_id].append(subject)

    return subject_dict
