import torchio as tio
from glob import glob
from pathlib import Path

def read_subjects_from_disk(directory):
    """
    Get list of TorchIO subjects from disk. Each subject has an image,
    and optionally a mask and segmentation. Each subject is restricted
    to a single modality.
    
    :param: directory directory where MRI modilities and skull-stripping mask are located
    start_end  : interval of indeces that is use to create the loader
    modality   : string that matches modality on disk 

    Return
    ------
        dataset : tio.SubjectsDataset
        loader  : torch.utils.data.DataLoader
    """

    glob_path = directory / 'patient{[0-9][0-9][0-9]}' / 'patient{[0-9][0-9][0-9]}_frame{[0-9][0-9]}.nii.gz'
    glob_path_gt = directory / 'patient{[0-9][0-9][0-9]}' / 'patient{[0-9][0-9][0-9]}_frame{[0-9][0-9]}_gt.nii.gz'

    paths = sorted(glob(glob_path))
    paths_gt = sorted(glob(glob_path_gt))

    assert len(paths) == len(paths_gt), 'Number of images and ground truth masks do not match'

    # Images
    loaded_subjects = []
    for path, path_gt in zip(paths, paths_gt):
        # For each subject's image, try to get corresponding mask and segmentation
        subject_kwargs = {
            'img': tio.ScalarImage(path),
            'seg': tio.ScalarImage(path_gt),
        }
        _sub = tio.Subject(**subject_kwargs)
        loaded_subjects.append(_sub)

    return loaded_subjects