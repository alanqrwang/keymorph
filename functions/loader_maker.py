import os
import torch
import sklearn
import numpy as np
import torchio as tio


def create(directory,
           start_end,
           transform,
           modality,
           batch_size,
           shuffle,
           drop_last,
           num_workers,
           seed=23):
    """
    Create dataloader
    
    Arguments
    ---------
    directory  : directory where MRI modilities and skull-stripping mask are located
    transform  : TorchIO transformation
    start_end  : interval of indeces that is use to create the loader
    batch_size : batch size 
    modality   : string T1, T2 or PD
    shuffle    : shuffle while sampling subjects
    drop_last  : to drop the last incomplete batch
    seed       : seed numberused to shuffle subject in directory

    Return
    ------
        dataset : tio.SubjectsDataset
        loader  : torch.utils.data.DataLoader
    """

    modality = modality.upper()
    start, end = start_end[0], start_end[1]
    suffix = '_mask/'

    """Get PATHS"""
    paths = []
    subjects = []
    for d in np.sort(os.listdir(directory + '{}/'.format(modality))):
        if 'ipynb' in d:
            continue
        paths += [directory + '{}/'.format(modality) + d]
        subjects += [d]

    mask_path = []
    for d in subjects:
        mask_path += [directory + '{}'.format(modality) + suffix + d[:-4] + '_mask' + d[-4:] + '.gz']

    '''Shuffle Subjects'''
    indices = np.arange(len(paths))
    indices = sklearn.utils.shuffle(indices, random_state=seed)

    paths = (np.array(paths)[indices]).tolist()
    mask_path = (np.array(mask_path)[indices]).tolist()

    subjects = (np.array(subjects)[indices]).tolist()

    # Split for train, val or test
    paths = paths[start:end]
    mask_path = mask_path[start:end]

    subjects = subjects[start:end]

    """Making loader"""
    loaded_subjects = []
    for i in range(len(paths)):
        _ls = tio.Subject(mri=tio.ScalarImage(paths[i]),
                          mask=tio.LabelMap(mask_path[i]),
                          name=subjects[i])
        loaded_subjects.append(_ls)

    dataset = tio.SubjectsDataset(loaded_subjects,
                                  transform=transform)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         drop_last=drop_last,
                                         num_workers=num_workers)
    return dataset, loader


def create_simple(directory,
                  transform,
                  modality):
    """
    Create dataloader
    
    Arguments
    ---------
    directory  : directory where MRI modilities and skull-stripping mask are located
    transform  : TorchIO transformation
    modality   : string T1, T2 or PD

    Return
    ------
        dataset : tio.SubjectsDataset
        loader  : torch.utils.data.DataLoader
    """

    modality = modality.upper()

    """Get PATHS"""
    paths = []
    subjects = []
    for d in np.sort(os.listdir(directory + '{}/'.format(modality))):
        if 'ipynb' in d:
            continue
        paths += [directory + '{}/'.format(modality) + d]
        subjects += [d]

    indices = np.arange(len(paths))

    """Making loader"""
    loaded_subjects = []
    for i in range(len(paths)):
        _ls = tio.Subject(mri=tio.ScalarImage(paths[i]),
                          name=subjects[i])
        loaded_subjects.append(_ls)

    dataset = tio.SubjectsDataset(loaded_subjects,
                                  transform=transform)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False)
    return dataset, loader
