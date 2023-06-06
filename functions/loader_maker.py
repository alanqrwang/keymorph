import os
import torch
import sklearn
import numpy as np
import torchio as tio
import torch.nn.functional as F
from torchio.transforms import Lambda, Compose, RandomBiasField, RandomNoise
from torch.utils.data import DataLoader

def create(directory,
           start_end,
           transform,
           modality):
    """
    Create dataloader
    
    Arguments
    ---------
    directory  : directory where MRI modilities and skull-stripping mask are located
    start_end  : interval of indeces that is use to create the loader
    transform  : TorchIO transformation
    modality   : string T1, T2 or PD

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

    seg_path = []
    for d in subjects:
        seg_path += [directory+'{}_seg/'.format(modality) + d[:-4] + '_seg' + d[-4:] + '.gz']

    # Split for train, val or test
    paths = paths[start:end]
    mask_path = mask_path[start:end]
    seg_path = seg_path[start:end]

    subjects = subjects[start:end]

    """Making loader"""
    loaded_subjects = []
    for i in range(len(paths)):
        _ls = tio.Subject(mri=tio.ScalarImage(paths[i]),
                          mask=tio.LabelMap(mask_path[i]),
                          seg = tio.LabelMap(seg_path[i]),
                          name=subjects[i])
        loaded_subjects.append(_ls)

    dataset = tio.SubjectsDataset(loaded_subjects,
                                  transform=transform)
    return dataset


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

    """Making loader"""
    loaded_subjects = []
    for i in range(len(paths)):
        _ls = tio.Subject(mri=tio.ScalarImage(paths[i]),
                          name=subjects[i])
        loaded_subjects.append(_ls)

    dataset = tio.SubjectsDataset(loaded_subjects,
                                  transform=transform)

    return dataset

class IXIDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, seed, modality, downsample_rate=1, 
                start_end=(0,427), transform='none'):
        super().__init__()
        self.directory = root_path
        self.seed = seed
        self.downsample_rate = downsample_rate
        assert transform in ['none', 'biasfield+noise']
        if transform is None or transform == 'none':
            self.transform = Lambda(lambda x: x.permute(0,1,3,2))
        elif transform == 'biasfield+noise':
            self.transform = tio.Compose(
                            [RandomBiasField(),
                            RandomNoise(),
                            Lambda(lambda x: x.permute(0,1,3,2))])
        self.dataset = create(self.directory,
                            start_end=start_end,
                            modality=modality,
                            transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fixed_sub = self.dataset[index]
        mri_fixed = fixed_sub['mri'][tio.DATA]
        mask_fixed = fixed_sub['mask'][tio.DATA].float()
        seg_fixed = fixed_sub['seg'][tio.DATA].float()
        mri_fixed = mask_fixed * mri_fixed

        mri_fixed = F.avg_pool3d(mri_fixed, self.downsample_rate, self.downsample_rate)
        seg_fixed = F.interpolate(seg_fixed[None], scale_factor=1/self.downsample_rate, mode='nearest')[0]
        seg_fixed = one_hot(seg_fixed)

        return mri_fixed, seg_fixed

class PairedIXIDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, seed, modalities, downsample_rate=1, 
                start_end=(0,427), mix_modalities=False, transform='none'):
        super().__init__()
        self.directory = root_path
        self.seed = seed
        self.downsample_rate = downsample_rate
        self.mix_modalities = mix_modalities
        assert transform in ['none', 'biasfield+noise']
        if transform is None or transform == 'none':
            self.transform = Lambda(lambda x: x.permute(0,1,3,2))
        elif transform == 'biasfield+noise':
            self.transform = tio.Compose(
                            [RandomBiasField(),
                            RandomNoise(),
                            Lambda(lambda x: x.permute(0,1,3,2))])

        self.datasets = [create(self.directory,
                            start_end=start_end,
                            modality=m,
                            transform=self.transform,
                            ) for m in modalities]
        if mix_modalities:
            combine_dataset = None
            for d in self.datasets:
                combine_dataset = d if combine_dataset is None else combine_dataset + d
            self.datasets = combine_dataset

    def __len__(self):
        if self.mix_modalities:
            return len(self.datasets)
        else:
            return len(self.datasets[0])

    def __getitem__(self, index):
        if self.mix_modalities:
            dataset = self.datasets
        else:
            mod_idx = np.random.randint(len(self.datasets))
            dataset = self.datasets[mod_idx]

        index2 = np.random.randint(0, self.__len__())

        fixed_sub = dataset[index]
        mri_fixed = fixed_sub['mri'][tio.DATA]
        mask_fixed = fixed_sub['mask'][tio.DATA].float()
        seg_fixed = fixed_sub['seg'][tio.DATA].float()
        mri_fixed = mask_fixed * mri_fixed

        moving_sub = dataset[index2]
        mri_moving = moving_sub['mri'][tio.DATA]
        mask_moving = moving_sub['mask'][tio.DATA].float()
        seg_moving = moving_sub['seg'][tio.DATA].float()
        mri_moving = mask_moving * mri_moving 

        
        mri_fixed = F.avg_pool3d(mri_fixed, self.downsample_rate, self.downsample_rate)
        mri_moving = F.avg_pool3d(mri_moving, self.downsample_rate, self.downsample_rate)
        seg_fixed = F.interpolate(seg_fixed[None], scale_factor=1/self.downsample_rate, mode='nearest')[0]
        seg_moving = F.interpolate(seg_moving[None], scale_factor=1/self.downsample_rate, mode='nearest')[0]

        seg_fixed = one_hot(seg_fixed)
        seg_moving = one_hot(seg_moving)
        return mri_fixed, mri_moving, seg_fixed, seg_moving

class PairedSubjectIXIDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, seed, modalities, downsample_rate=1, 
               start_end=(0,427), transform='none'):
        super().__init__()
        self.directory = root_path
        self.seed = seed
        self.downsample_rate = downsample_rate
        assert transform in ['none', 'biasfield+noise']
        if transform is None or transform == 'none':
            self.transform = Lambda(lambda x: x.permute(0,1,3,2))
        elif transform == 'biasfield+noise':
            self.transform = tio.Compose(
                            [RandomBiasField(),
                            RandomNoise(),
                            Lambda(lambda x: x.permute(0,1,3,2))])

        self.datasets = [create(self.directory,
                            start_end=start_end,
                            modality=m,
                            transform=self.transform,
                            ) for m in modalities]

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        mod_idxs = np.random.choice(len(self.datasets), size=2, replace=False)
        mod1 = self.datasets[mod_idxs[0]]
        mod2 = self.datasets[mod_idxs[1]]

        fixed_sub = mod1[index]
        mri_fixed = fixed_sub['mri'][tio.DATA]
        mask_fixed = fixed_sub['mask'][tio.DATA].float()
        mri_fixed = mask_fixed * mri_fixed

        moving_sub = mod2[index]
        mri_moving = moving_sub['mri'][tio.DATA]
        mask_moving = moving_sub['mask'][tio.DATA].float()
        mri_moving = mask_moving * mri_moving 

        mri_fixed = F.avg_pool3d(mri_fixed, self.downsample_rate, self.downsample_rate)
        mri_moving = F.avg_pool3d(mri_moving, self.downsample_rate, self.downsample_rate)

        return mri_fixed, mri_moving

def one_hot(asegs):
    subset_regs = [[0,0],   #Background
                  [13,52], #Pallidum   
                  [18,54], #Amygdala
                  [11,50], #Caudate
                  [3,42],  #Cerebral Cortex
                  [17,53], #Hippocampus
                  [10,49], #Thalamus
                  [12,51], #Putamen
                  [2,41],  #Cerebral WM
                  [8,47],  #Cerebellum Cortex
                  [4,43],  #Lateral Ventricle
                  [7,46],  #Cerebellum WM
                  [16,16]] #Brain-Stem

    _, dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(chs, dim1, dim2, dim3)

    for i,s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1]) 
        one_hot[i,:,:,:] = (combined_vol*1).float()

    mask = one_hot.sum(0).squeeze() 
    ones = torch.ones_like(mask)
    non_roi = ones-mask    
    one_hot[-1,:,:,:] = non_roi    

    assert one_hot.sum(0).sum() == dim1*dim2*dim3, 'One-hot encoding does not add up to 1'
    return one_hot

def get_loaders(root_path, 
                seed, 
                batch_size, 
                modalities, 
                downsample_rate, 
                num_val_subjects, 
                num_test_subjects, 
                mix_modalities=False, 
                transform='none'):
    modalities = [modalities] if not isinstance(modalities, (list, tuple)) else modalities

    ds_train = PairedIXIDataset(root_path, seed, modalities, downsample_rate, 
                                start_end=(0, 427), mix_modalities=mix_modalities, 
                                transform=transform)
    ds_val = PairedIXIDataset(root_path, seed, modalities, downsample_rate, 
                              start_end=(428, 428+num_val_subjects), mix_modalities=True)
    ds_test = {}
    for mod in modalities:
        ds_test.update({mod : 
                        IXIDataset(root_path, seed, mod, downsample_rate, 
                                   start_end=(428, 428+num_test_subjects))})

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = {}
    for mod, ds in ds_test.items():
        test_loader.update({mod : 
                        DataLoader(ds, batch_size=batch_size, shuffle=False)})
    return train_loader, val_loader, test_loader

def get_loader_same_sub(root_path, seed, batch_size, modalities, downsample_rate):
    modalities = [modalities] if not isinstance(modalities, (list, tuple)) else modalities
    ds_train = PairedSubjectIXIDataset(root_path, seed, modalities, downsample_rate, start_end=(0, 427))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    return train_loader