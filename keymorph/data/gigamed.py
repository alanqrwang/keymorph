import numpy as np
import os
import glob
from torch.utils.data import Dataset
import torchio as tio
 
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile

class GigaMedDataset(Dataset):
    def __init__(self, dataset_name: str, transform=None):
        data_folder_name = 'nnUNetPlans_3d_fullres'
        MFM_preprocessed_folder = '/midtier/sablab/scratch/data/'
 
        dataset_data_folder = join(MFM_preprocessed_folder, dataset_name, data_folder_name)
 
        case_identifiers = [join(dataset_data_folder,i) for i in os.listdir(dataset_data_folder) if i.endswith("npz")]
        self.dataset = []
        for data_file_path in case_identifiers:
            self.dataset.append(data_file_path)
            # self.dataset[c]['data_file'] = data_file_path
            # self.dataset[c]['properties_file'] = data_file_path.replace(".npz", ".pkl")
        
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        npz_data = np.load(self.dataset[index])
        img = npz_data['data']
        seg = npz_data['seg'].astype(int)+1
        subject_kwargs = {
            'img' : tio.ScalarImage(tensor=img),
            'seg' : tio.LabelMap(tensor=seg),
        }
        subject = tio.Subject(**subject_kwargs)
        if self.transform:
            subject = self.transform(subject)
        return subject
