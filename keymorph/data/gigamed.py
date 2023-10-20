import os
import torchio as tio
 
def read_subjects_from_disk(directory: str, train: bool, dataset_name: str):

    if train:
        split_folder_img, split_folder_seg = 'imagesTr', 'labelsTr'
    else:
        split_folder_img, split_folder_seg = 'imagesTs', 'labelsTs'
    img_data_folder = os.path.join(directory, dataset_name, split_folder_img)
    seg_data_folder = os.path.join(directory, dataset_name, split_folder_seg)

    img_data_paths = [os.path.join(img_data_folder,i) for i in os.listdir(img_data_folder)]
    
    loaded_subjects = []
    for img_path in img_data_paths:
        basename = os.path.basename(img_path)
        path_name = basename.split('.')[0]
        extension = '.'.join(basename.split('.')[1:])
        seg_path = os.path.join(seg_data_folder, path_name[:-5]+'.'+extension)
        subject_kwargs = {
            'img' : tio.ScalarImage(img_path),
            'seg' : tio.LabelMap(seg_path),
        }
        subject = tio.Subject(**subject_kwargs)
        loaded_subjects.append(subject)
    print(dataset_name, len(loaded_subjects))
    
    return loaded_subjects