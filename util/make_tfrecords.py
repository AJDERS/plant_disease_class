import os 
import numpy as np
from tqdm import tqdm
import csv

def make_records(storage_dir, type_):
    try:
        os.mkdir(os.path.join(storage_dir, type_))
    except:
        return False
    label_dict = {}
    i = 0
    for mode in ['train', 'valid', 'eval']:
        try:
            os.path.join(storage_dir, type_, mode)
        except:
            return False
        data_lst = []
        label_lst = []
        num_classes = 0
        subfolder = os.path.join(storage_dir,f'{mode}')
        subfolder_files = os.listdir(subfolder)
        for f in tqdm(subfolder_files):
            if type_ in f:
                num_classes += 1
                data = np.load(os.path.join(subfolder, f))
                data = data[:,:,:,:100]
                label = f.split('___')[-1].split('.')[0]
                if not label in label_dict.keys():
                    label_dict[label] = i
                    i += 1
                label_data = np.full((data.shape[-1]), label_dict[label])
                data_lst.append(data)
                label_lst.append(label_data)                
        np.save(
            f'{os.path.join(storage_dir, type_, mode)}_data.npy',
            np.transpose(np.concatenate(data_lst, axis=-1), (3,0,1,2))
        )
        np.save(
            f'{os.path.join(storage_dir, type_, mode)}_label.npy',
            np.concatenate(label_lst, axis=-1)
        )
    path = 'label_dict.csv'
    with open(f'{os.path.join(storage_dir, type_, path)}', 'w') as f:
        for key in label_dict.keys():
            f.write(f"{key},{label_dict[key]}\n")
    return True
        