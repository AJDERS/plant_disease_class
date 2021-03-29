import os 
import numpy as np
from tqdm import tqdm
from PIL import Image
from random import sample

def main():
    os.makedirs('storage/train', exist_ok=True)
    os.makedirs('storage/valid', exist_ok=True)
    os.makedirs('storage/eval', exist_ok=True)
    for mode in ['train','valid','eval']:
        print(f'Processing "{mode}" data.')
        if mode != 'train':
            subfolders = os.listdir(f'data/valid')
        else:
            subfolders = os.listdir(f'data/{mode}')
        subfolder_dict = {}
        for subfolder in tqdm(subfolders):
            if '.' in subfolder:
                continue
            subfolder_dict[subfolder] = {}
            if mode != 'train':
                images = os.listdir(f'data/valid/{subfolder}')
            else:
                images = os.listdir(f'data/{mode}/{subfolder}')
            tmp_lst = []
            if mode != 'train':
                subfolder_dict['valid'] = images[int(len(images)/2):]
                subfolder_dict['evalu'] = images[:int(len(images)/2)]
            for image in images:
                if mode != 'train':
                    im = Image.open(f'data/valid/{subfolder}/{image}')
                else:
                    im = Image.open(f'data/{mode}/{subfolder}/{image}')
                data = np.asarray(im)
                if mode == 'valid':
                    if image in subfolder_dict['valid']:
                        tmp_lst.append(data)
                elif mode == 'eval':
                    if image in subfolder_dict['evalu']:
                        tmp_lst.append(data)
                else:
                    tmp_lst.append(data)
            all_ims = np.stack(tmp_lst, axis=-1)
            np.save(f'storage/{mode}/{subfolder}.npy', all_ims)



if __name__=='__main__':
    main()
