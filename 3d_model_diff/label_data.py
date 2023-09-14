import torch
import numpy as np 
import os
from torch.utils.data import DataLoader


def gather_npy_filenames(directory_path): 
    file_extension = ".npy"
    npy_files = [file for file in os.listdir(directory_path) if file.endswith(file_extension)]
    return npy_files

def label_gathered_datasets(npy_files,directory_path):
    total_ds = []
    label_map = {
        'bathtub':0,
        'monitor':1,
        'bed':2,
        'chair':3,
        'toilet':4,
        'desk':5
    }
    for file in npy_files:
        print(file)
        path = directory_path+'/'+file
        data = np.load(path)
        print('loaded')
        label = file[6:-5]
        labeled = [([vox],label_map[label]) for vox in data]
        total_ds.append(labeled)

    total_ds = np.concatenate(total_ds)
    return np.array(total_ds)

def add_label(model,label):
    return (model,label)

def label_dataloader(dataset):
    data_list, label_list = zip(*dataset)

    data_tensor = torch.tensor(data_list)
    label_tensor = torch.tensor(label_list)

    tensor_dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)

    dataloader = DataLoader(tensor_dataset, batch_size=1, shuffle=True)
    return dataloader

if __name__ == '__main__':
    files = gather_npy_filenames('/dcs/pg22/u2294454/fresh_diffusion_2')
    print('gathered')
    print(label_gathered_datasets(files).shape)