# import open3d as o3d
import numpy as np
from image_unet import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def image_processing(folder_name):
    
    transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(folder_name, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader


def visualize_image(voxel_grid):
    fig = plt.figure(figsize = (10,10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(voxel_grid.numpy(), edgecolor='k')

    # ax = fig.add_subplot(projection='3d')
    # ax.voxels(voxel_grid)

    pxl_plot = plt.imshow(voxel_grid[25])
    plt.savefig('test.png')

def load_all_images(folder_name):
    model_files = [f for f in listdir(folder_name) if 
                   isfile(join(folder_name, f))]
    dataset = []
    ctr = 0
    for file in model_files:
        print(ctr)
        ctr += 1
        
        image = image_processing(folder_name+file)
        # exit(0)
        dataset.append(image)
    print(len(dataset))
    return dataset
    
def read_images(tensor_images, mode):
#     channeled_models = np.array([arr.reshape(1, 64, 64) for arr in models]
# )
    # if mode == 'train':
    dataload = DataLoader(tensor_images,batch_size=16)
    return dataload
    # else: # load generated
    #     return models

def save_image(vox_tensor):
    vox_tensor = torch.round(vox_tensor)

if __name__ == '__main__':
    # visualize_image(x)
    # tensor_imgs = load_all_images('/dcs/pg22/u2294454/fresh_diffusion/image_dataset/Desert/')
    dataload = image_processing('/dcs/pg22/u2294454/fresh_diffusion/image_dataset')
    print(len(dataload))
    # print(tensor_imgs[0].shape)