import numpy as np
from base_unet_3d_new import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader
from PIL import Image

def visualize_model(voxel_grid, mode='vox'):

    if mode == 'vox':
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        print(voxel_grid.shape)
        ax.voxels(voxel_grid.reshape(64,64,64), edgecolor='k')
        plt.show()
    
    elif mode == 'img':
        cuts = [0,10,20,30,40,50,60,63]
        fig, axes = plt.subplots(2, 4)
        axes = axes.flatten()
        
        slices = []
        for cut in cuts:
            image = voxel_grid.transpose(0,2)[cut].cpu().numpy()
            image = np.reshape(image,(64,64))
            slices.append(image)

        for slice, ax in zip(slices,axes):
            ax.imshow(slice,cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print('Please enter a valid visualization mode (img OR vox)')
        exit(0)

def load_3d_model(filename, mode='train'):
    
    if mode == 'train':
        models = torch.from_numpy(np.load(filename))
        channeled_models = np.array([arr.reshape(1,64, 64, 64) for arr in models.numpy()])
        dataload = DataLoader(channeled_models,batch_size=1)
        return dataload
    
    elif mode == 'sample':
        models = torch.load(filename,map_location='cpu')
        models = models.float().clip(min=0, max=1)
        models = torch.round(models).type(torch.uint8)
        models = (models * 255).type(torch.uint8)
        return models
    
    else: 
        print('Please enter a valide loading mode (train OR sample)')
        exit(0)

if __name__ == '__main__':
    
    data = load_3d_model('/dcs/pg22/u2294454/fresh_diffusion_2/3d_model_diff/model_hybrid_bathtub.pt',mode='sample')
    visualize_model(data,mode='vox')