import numpy as np
from unet_modules import *
import matplotlib.pyplot as plt
from data_load import load_3d_model
import argparse
import os

def show_model():

    parser = argparse.ArgumentParser()


    parser.add_argument("--filename",
                        default='TOILET.pt',
                        help="sampled data in .pt - input filename please",
                        type=str)

    parser.add_argument("--mode",
                        default='img',
                        help="choose between 'vox' or 'image' type ",
                        type=str)
    
    args = parser.parse_args()
    
    filename = args.filename
    mode = args.mode
    
    
    path = os.getcwd() + '/' + filename

    data = load_3d_model(path,mode='sample')
    visualize_model(data,mode=mode)

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

if __name__ == '__main__':
    show_model()