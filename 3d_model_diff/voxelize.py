import open3d as o3d
import numpy as np
from base_unet_3d_new import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader
from data_utils import load_3d_model, visualize_model

'''

This class is not part of the training or sampling loop.

This is mainly due to the fact the the open3d package 
requires Python3.6 or less.

It was only used to voxelize the .off models.

The .npy models were made using this class. 

Runs on CPU-ONLY.

'''
def mesh_to_voxel(filename):
    mesh = o3d.io.read_triangle_mesh(filename)

    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
            center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                                voxel_size=0.02)
    grid_100x3 = np.zeros((64,64,64))
    voxels = np.array(voxel_grid.get_voxels())
    print(f'{filename} amount of voxels: {len(voxels)}')
    for voxel in voxels:
        grid_100x3[voxel.grid_index[0]][voxel.grid_index[1]][voxel.grid_index[2]] = 1
    return grid_100x3

def gather_data(folder_name):

    ''' Gather data from the 3d_dataset folder and convert to voxels.npy '''

    model_files = [f for f in listdir(folder_name) if 
                   isfile(join(folder_name, f))]
    print(model_files)
    dataset = []
    for file in model_files:
        voxels = mesh_to_voxel(folder_name+file)
        dataset.append(voxels)
    return dataset

if __name__ == '__main__':
    toilet_data = gather_data('/dcs/pg22/u2294454/fresh_diffusion_2/3d_dataset/monitor/train/')
    print(toilet_data)