import open3d as o3d
import numpy as np
from base_unet_3d import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader
from data_utils import load_3d_model, visualize_model

def mesh_to_voxel(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    # maybe use to scale all models to same size
    # print(f'{filename}') #=> {np.max(mesh.get_max_bound() - mesh.get_min_bound())}')

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
    model_files = [f for f in listdir(folder_name) if 
                   isfile(join(folder_name, f))]
    dataset = []
    ctr = 0
    for file in model_files:
        print(str(file))
        # ctr += 1
        # print(str(file))
        voxels = mesh_to_voxel(folder_name+file)
        # exit(0)
        dataset.append(voxels)
    print(len(dataset))
    return dataset

if __name__ == '__main__':
    chair = load_3d_model('/dcs/pg22/u2294454/fresh_diffusion_2/3d_model_diff/model_test.pt',mode='test').reshape((64,64,64))
    visualize_model(chair)