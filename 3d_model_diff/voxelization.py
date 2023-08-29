# import open3d as o3d
import numpy as np
from base_unet_3d import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader
# import open3d as o3d

# mesh_data = o3d.data.BunnyMesh()
# print(mesh_data)

#################################################################################

# def mesh_to_voxel(filename):
#     mesh = o3d.io.read_triangle_mesh(filename)
#     # maybe use to scale all models to same size
#     # print(f'{filename}') #=> {np.max(mesh.get_max_bound() - mesh.get_min_bound())}')

#     mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
#             center=mesh.get_center())
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
#                                                                 voxel_size=0.02)
#     grid_100x3 = np.zeros((64,64,64))
#     voxels = np.array(voxel_grid.get_voxels())
#     print(f'{filename} amount of voxels: {len(voxels)}')
#     for voxel in voxels:
#         grid_100x3[voxel.grid_index[0]][voxel.grid_index[1]][voxel.grid_index[2]] = 1
#     return grid_100x3

def visualize_model(voxel_grid):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    print(voxel_grid.shape)
    ax.voxels(voxel_grid, edgecolor='k')

    # ax = fig.add_subplot(projection='3d')
    # ax.voxels(voxel_grid)
    plt.show()
    # pxl_plot = plt.imshow(voxel_grid[25])
    # plt.savefig('test.png')

###########################################################################

# def gather_data(folder_name):
#     model_files = [f for f in listdir(folder_name) if 
#                    isfile(join(folder_name, f))]
#     dataset = []
#     ctr = 0
#     for file in model_files:
#         print(str(file))
#         # ctr += 1
#         # print(str(file))
#         voxels = mesh_to_voxel(folder_name+file)
#         # exit(0)
#         dataset.append(voxels)
#     print(len(dataset))
#     return dataset
    
def load_3d_model(filename, mode):
    models = torch.load(filename,map_location=torch.device('cpu'))
    models = models.clip(min=0, max=1)
    models = torch.round(models).type(torch.uint8)
    models = (models * 255).type(torch.uint8)
    channeled_models = np.array([arr.reshape(1,64, 64, 64) for arr in models.numpy()]
)
    if mode == 'train':
        dataload = DataLoader(channeled_models,batch_size=1)
        return dataload
    else: # load generated
        return models

def save_3d_model(vox_tensor):
    vox_tensor = torch.round(vox_tensor)

if __name__ == '__main__':
    # voxelize = mesh_to_voxel('sofa/train/sofa_0093.off')
    # visualize_model(voxelize)
    chair = load_3d_model('/dcs/pg22/u2294454/fresh_diffusion_2/3d_model_diff/model_test.pt',mode='test').reshape((64,64,64))
    visualize_model(chair)

    # np.save('/dcs/pg22/u2294454/fresh_diffusion_2/tst.npy',np.array([4]))
    print('START.')

    # bathtub = gather_data('/dcs/pg22/u2294454/fresh_diffusion_2/3d_dataset/bathtub/train/')
    # np.save('/dcs/pg22/u2294454/fresh_diffusion_2/voxel_bathtubs.npy',bathtub)
    # print ('BATHTUB DONE.')

    # bed = gather_data('/dcs/pg22/u2294454/fresh_diffusion_2/3d_dataset/bed/train/')
    # np.save('/dcs/pg22/u2294454/fresh_diffusion_2/voxel_beds.npy',bed)
    # print ('BED DONE.')

    # desk = gather_data('/dcs/pg22/u2294454/fresh_diffusion_2/3d_dataset/desk/train/')
    # np.save('/dcs/pg22/u2294454/fresh_diffusion_2/voxel_desks.npy',desk)
    # print ('DESK DONE.')

    # monitor = gather_data('/dcs/pg22/u2294454/fresh_diffusion_2/3d_dataset/monitor/train/')
    # np.save('/dcs/pg22/u2294454/fresh_diffusion_2/voxel_monitors.npy',monitor)
    # print ('MONITOR DONE.')

    # toilet = gather_data('/dcs/pg22/u2294454/fresh_diffusion_2/3d_dataset/toilet/train/')
    # np.save('/dcs/pg22/u2294454/fresh_diffusion_2/voxel_toilets.npy',toilet)
    # print ('TOILET DONE.')


    # test = np.save('voxel_desks.npy')
    # print(test.shape)