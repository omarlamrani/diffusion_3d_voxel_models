import numpy as np
from unet_modules import *
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader
from PIL import Image


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
    