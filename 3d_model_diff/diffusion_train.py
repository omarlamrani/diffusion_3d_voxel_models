import sys
from base_unet_3d import UNet
from torch import optim
import torch.nn as nn
from var_schedule import *
from tqdm import tqdm
from voxelization import load_3d_model
# print(sys.version)

# def get_loss(model, x_0, t):
#     x_noisy, noise = forward_diffusion_sample(x_0, t)
#     noise_pred = model(x_noisy, t)
#     return F.l1_loss(noise, noise_pred)

def train_model():
    epochs = 502
    gpu = 'cuda'
    model = UNet(device=gpu).float().to(gpu)
    optimizer = optim.AdamW(model.parameters(),lr=3e-4)
    mse = nn.MSELoss() #Will need variational lower bound frop variance prediction
    var_schedule = VarianceScheduler(timesteps=1000,device=gpu)
    dataloader = load_3d_model('/dcs/pg22/u2294454/fresh_diffusion/voxel_beds.npy',mode='train')
    # prog = tqdm(dataloader)
    epoch_loss = 10
    for epoch in range(epochs):
        # progress = tqdm()
        print(f'==> EPOCH N*{epoch}')
        for i, vox_models in enumerate(dataloader):
            vox_models = vox_models.to(gpu)
            t = var_schedule.timesteps_sampling(vox_models.shape[0]).to(gpu)
            x_t, noise = var_schedule.forward_diffusion_sample(vox_models,t)
            # print(x_t.dtype)
            pred_noise = model(x_t.float(),t)
            # print(noise.dtype)
            # print(pred_noise.dtype)
            loss = mse(noise,pred_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # prog.set_postfix(MSE=loss.item())
            epoch_loss = loss
        print(epoch_loss)
        if epoch%250 == 0:
            torch.save(model.state_dict(),'epoch_'+str(epoch)+'_beds_gpu_'+str(epoch_loss)+'.pth')


if __name__ == '__main__':
    train_model()
