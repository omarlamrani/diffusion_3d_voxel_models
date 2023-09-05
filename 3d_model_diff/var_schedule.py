import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from voxelization import *
from base_unet_3d_new import *
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
# from scipy.misc import imsave
# from hybrid_loss import *
import os


class VarianceScheduler:
    def __init__(self,timesteps,device='cuda', start=0.0001, end=0.02, type="linear"):
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.device = device

        if type == 'linear':
            self.betas = self.linear_schedule(self.timesteps,self.start,self.end).to(self.device)
        else:
            self.betas = self.cosine_schedule(self.timesteps,self.start,self.end).to(self.device)

        self.alphas = (1 - self.betas).to(self.device)
        self.cumprod_alphas = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.cumprod_alphas[:-1]]).to(self.device)
        self.alphas_cumprod_next = torch.cat([self.cumprod_alphas[1:],torch.tensor([0.0]).to(self.device)]).to(self.device)
        

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.cumprod_alphas).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.cumprod_alphas).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.cumprod_alphas).to(self.device)
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1.0 / self.cumprod_alphas).to(self.device)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1.0 / self.cumprod_alphas - 1).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0  - self.alphas_cumprod_prev) / (1.0 - self.cumprod_alphas)
        ).to(self.device)

        self.log_posterior_variance = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        ).to(self.device)

    def linear_schedule(self,timesteps,start,end):
        return torch.linspace(start, end, timesteps).to(self.device)
    
    def cosine_schedule(self,timesteps,start,end):
        betas = []
        for i in reversed(range(timesteps)):
            T = timesteps - 1
            beta = start + 0.5*(end - start) * (1 + np.cos((i/T) * np.pi))
            betas.append(beta)    
        return torch.Tensor(betas).to(self.device)

    def get_sampled_t_values(self,values, t, x_shape): #gud
        batch_size = t.shape[0]
        # print(batch_size)
        out = values.gather(-1, t).to(self.device)
        # return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)
        new_shape = [batch_size] + [1] * (len(x_shape) - 1)
        out = out.view(*new_shape)
        
        # Move the tensor to the specified device
        out = out.to(self.device)
        
        return out

    def fwd_diff_t(self,x, t):
        
        epsilon = torch.randn_like(x)
        # sqrt_alphas_cumprod = torch.sqrt(self.cumprod_alphas).to(self.device)
        # sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.cumprod_alphas).to(self.device)
        sqrt_alphas_cumprod_t = self.get_sampled_t_values(self.sqrt_alphas_cumprod, t, x.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_sampled_t_values(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        ).to(self.device)
        # avoid iterative shit
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * epsilon.to(self.device), epsilon.to(self.device)
    
    def sample_timesteps(self,n):
        return torch.randint(1,self.timesteps,size=(n,))


if __name__ == '__main__':
    var = VarianceScheduler(10,device='cpu',type='cosine')
    t = var.sample_timesteps(1)
    print(t)
    test = torch.randint(3, 5, (1,6,),dtype=float)
    x_t, noise = var.fwd_diff_t(test,t)
    print(x_t)
    print(noise)
    print('OK')

#     # x = torch.load('model_test.pt')
#     # # transform = T.ToPILImage()
#     # print(x[10])
#     # exit(0)
#     device = 'cpu'
#     path = '/dcs/pg22/u2294454/fresh_diffusion_2/3d_model_diff/epoch_models_cosine_var500_beds.pth'
#     model = UNet(c_out=2,device=device)
#     model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#     model.eval()
#     var = VarianceScheduler(timesteps=1000,device=device)
#     x = var.sample_with_var(model,1).to(device)
    
#     # im = Image.fromarray(x.cpu().numpy())
#     # im.save("your_file.jpeg")
#     print(x.shape)
#     torch.save(x,'model_test.pt')

#     ##############################################################################################

#     x = torch.load('model_test.pt')
#     # transform = T.ToPILImage()
#     print(x)
#     # convert the tensor to PIL image using above transform
#     # img = transform(x)
#     cuts = [0,10,20,30,40,50,60]

#     for ctr, cut in enumerate(cuts):
#         iamg = x.transpose(0,2)[ctr].cpu().numpy()
#         print(iamg)
#         # print(transform(x[ctr]))
#         print('#################################################')
#         image = Image.fromarray(iamg, 'L')  # 'L' mode is for grayscale images

# # Save the image
#         image.save('bin_img_'+str(ctr)+'.JPG')
#         # img.save("models_"+str(ctr)+".jpg")
#     # display the PIL image
    
#     print('ok')
    