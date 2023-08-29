import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from voxelization import *
from base_unet_3d import *
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
# from scipy.misc import imsave
# from hybrid_loss import *


class VarianceScheduler:
    def __init__(self, timesteps,device='cuda', start=0.0001, end=0.02, type="linear"):
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.device = device

        if type == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps,self.start,self.end).to(self.device)
        else:
            self.betas = self.cosine_scheduler(self.timesteps,self.start,self.end).to(self.device)

        self.alphas = (1 - self.betas).to(self.device)
        self.cumprod_alphas = torch.cumprod(self.alphas, dim=0).to(self.device) # change dim 0?
        # self.alphas_cumprod_prev = np.append(1.0, self.cumprod_alphas[:-1])
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.cumprod_alphas[:-1]]).to(self.device)
        # self.alphas_cumprod_next = np.append(self.cumprod_alphas[1:], 0.0)
        self.alphas_cumprod_next = torch.cat([self.cumprod_alphas[1:],torch.tensor([0.0]).to(self.device)]).to(self.device)
        

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.cumprod_alphas).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.cumprod_alphas).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.cumprod_alphas).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.cumprod_alphas).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.cumprod_alphas - 1).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0  - self.alphas_cumprod_prev) / (1.0 - self.cumprod_alphas)
        ).to(self.device)

        self.log_posterior_variance = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        ).to(self.device)

    def linear_beta_schedule(self,timesteps,start,end):
        return torch.linspace(start, end, timesteps).to(self.device)
    
    def cosine_scheduler(self,timesteps,start,end):
        betas = []
        for i in reversed(range(timesteps)):
            T = timesteps - 1
            beta = start + 0.5*(end - start) * (1 + np.cos((i/T) * np.pi))
            betas.append(beta)    
        return torch.Tensor(betas).to(self.device)

    def get_index_from_list(self,vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t).to(self.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def forward_diffusion_sample(self,x, t):
        
        epsilon = torch.randn_like(x,dtype=torch.float)
        sqrt_alphas_cumprod = torch.sqrt(self.cumprod_alphas).to(self.device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.cumprod_alphas).to(self.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(sqrt_alphas_cumprod, t, x.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        ).to(self.device)
        # avoid iterative shit
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * epsilon.to(self.device), epsilon.to(self.device)
    
    def timesteps_sampling(self,n):
        return torch.randint(low=1,high=self.timesteps,size=(n,))

#     def sample(self,model,n):
#         model.eval()
#         with torch.no_grad():
#             # generate random shit using normal dist
#             x = torch.randn((n,1,64,64,64)).to(self.device)
#             # goes from T to 0 sampling all transitions
#             for i in tqdm(reversed(range(1,self.timesteps)), position=0):
#                 t = (torch.ones(n)*i).long().to(self.device)
#                 print(i)
#                 pred_noise = model(x,t)
#                 beta = self.betas[t][:,None,None,None].to(self.device)
#                 alpha = self.alphas[t][:,None,None,None].to(self.device)
#                 cumprod_alpha = self.cumprod_alphas[t][:,None,None,None].to(self.device)
#                 if i>1:
#                     noise = torch.randn_like(x).to(self.device)
#                 else:
#                     noise = torch.zeros_like(x).to(self.device)
#                 x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - cumprod_alpha))) * pred_noise) + torch.sqrt(beta) * noise
#         model.train()
#         x = x.clip(min=0, max=1)
#         x = torch.round(x).type(torch.uint8)
#         x = (x * 255).type(torch.uint8)
        
#         return x.reshape(64,64,64).to(self.device)
    
#     def sample_with_var(self,model,n):
#         model.eval()
#         with torch.no_grad():
#             # generate random shit using normal dist
#             x = torch.randn((n,1,64,64,64)).to(self.device)
#             sample_t = None
#             # pred_x = None
#             for i in tqdm(reversed(range(1,self.timesteps)), position=0):
#                 t = (torch.ones(n)*i).long().to(self.device)
#                 print(i)
#                 pred_noise,_,pred_log_var,pred_pre_t = p_mean_variance(self,model,x,t)
#                 eps = torch.randn_like(x)
#                 nonzero_mask = (
#                     (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
#                 )
#                 sample_t = pred_noise + nonzero_mask * torch.exp(0.5*pred_log_var)*eps
#                 print(type(sample_t))
#                 print(sample_t.shape)
#                 x = pred_pre_t
#                 print(type(x))
#                 print(x.shape)

#                 # beta = self.betas[t][:,None,None,None].to(self.device)
#                 # alpha = self.alphas[t][:,None,None,None].to(self.device)
#                 # cumprod_alpha = self.cumprod_alphas[t][:,None,None,None].to(self.device)
#                 # if i>1:
#                 #     noise = torch.randn_like(x).to(self.device)
#                 # else:
#                 #     noise = torch.zeros_like(x).to(self.device)
#                 # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - cumprod_alpha))) * pred_noise) + torch.sqrt(beta) * noise
#         model.train()
#         x = x.clip(min=0, max=1)
#         x = torch.round(x).type(torch.uint8)
#         x = (x * 255).type(torch.uint8)

#         return x




# if __name__ == '__main__':
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
    