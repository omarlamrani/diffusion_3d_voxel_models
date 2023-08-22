import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from voxelization import *
from image_unet import *
from tqdm import tqdm

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
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.cumprod_alphas)
        ).to(self.device)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        # self.posterior_log_variance_clipped = np.log(
        #     np.append(self.posterior_variance[1], self.posterior_variance[1:])
        # )
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

    def sample(self,model,n):
        model.eval()
        with torch.no_grad():
            # generate random shit using normal dist
            x = torch.randn((n,3,64,64)).to(self.device)
            # goes from T to 0 sampling all transitions
            for i in tqdm(reversed(range(1,self.timesteps)), position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                print(i)
                pred_noise = model(x,t)
                beta = self.betas[t][:,None,None,None].to(self.device)
                alpha = self.alphas[t][:,None,None,None].to(self.device)
                cumprod_alpha = self.cumprod_alphas[t][:,None,None,None].to(self.device)
                if i>1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - cumprod_alpha))) * pred_noise) + torch.sqrt(beta) * noise
        model.train()
        x = torch.clip(x, min=0, max=1)
        # x = torch.round(x).type(torch.uint8)
        return x.to(self.device)


if __name__ == '__main__':
    
    device = 'cpu'
    model = UNet(device=device)
    var = VarianceScheduler(timesteps=10,device=device,type='cos')
    # x = var.sample(model,1).to('cuda')
    # torch.save(x,'gpu_sample.pt')
    print(var.betas)