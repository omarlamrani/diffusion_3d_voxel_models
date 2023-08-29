# import sys
from base_unet_3d import UNet
# from torch import optim
# import torch.nn as nn
from var_schedule import *
from tqdm import tqdm
# from voxelization import load_3d_model
import torch
from hybrid_loss import *
# import numpy as np 

def sample(self,model,n):
        model.eval()
        with torch.no_grad():
            # generate random shit using normal dist
            x = torch.randn((n,1,64,64,64)).to(self.device)
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
        x = x.clip(min=0, max=1)
        x = torch.round(x).type(torch.uint8)
        x = (x * 255).type(torch.uint8)
        
        return x.reshape(64,64,64).to(self.device)
    
def sample_with_var(model,n,timesteps,device,var_schedule):
    model.eval()
    with torch.no_grad():
        # generate random shit using normal dist
        # x = torch.randn((n,1,64,64,64)).to(device)
        sample =  torch.randn((n,1,64,64,64)).to(device)
        # pred_x = None
        pre_sample = 0
        post_sample = 0
        for i in tqdm(reversed(range(0,timesteps)), position=0):
            t = (torch.ones(n)*i).long().to(device)
            print(i)
            pred_noise,_,pred_log_var,_ = p_mean_variance(model,sample,t,var_schedule)
            eps = torch.randn_like(sample)
            print(t)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(sample.shape) - 1)))
            )
            # pre
            if i == 0:
                pre_sample = sample
            sample = pred_noise + nonzero_mask * torch.exp(0.5*pred_log_var)*eps
            if i == 0:
                nonzero_mask = (
                (t != 99).float().view(-1, *([1] * (len(sample.shape) - 1)))
                )
                post_sample = pred_noise + nonzero_mask * torch.exp(0.5*pred_log_var)*eps

            # print(sample)
            # print(x)
    
    # print(sample)
    # print(x)
    # x = x.clip(min=0, max=1)
    # x = torch.round(x).type(torch.uint8)
    # sample = sample.clip(min=0, max=1)
    # sample = torch.round(sample).type(torch.uint8)
    # print('VALUE OF X PRE 255')
    # print(sample)
    # print(x)

    # sample = (sample * 255).type(torch.uint8)
    # print('VALUE OF X POST 255')
    # print(sample)
    return sample, pre_sample, post_sample




if __name__ == '__main__':

    device = 'cuda'
    path = '/dcs/pg22/u2294454/fresh_diffusion_2/3d_model_diff/epoch_models_cosine_var350_chair.pth'
    model = UNet(c_out=2,device=device).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    var = VarianceScheduler(timesteps=500,device=device)
    x, pre_x, post_x = sample_with_var(model,1,var.timesteps,device,var)
    x = x.to(device)
    pre_x = pre_x.to(device)
    post_x = post_x.to(device)
    # im = Image.fromarray(x.cpu().numpy())
    # im.save("your_file.jpeg")
    print(x.shape)
    torch.save(x,'model_test.pt')
    # torch.save(pre_x,'pre_model_test.pt')
    # torch.save(post_x,'post_model_test.pt')

    
    ##############################################################################################
    def img_gen(file,strin):
        x = torch.load(file)
        # transform = T.ToPILImage()
        print(x)
        x = x.clip(min=0, max=1)
        x = torch.round(x).type(torch.uint8)
        x = (x * 255).type(torch.uint8)
        # x = ((x+1)*127.5).clamp(0, 255).to(torch.uint8)
        torch.reshape(x,(64,64,64))
        # convert the tensor to PIL image using above transform
        # img = transform(x)
        cuts = [0,5,10,15,20,25,30,35,40,45,50,55,60]

        for ctr, cut in enumerate(cuts):
            iamg = x.transpose(0,2)[ctr].cpu().numpy()
            print(iamg)
            print(iamg.shape)
            iamg = np.reshape(iamg,(64,64))
            # print(transform(x[ctr]))
            print('#################################################')
            image = Image.fromarray(iamg, 'L')  # 'L' mode is for grayscale images

    # Save the image
            image.save(strin+'_bin2_new_img_'+str(ctr)+'.JPG')
        # img.save("models_"+str(ctr)+".jpg")
    # display the PIL image
    img_gen('model_test.pt','x')
    # img_gen('pre_model_test.pt', 'pre_x')
    # img_gen('post_model_test.pt', 'post_x')
    print('ok')
    