from base_unet_3d_new import UNetModel
from var_schedule import *
from tqdm import tqdm
import torch
from hybrid_loss import *
from data_utils import load_3d_model,visualize_model
import os
from os.path import abspath


def sample(model,timesteps,device,var_schedule): 
        
    ''' This functions takes care of sampling models trained to predict
    ONLY the mean of the noise '''

    model.eval()
    with torch.no_grad():
        x = torch.randn((1,1,64,64,64)).to(device)

        for i in tqdm(reversed(range(1,timesteps)), position=0):
            t = (torch.ones(1)*i).long().to(device)
            print(i)
            pred_noise = model(x,t)
            beta = var_schedule.betas[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)
            alpha = var_schedule.alphas[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)
            cumprod_alpha = var_schedule.cumprod_alphas[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)
            if i>=2:
                noise = torch.randn_like(x).to(device)
            else:
                noise = torch.zeros_like(x).to(device)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - cumprod_alpha))) * pred_noise) + torch.sqrt(beta) * noise
    model.train()
    x = x.clip(min=0, max=1)
    x = torch.round(x).type(torch.uint8)
    x = (x * 255).type(torch.uint8)
    
    return x.reshape(64,64,64).to(device)
    
def sample_with_var(model,timesteps,device,var_schedule,y=None):

    ''' This functions takes care of sampling models trained to predict
    the mean AND variance of the noise '''

    model.eval()
    with torch.no_grad():

        sample =  torch.randn((1,1,64,64,64)).to(device)

        for i in tqdm(reversed(range(0,timesteps)), position=0):
            t = (torch.ones(1)*i).long().to(device)
            pred_noise,_,pred_log_var,_ = p_mean_variance(model,sample,t,var_schedule,y)
            eps = torch.randn_like(sample)
            # print(t)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(sample.shape) - 1)))
            )

            sample = pred_noise + nonzero_mask * torch.exp(0.5*pred_log_var)*eps

    return sample




if __name__ == '__main__':

    device = 'cuda'
    cwd = os.getcwd()
    path = cwd+'/epoch_model_cosine_var60_uncond.pth'
    data = torch.load(path, map_location=device)

    model = UNetModel_conditional(device=device,channels_out=2,num_classes=6).to(device)
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()
    var = VarianceScheduler(timesteps=500,device=device,type='cosine')

    label_map = {
        'bathtub':0,
        'monitor':1,
        'bed':2,
        'chair':3,
        'toilet':4,
        'desk':5
    }

    y = torch.tensor([label_map['desk']]).to(device)
    x = sample_with_var(model,1,var.timesteps,device,var,y=y)
    x = x.to(device)
    print('sample done')
    print(x.shape)
    torch.save(x,'model_hybrid_bathtub.pt')
