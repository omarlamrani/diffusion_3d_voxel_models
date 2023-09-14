from unet_modules import UNetModel
from var_schedule import *
from tqdm import tqdm
import torch
from hybrid_loss import *
from data_load import load_3d_model
from visualize import visualize_model
import os
from os.path import abspath
import random
import argparse

def sample_3d_model():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs",
                        default=502,
                        help="amout of passes through the dataset",
                        type=int)

    parser.add_argument("--device",
                        default='cuda',
                        help="choose between cpu or cuda(gpu)",
                        type=str)

    parser.add_argument("--sched",
                        default='cosine',
                        help="choose between cosine or linear schedule",
                        type=str)

    parser.add_argument("--timesteps",
                        default=750,
                        help="choose noising timesteps",
                        type=int)

    parser.add_argument("--s_type",
                        default='dataset',
                        help="choose between 'model' or 'dataset' sampling ",
                        type=str)
    
    parser.add_argument("--loss",
                        default='mse',
                        help="choose between 'mse' or 'hybrid' sampling ",
                        type=str)
    
    parser.add_argument("--object",
                        default=None,
                        help="choose between all 6 classes (monitor,desk,chair,bathtub,toilet,bed) ",
                        type=str)

    args = parser.parse_args()

    epochs = args.epochs
    device = args.device
    sched_type = args.sched
    timesteps = args.timesteps
    sample_type = args.s_type
    object = args.object
    loss = args.loss

    if sample_type == 'dataset':

        path = 'voxel_'+object+'s.npy'
        dataset = np.load(path,mmap_mode='r')
        rand_idx = random.randint(0,len(dataset)-1)
        rand_model = dataset[rand_idx]
        visualize_model(rand_model)

    elif sample_type == 'model':

        if loss == 'mse':

            cwd = os.getcwd()
            path = cwd+'/GOOD_MSE_MODEL.pth'
            model = UNetModel(device=device).to(device)
            model.load_state_dict(torch.load(path,map_location=device))
            model.eval()
            var = VarianceScheduler(timesteps=timesteps,device=device,type=sched_type)
            x = sample(model,timesteps,device,var)
            x = x.to(device)
            print('sample done')
            print(x.shape)
            torch.save(x,'mse_model_sample.pt')


        elif loss == 'hybrid':

            cwd = os.getcwd()

            if object == None:
                path = cwd+'/GOOD_VAR_MODEL.pth'
                model = UNetModel(device=device,channels_out=2).to(device)
                model.load_state_dict(torch.load(path,map_location=device))
                model.eval()
                var = VarianceScheduler(timesteps=timesteps,device=device,type=sched_type)
                x = sample_with_var(model,timesteps,device,var)
                x = x.to(device)
                print('sample done')
                
                torch.save(x,'hybrid_model_sample.pt')

            else:
                path = cwd+'/GOOD_COND.pth'
                model = UNetModel_conditional(device=device,channels_out=2,num_classes=6).to(device)
                model.load_state_dict(torch.load(path,map_location=device))
                model.eval()
                
                
                var = VarianceScheduler(timesteps=timesteps,device=device,type=sched_type)
                label_map = {
                    'bathtub':0,
                    'monitor':1,
                    'bed':2,
                    'chair':3,
                    'toilet':4,
                    'desk':5
                }

                y = torch.tensor([label_map[object]]).to(device)
                
                x = sample_with_var(model,timesteps,device,var,y=y)
                x = x.to(device)
                
                print('sample done')
                
                torch.save(x,'cond_model_sample'+object+'.pt')

        else:
            print('Please choose a suitable loss type (mse) or (hybrid)')



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
            pred_noise,_,pred_log_var,_ = p_mean_var(model,sample,t,var_schedule,y)
            eps = torch.randn_like(sample)
            # print(t)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(sample.shape) - 1)))
            )

            sample = pred_noise + nonzero_mask * torch.exp(0.5*pred_log_var)*eps

    return sample




if __name__ == '__main__':

    sample_3d_model()
    # exit(0)
    # device = 'cuda'
    # cwd = os.getcwd()
    # path = cwd+'/epoch_model_cosine_var60_uncond.pth'
    # data = torch.load(path, map_location=device)

    # model = UNetModel_conditional(device=device,channels_out=2,num_classes=6).to(device)
    # model.load_state_dict(torch.load(path,map_location=device))
    # model.eval()
    # var = VarianceScheduler(timesteps=500,device=device,type='cosine')

    # label_map = {
    #     'bathtub':0,
    #     'monitor':1,
    #     'bed':2,
    #     'chair':3,
    #     'toilet':4,
    #     'desk':5
    # }

    # y = torch.tensor([label_map['desk']]).to(device)
    # x = sample_with_var(model,1,var.timesteps,device,var,y=y)
    # x = x.to(device)
    # print('sample done')
    # print(x.shape)
    # torch.save(x,'model_hybrid_bathtub.pt')
