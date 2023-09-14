import sys
from unet_modules import UNetModel_conditional
from torch import optim
import torch.nn as nn
from var_schedule import *
from tqdm import tqdm
from label_data import *
import copy
from data_load import load_3d_model
from hybrid_loss import *


def train_model():
    
    import argparse

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
                        default='linear',
                        help="choose between cosine or linear schedule",
                        type=str)
    
    parser.add_argument("--timesteps",
                        default=750,
                        help="choose noising timesteps",
                        type=int)
    
    parser.add_argument("--loss",
                        default='mse',
                        help="choose between 'mse' or 'hybrid' loss ",
                        type=str)
    
    parser.add_argument("--ema",
                        default=True,
                        help="use EMA or not ",
                        type=bool)

    args = parser.parse_args()

    epochs = args.epochs
    device = args.device
    sched_type = args.sched
    timesteps = args.timesteps
    loss_type = args.loss
    ema_option = args.ema

    mse = None
    model = None
    loss_func = None

    if loss_type == 'mse':
        loss_func = nn.MSELoss().to(device=device)
        model = model = UNetModel_conditional(device=device,num_classes=6).float().to(device)
    else:
        loss_func = HybridLoss(device=device)
        model = UNetModel_conditional(device=device,channels_out=2,num_classes=6).float().to(device)

    data_path = os.getcwd()
    files = gather_npy_filenames(data_path)
    labeled_data = label_gathered_datasets(files,data_path)
    dataloader = label_dataloader(labeled_data)

    # folder = r"/dcs/pg22/u2294454/fresh_diffusion_2/image_dataset"
    print('data labelled')

    # files = gather_npy_filenames('/dcs/pg22/u2294454/fresh_diffusion_2')
    # labeled_data = label_gathered_datasets(files)
    # dataloader = label_dataloader(labeled_data)
        
    # model = UNet_conditional(device=device,c_out=2,num_classes=6).float().to(device)
    optimizer = optim.AdamW(model.parameters(),lr=3e-4)
    var_schedule = VarianceScheduler(timesteps=timesteps,device=device,type=sched_type)

    epoch_loss = float('inf')

    ema = None
    mod_copy_ema = None

    print('setup')

    if ema_option:
        ema = EMA(beta=0.99)
        mod_copy_ema = copy.deepcopy(model).eval().requires_grad_(False)

    print('ema')

    if loss_type == 'mse':
        for epoch in range(epochs):
            print(f'==> EPOCH N*{epoch}')
            for i,(vox_models, labels) in enumerate(dataloader):
                vox_models = vox_models.float().to(device)
                if np.random.random() < 0.1:
                    labels = torch.tensor([2310]).to(device)
                labels = labels.to(device)
                t = var_schedule.sample_timesteps(vox_models.shape[0]).to(device)
                x_t, noise = var_schedule.fwd_diff_t(vox_models,t)
                pred_noise = model(x_t.float(),t,labels)
                loss = loss_func(noise,pred_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema_option: ema.step_ema(mod_copy_ema,model)
                epoch_loss = loss
            if epoch%60 == 0:
                torch.save(model.state_dict(),'model_e'+str(epoch)+'_t'+str(timesteps)+'_'+loss_type+'_'+sched_type+'.pth')
                if ema_option:
                    torch.save(mod_copy_ema.state_dict(), 'ema_e'+str(epoch)+'_t'+str(timesteps)+'_'+loss_type+'_'+sched_type+'.pth')
    else:

        for epoch in range(epochs):
            print(f'==> EPOCH N*{epoch} <==')
            for i, (vox_models, labels) in enumerate(dataloader):
                vox_models = vox_models.float().to(device)
                if np.random.random() < 0.1:
                    labels = torch.tensor([2310]).to(device)
                labels = labels.to(device)
                t = var_schedule.sample_timesteps(vox_models.shape[0]).to(device)
                x_t, noise = var_schedule.fwd_diff_t(vox_models,t)
                x_t.to(device)
                noise.to(device)
                mod_out = model(x_t.float(),t,labels).to(device) 

                mean_noise, var_noise = torch.split(mod_out,1,dim=1)
                mean_noise.to(device)
                var_noise.to(device)

                loss = loss_func.calculate_loss(model,vox_models,x_t,t,var_schedule,mean_noise,var_noise,noise,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema_option: ema.step_ema(mod_copy_ema,model)
                epoch_loss = loss
            print(epoch_loss)


            if epoch%60 == 0:
                torch.save(model.state_dict(),'model_e'+str(epoch)+'_t'+str(timesteps)+'_'+loss_type+'_'+sched_type+'.pth')
                if ema_option:
                    torch.save(mod_copy_ema.state_dict(), 'ema_e'+str(epoch)+'_t'+str(timesteps)+'_'+loss_type+'_'+sched_type+'.pth')

if __name__ == '__main__':
    train_model()
