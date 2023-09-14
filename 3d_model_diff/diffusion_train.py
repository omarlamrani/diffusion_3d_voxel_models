import sys
from base_unet_3d_new import UNetModel,EMA
from torch import optim
import torch.nn as nn
from var_schedule import *
from tqdm import tqdm
from label_data import *
import copy
from data_utils import load_3d_model
from hybrid_loss import *

def train_model():
    # epochs = 502
    # gpu = 'cuda'
    # model = UNet(device=gpu).float().to(gpu)
    # optimizer = optim.AdamW(model.parameters(),lr=3e-4)
    # mse = nn.MSELoss() #Will need variational lower bound frop variance prediction
    # var_schedule = VarianceScheduler(timesteps=1000,device=gpu)
    # dataloader = load_3d_model('/dcs/pg22/u2294454/fresh_diffusion/voxel_beds.npy',mode='train')
    # # prog = tqdm(dataloader)
    # epoch_loss = 10
    # for epoch in range(epochs):
    #     # progress = tqdm()
    #     print(f'==> EPOCH N*{epoch}')
    #     for i, vox_models in enumerate(dataloader):
    #         vox_models = vox_models.to(gpu)
    #         t = var_schedule.timesteps_sampling(vox_models.shape[0]).to(gpu)
    #         x_t, noise = var_schedule.forward_diffusion_sample(vox_models,t)
    #         # print(x_t.dtype)
    #         pred_noise = model(x_t.float(),t)
    #         # print(noise.dtype)
    #         # print(pred_noise.dtype)
    #         loss = mse(noise,pred_noise)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # prog.set_postfix(MSE=loss.item())
    #         epoch_loss = loss
    #     print(epoch_loss)
    #     if epoch%250 == 0:
    #         torch.save(model.state_dict(),'epoch_'+str(epoch)+'_beds_gpu_'+str(epoch_loss)+'.pth')
    
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
                        default='cosine',
                        help="choose between 'cosine' or 'linear' schedule",
                        type=str)
    
    parser.add_argument("--timesteps",
                        default=500,
                        help="choose noising timesteps",
                        type=int)
    
    parser.add_argument("--loss",
                        default='mse',
                        help="choose between 'mse' or 'hybrid' loss ",
                        type=str)
    
    parser.add_argument("--ema",
                        default=False,
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

    loss_t = []

    if loss_type == 'mse':
        loss_func = nn.MSELoss()
        model = UNetModel(device=device).float().to(device)
    else:
        loss_func = HybridLoss(device=device)
        model = UNetModel(device=device,channels_out=2).float().to(device)

    # data_path = 'C:\\Users\\lamra\\OneDrive\\test\\Desktop\\fresh_diffusion_2\\voxel_beds.npy'
    data_path = '/dcs/pg22/u2294454/fresh_diffusion_2/voxel_chairs.npy'
    dataloader = load_3d_model(data_path,mode='train')

    # folder = r"/dcs/pg22/u2294454/fresh_diffusion_2/image_dataset"
    

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
            if epoch < 10:
                copy_loss_t = torch.tensor(loss_t).cpu().numpy()
                np.save('mse_'+str(epoch)+'_loss.npy',copy_loss_t)
            for i, vox_models in enumerate(dataloader):
                vox_models = vox_models.float().to(device)
                t = var_schedule.sample_timesteps(vox_models.shape[0]).to(device)
                x_t, noise = var_schedule.fwd_diff_t(vox_models,t)
                pred_noise = model(x_t.float(),t)
                loss = loss_func(noise,pred_noise)
                loss_t.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema_option: ema.step_ema(mod_copy_ema,model)
                epoch_loss = loss
                # print(i)
            print(epoch_loss)
            if (epoch%100 == 0):
                torch.save(model.state_dict(),'new_e'+str(epoch)+'_t'+str(timesteps)+'_chair_noEMA_MSE_cos.pth')
                # torch.save(mod_copy_ema.state_dict(), 'epoch_ema_cosine_var'+str(epoch)+'_uncond'+'.pth')

    else:
        for epoch in range(epochs):
            print(f'==> EPOCH N*{epoch} <==')
            if epoch < 10:
                copy_loss_t = torch.tensor(loss_t).cpu().numpy()
                np.save('hybrid_'+str(epoch)+'_loss.npy',copy_loss_t)
            # torch.save(model.state_dict(),'new_e'+str(epoch)+'_t500_chair_noEMA_HYBRID_cos.pth')
            for i, vox_models in enumerate(dataloader): 
                vox_models = vox_models.float().to(device)
                t = var_schedule.sample_timesteps(vox_models.shape[0]).to(device)
                # forward q_sample
                x_t, noise = var_schedule.fwd_diff_t(vox_models,t)
                x_t.to(device)
                noise.to(device)
                # print(i)
                # KL_RESCALED LOSS_TYPE ==> HYBRID LOSS
                mod_out = model(x_t.float(),t).to(device) # 1st 3 channels (dim1) = mean, last 3 = var
                # print(mod_out.shape) # (B,6,64,64)

                mean_noise, var_noise = torch.split(mod_out,1,dim=1)
                mean_noise.to(device)
                var_noise.to(device)
                # print(mean_noise.shape)
                # print(var_noise.shape)
                loss = loss_func.calculate_loss(model,vox_models,x_t,t,var_schedule,mean_noise,var_noise,noise)
                # if i%5 == 0:
                loss_t.append(loss)
                # # update only the var not the mean to avoid correlated predictions
                # locked_mean_only_var = torch.cat([mean_noise.detach(),var_noise], dim=1).to(device)

                # l_vlb, pred_x_0 = vb_term(model,vox_models,x_t,t,locked_mean_only_var,var_schedule)
                # l_vlb.to(device)
                # pred_x_0.to(device)
                # # print(l_vlb)

                # rescaled_l_vlb = (l_vlb * var_schedule.timesteps / 1000).to(device)
                # # print(rescaled_l_vlb)

                # mse = ((noise - mean_noise) ** 2).to(device)
                # full_mse = mse.mean(dim=list(range(1, len(mse.shape)))).to(device)

                # loss = (full_mse + rescaled_l_vlb).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema_option: ema.step_ema(mod_copy_ema,model)
                epoch_loss = loss
            print(epoch_loss)


            if (epoch%50 == 0 and not epoch == 0):
                torch.save(model.state_dict(),'new_2e'+str(epoch)+'_t500_chair_noEMA_HYBRID_cos.pth')
                # torch.save(mod_copy_ema.state_dict(), 'epoch_ema_cosine_var'+str(epoch)+'_uncond'+'.pth')
                # torch.save(optimizer.state_dict(), 'epoch_opt_cosine_var'+str(epoch)+'_uncond'+'.pth')

if __name__ == '__main__':
    train_model()
