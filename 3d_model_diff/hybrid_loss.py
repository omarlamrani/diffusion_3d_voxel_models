import sys
from unet_modules import UNetModel
from torch import optim
import torch.nn as nn
from var_schedule import *
from tqdm import tqdm
from data_load import load_3d_model
import torch
import numpy as np 
from label_data import *

''' This class took multiple elements from:
    --> Improved DDPM by Nichol and Dhariwal
    --> Aleksa Gordic - Ultimate Guide to Diff Models'''

''' Most elements had to be highly modified to fit along with our work'''

class HybridLoss():
    def __init__(self,device):
           self.device = device

    def calculate_loss(self,model,x_0,x_t,t,sched,mean_noise,var_noise,noise,y=None):
        locked_mean_only_var = torch.cat([mean_noise.detach(),var_noise], dim=1).to(self.device)

        l_vlb, pred_x_0 = vlb(model,x_0,x_t,t,locked_mean_only_var,sched,y)
        l_vlb.to(self.device)
        pred_x_0.to(self.device)

        rescaled_l_vlb = (l_vlb * sched.timesteps / 1000).to(self.device)

        mse = ((noise - mean_noise) ** 2).to(self.device)
        full_mse = mse.mean(dim=list(range(1, len(mse.shape)))).to(self.device)

        return (full_mse + rescaled_l_vlb).mean()

def vlb (model,x_0, x_t, t,locked_output,var,y=None):

    ''' Get VLB'''

    (true_mean, true_var, true_log_var) = q_mean_var(x_0, x_t, t,var)
        
    model_mean, true_var, model_log_variance,eps_from_x_0 = p_mean_var(model, x_t, t,var,y)
    
    kl = normal_kl(
        true_mean, true_log_var, model_mean, model_log_variance
    )
    kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)

    

    decoder_nll = -discretized_gaussian_log_likelihood(
        x_0, means=model_mean, log_scales=0.5 * model_log_variance
    )
    decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.0)

    output = torch.where((t == 0), decoder_nll, kl) 
    return output, eps_from_x_0

def q_mean_var(x_0, x_t, t,var_schedule):
        ''' Compute both the mean and variance '''

        posterior_mean = (
             _extract_into_tensor((var_schedule.betas * torch.sqrt(var_schedule.alphas_cumprod_prev) / (1.0 - var_schedule.cumprod_alphas)),t,x_t.shape,var_schedule)
                * x_0
            + _extract_into_tensor((1.0 - var_schedule.alphas_cumprod_prev) * torch.sqrt(var_schedule.alphas) / (1.0 - var_schedule.cumprod_alphas),t,x_t.shape,var_schedule) 
                * x_t)
        posterior_variance = _extract_into_tensor(
             (var_schedule.betas * (1.0 - var_schedule.alphas_cumprod_prev) / (1.0 - var_schedule.cumprod_alphas)),
             t, x_t.shape,var_schedule)
        posterior_log_variance_clipped = _extract_into_tensor(
             var_schedule.log_posterior_variance,
             t, x_t.shape,var_schedule)

        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

def p_mean_var(model, x_t, t,var_schedule,y=None):


        if not y == None:
            model_output = model(x_t, t,y)
        else:
            model_output = model(x_t, t)
             

        model_output, model_var_values = torch.split(model_output, 1, dim=1)
        min_log = _extract_into_tensor(
            var_schedule.log_posterior_variance, t, x_t.shape,var_schedule
        )
        max_log = _extract_into_tensor(torch.log(var_schedule.betas), t, x_t.shape,var_schedule)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        eps_from_x_0 = (
                _extract_into_tensor(var_schedule.sqrt_inv_alphas_cumprod, t, x_t.shape,var_schedule) * x_t
                - _extract_into_tensor(var_schedule.sqrt_inv_alphas_cumprod_minus_one, t, x_t.shape,var_schedule) * model_output
        )
        model_mean, _, _ = q_mean_var(
            eps_from_x_0, x_t, t,var_schedule
        )

        return (model_mean, model_variance, model_log_variance, eps_from_x_0)

''' The functions down from here were taken from the Improved DDPM paper'''

def normal_kl(mean1, logvar1, mean2, logvar2):

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs

def approx_standard_normal_cdf(x):

    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def _extract_into_tensor(arr, timesteps, broadcast_shape,var_schedule):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    This function has been taken from the Improved DDPM page

    It solved the issues we were having, but we can't credit it as ours
    """
    res = arr.to(device=var_schedule.device)[timesteps].float() #cna be remove if all are torch
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

