import sys
from image_unet import UNet
from torch import optim
import torch.nn as nn
from var_schedule import *
from tqdm import tqdm
# from voxelization import load_3d_model
# print(sys.version)

# def get_loss(model, x_0, t):
#     x_noisy, noise = forward_diffusion_sample(x_0, t)
#     noise_pred = model(x_noisy, t)
#     return F.l1_loss(noise, noise_pred)

# def train_model():


def vb_term (model,x_0, x_t, t, locked_output):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """

        (true_mean, true_var, true_log_var) = q_posterior_mean_variance(x_0, x_t, t)
        # print('mean, var, log_var')
        # print(true_mean.shape)
        # print(true_var.shape)
        # print(true_log_var.shape)
        
        model_mean, model_variance, model_log_variance, eps_from_x_0 = p_mean_variance(model, x_t, t)
        # print(model_mean.shape)
        # print(model_variance.shape)
        # print(model_log_variance.shape)
        # print(eps_from_x_0.shape)
        kl = normal_kl(
            true_mean, true_log_var, model_mean, model_log_variance
        )
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)

        

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl) # CAREFUL IF YOU DONT START AT 0
        return output, eps_from_x_0

def q_posterior_mean_variance(x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        # print(var_schedule.betas.shape)
        # print(np.sqrt(var_schedule.alphas_cumprod_prev).shape)
        # print(x_0.shape)
        posterior_mean = (
             _extract_into_tensor((var_schedule.betas * torch.sqrt(var_schedule.alphas_cumprod_prev) / (1.0 - var_schedule.cumprod_alphas)),t,x_t.shape)
                * x_0
            + _extract_into_tensor((1.0 - var_schedule.alphas_cumprod_prev) * torch.sqrt(var_schedule.alphas) / (1.0 - var_schedule.cumprod_alphas),t,x_t.shape) 
                * x_t)
        posterior_variance = _extract_into_tensor(
             (var_schedule.betas * (1.0 - var_schedule.alphas_cumprod_prev) / (1.0 - var_schedule.cumprod_alphas)),
             t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
             var_schedule.log_posterior_variance,
             t, x_t.shape)

        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

def p_mean_variance(model, x_t, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B, C = x_t.shape[:2]
        # print(f'C: {C}')

        model_output = model(x_t, t)

        model_output, model_var_values = torch.split(model_output, 3, dim=1)
        min_log = _extract_into_tensor(
            var_schedule.log_posterior_variance, t, x_t.shape
        )
        max_log = _extract_into_tensor(torch.log(var_schedule.betas), t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        eps_from_x_0 = (
                _extract_into_tensor(var_schedule.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(var_schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * model_output
        )
        model_mean, _, _ = q_posterior_mean_variance(
            x_0=eps_from_x_0, x_t=x_t, t=t
        )

        return (model_mean, model_variance, model_log_variance, eps_from_x_0)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
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
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

# def train_cycle(batch):
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=var_schedule.device)[timesteps].float() #cna be remove if all are torch
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# t = var_schedule.timesteps_sampling(vox_models.shape[0]).to(gpu)
# x_t, noise = var_schedule.forward_diffusion_sample(vox_models,t)


epochs = 502
gpu = 'cuda'
model = UNet(device=gpu,c_out=6).float().to(gpu)
optimizer = optim.AdamW(model.parameters(),lr=3e-4)
mse = nn.MSELoss() #Will need variational lower bound frop variance prediction
var_schedule = VarianceScheduler(timesteps=1000,device=gpu,type='cosine')
folder = r"/dcs/pg22/u2294454/fresh_diffusion_2/image_dataset"
dataloader  = image_processing(folder)
prog = tqdm(dataloader)
epoch_loss = 10
ctr = 0
for epoch in range(epochs):
    # progress = tqdm()
    print(f'==> EPOCH N*{epoch}')
    for vox_models,_ in dataloader:
        # print(ctr)
        ctr += 1
        vox_models = vox_models.float().to(gpu)
        t = var_schedule.timesteps_sampling(vox_models.shape[0]).to(gpu)
        # forward q_sample
        x_t, noise = var_schedule.forward_diffusion_sample(vox_models,t)
        x_t.to(gpu)
        noise.to(gpu)
        # print(x_t.dtype)
        # print(t.shape)
        # print(x_t.shape)
        # print(noise.shape)

        # KL_RESCALED LOSS_TYPE ==> HYBRID LOSS
        mod_out = model(x_t.float(),t).to(gpu) # 1st 3 channels (dim1) = mean, last 3 = var
        # print(mod_out.shape) # (B,6,64,64)

        mean_noise, var_noise = torch.split(mod_out,3,dim=1)
        mean_noise.to(gpu)
        var_noise.to(gpu)
        # print(mean_noise.shape)
        # print(var_noise.shape)

        # update only the var not the mean to avoid correlated predictions
        locked_mean_only_var = torch.cat([mean_noise.detach(),var_noise], dim=1).to(gpu)

        l_vlb, pred_x_0 = vb_term(model,vox_models,x_t,t,locked_mean_only_var)
        l_vlb.to(gpu)
        pred_x_0.to(gpu)
        # print(l_vlb)

        rescaled_l_vlb = (l_vlb * var_schedule.timesteps / 1000).to(gpu)
        # print(rescaled_l_vlb)

        mse = ((noise - mean_noise) ** 2).to(gpu)
        full_mse = mse.mean(dim=list(range(1, len(mse.shape)))).to(gpu)

        loss = (full_mse + rescaled_l_vlb).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('DONE')
       
        epoch_loss = loss
    print(epoch_loss)
    if epoch%250 == 0 and not epoch == 0:
        torch.save(model.state_dict(),'epoch_image_cosine_var'+str(epoch)+'_glacier_'+str(epoch_loss)+'.pth')