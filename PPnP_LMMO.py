# %%
import sys
import os
from natsort import os_sorted


import torch
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt
sys.path.append("../bregman_sampling/Prox_GSPnP/PnP_restoration")

from utils.utils_restoration import Measures
import utils.utils_restoration as ur
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from utils.save_plots import save_plots, var2gray
from utils.save_progress import save_progress
import random 
import wandb
import torch.nn as nn

import sampling_tools as st

# Cuda 
device = 'cuda:0'
sys.path.append("../GS_denoising/")

#
#%%


# %%

def PPnP_LMMO():

    config_defaults = {
        'dataset_path': '../bregman_sampling/BregmanPnP/datasets',
        'dataset_name': 'set3c',
        'select_image' : 2,
        'patch_size': 256,

        'inverse_problem': 'deblurring',
        'alpha_poisson': 20,
        
        'iterations': 1000000,
        'n_samples_saved': 2000,
        'delta_frac': 20,
        'noise_lvl_denoiser': 20,
        'blur_mode': 'levin',
        'kernel_path' : "../bregman_sampling/Prox_GSPnP/PnP_restoration/kernels",
        'kernel_name' : "Levin09.mat",
        'kernel_size': 9,
        'kernel_std' : 3, # 3.0 if blur_mode == 'Gaussian' else 0.5
        'kernel' : 4, 
        'alpha': 1.0,
        'C_upper_lim': 1.,
        'C_lower_lim': 0.,
        'PnP_method': False,
        'PPnP_method': True,
        'flip_n_rot': False,

        'device': device,
        
    }

    path_save = './results/Results_PPnP_LMMO_' + config_defaults['inverse_problem'] + '_im{}_alpha_poisson_{}_delta_frac{}_noise_lvl_{}_alpha_{}_rot_n_flip_{}/'.format(config_defaults['select_image'],  config_defaults['alpha_poisson'], config_defaults['delta_frac'], config_defaults['noise_lvl_denoiser'], config_defaults['alpha'], config_defaults['flip_n_rot'])
    exp_name = 'Results_PPnP_LMMO_' + config_defaults['inverse_problem'] + '_im{}_alpha_poisson_{}_delta_frac{}_noise_lvl_{}_alpha_{}_rot_n_flip_{}'.format(config_defaults['select_image'], config_defaults['alpha_poisson'],config_defaults['delta_frac'], config_defaults['noise_lvl_denoiser'], config_defaults['alpha'], config_defaults['flip_n_rot'])

    wandb.init(entity='bloom', project="tk_Euclidean_experiments_im"+ str(config_defaults['select_image']) +"_"+ config_defaults['blur_mode'] +"_blur", name = exp_name , config=config_defaults, save_code=True)
    config = wandb.config

    #set data paths
    if not os.path.exists(path_save):
    # Create a new directory because it does not exist 
        os.makedirs(path_save)
        print("The new directory is created!")

    # prepare data paths
    input_path = os.path.join(config.dataset_path, config.dataset_name, '0')
    input_paths = os_sorted(
        [os.path.join(input_path, p) for p in os.listdir(input_path)])
    

    # load image (could also create a loop over input_paths)
    image_index = config.select_image
    input_im_uint = ur.imread_uint(input_paths[image_index])
    image_name = os.path.splitext(os.path.basename(input_paths[image_index]))[0]
    if config.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
        input_im_uint = ur.crop_center(input_im_uint, config.patch_size, config.patch_size)
    input_im = np.float32(input_im_uint / 255.0)

    # load kernels
    np.random.seed(seed=123)
    torch.manual_seed(123)

    input_tensor = ur.array2tensor(input_im).to(device)

    if config.inverse_problem == 'deblurring':
        if config.blur_mode == 'uniform':
            k = np.float32((1 / config.kernel_size**2) * np.ones((config.kernel_size, config.kernel_size)))
            k_tensor = torch.tensor(k).to(device)
            blur_im = ndimage.convolve(input_im, np.expand_dims(k, axis=2), mode="wrap")
            blur_im = ur.array2tensor(blur_im).to(device)
            # k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
        elif config.blur_mode == 'motion':
            kernel = MotionBlurOperator(kernel_size=config.kernel_size, intensity=config.kernel_std, device=device)
            k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
        elif config.blur_mode == 'Gaussian':
            kernel = GaussialBlurOperator(kernel_size=config.kernel_size, intensity=config.kernel_std, device=device)
            k_tensor = kernel.get_kernel().to(device, dtype=torch.float)

        elif config.blur_mode == 'levin':
            k_list = st.load_kernels(config)
            k_tensor = k_list[config.kernel].to(device, dtype=torch.float)

        k = k_tensor.clone().detach().cpu().numpy()
        k = np.squeeze(k)
        blur_im = ndimage.convolve(input_im, np.expand_dims(k, axis=2), mode="wrap")
        blur_im = ur.array2tensor(blur_im).to(device)
    else:
        input_tensor = ur.array2tensor(input_im).to(device)
        blur_im = input_tensor.clone()

    # Degrade image

    np.random.seed(seed=123)
    torch.manual_seed(123)

    noisy_im = torch.poisson(torch.maximum(config.alpha_poisson * blur_im, torch.zeros_like(input_tensor)))

    # plt.figure(0)
    # plt.imshow(np.uint8((noisy_im[0].cpu().numpy()).transpose(1,2,0)/np.max(noisy_im[0].cpu().numpy())*255))
    # plt.axis("off")
    # plt.savefig(path_save + "noisy.png", bbox_inches='tight')

    # print(ur.psnr(input_im, ur.tensor2array(noisy_im.cpu()/np.max(noisy_im.cpu().numpy()))))

    #%% Load denoiser
    class DnCNN(nn.Module):
        def __init__(self, nc_in, nc_out, depth, act_mode, bias=True, nf=64):
            super(DnCNN, self).__init__()

            self.depth = depth

            self.in_conv = nn.Conv2d(nc_in, nf, kernel_size=3, stride=1, padding=1, bias=bias)
            self.conv_list = nn.ModuleList(
                [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(self.depth - 2)])
            self.out_conv = nn.Conv2d(nf, nc_out, kernel_size=3, stride=1, padding=1, bias=bias)

            if act_mode == 'R':  # Kai Zhang's nomenclature
                self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

        def forward(self, x_in):

            x = self.in_conv(x_in)
            x = self.nl_list[0](x)

            for i in range(self.depth - 2):
                x_l = self.conv_list[i](x)
                x = self.nl_list[i + 1](x_l)
                
            return self.out_conv(x) + x_in

    n_ch = 3  # number of channels in the image, 3 if colour, 1 if grayscale
    str_desc = '3 channels (color imaging)' if n_ch==3 else '1 channel (grayscale imaging)'  # For printing
    name_file = 'logs/DNCNN/lightning_logs/out.ckpt'
    model_weights = torch.load(name_file, map_location=torch.device('cpu'))  # Remark: map_location=torch.device('cpu') could be removed since we are using GPUs, but this code will work as well if no GPU is available.

    avg, bn, depth = False, False, 20
    net = DnCNN(n_ch, n_ch, depth, 'R')

    model = nn.DataParallel(net)
    model = model.cuda()
        
    model.module.load_state_dict(model_weights["state_dict"], strict=True)
    model = model.eval() 

    print('Trained DnCNN loaded, with '+str_desc)

    ## write a denoise function like we had for dncnn
    denoise = lambda x: (x - model(x)).detach() 


    # %%
    # Gradient of Likelihood

    from utils import utils_sr

    def At(y, k_tensor):
        """
        Calculation A*x with A the linear degradation operator 
        """
        return utils_sr.Gt(y, k_tensor, sf=1)

    def A(y, k_tensor):
        """
        Calculation A*x with A the linear degradation operator 
        """
        return utils_sr.G(y, k_tensor, sf=1)

    # grad of likelihood
    def grad_f(z):
        # inv_z = 1/(alpha_poisson*x + beta)
        # return alpha_poisson*(torch.ones_like(x) - noisy_im*inv_z)
        inv_z = 1/(A(z*config.alpha_poisson, k_tensor) + beta)
        return config.alpha_poisson*(At(torch.ones_like(z) - noisy_im*inv_z, k_tensor))

    def projbox(x,lower: torch.Tensor,upper: torch.Tensor)->torch.Tensor:
        return torch.clamp(x, min = lower, max=upper)

    # %%

    from tqdm import tqdm
    from sampling_tools import welford

    # Algorithm parameters
    image_mean = config.alpha_poisson*torch.mean(input_tensor)
    beta = image_mean * 0.02
    L = 1.0
    AAT_norm = 1
    L_y =  config.alpha_poisson**2*(torch.max(noisy_im)/beta**2)*AAT_norm
    alpha = config.alpha
    eps = (config.noise_lvl_denoiser/255)**2
    max_lambd = 1.0/((2.0*alpha*L)/eps+4.0*L_y)
    lambd_frac = 0.99
    lambd = max_lambd*lambd_frac

    C_upper_lim = torch.tensor(config.C_upper_lim).to(device)
    C_lower_lim = torch.tensor(config.C_lower_lim).to(device)

    delta_max = 1.0/(L/eps+L_y)# (1.0/3.0)/((alpha*L)/eps+L_y+1/lambd)
    print("Stepsize PnP-ULA: ", (1.0/3.0)/((alpha*L)/eps+L_y+1/lambd))
    print("Stepsize PPnP-ULA: ", delta_max)
    delta_frac = config.delta_frac
    delta = delta_max*delta_frac

    # reset number of iterations
    N = config.iterations

    #pick a starting point
    # Xk = torch.clamp(At(noisy_im, k_tensor), 1e-4, 1)
    Xk = torch.clamp(noisy_im/torch.max(noisy_im), 1e-4, 1)

    #init the chain
    n_samples = config.n_samples_saved
    thinned_trace_counter = 0
    thinning_step = np.int64(N/n_samples)
    samples = []

    #record psnr
    measures = Measures(device=device)
    psnr_stats = []
    ssim_stats = []
    lpips_stats = []
    nrmse_stats = []
    psnr_stats.append(measures.psnr(input_tensor, Xk))
    ssim_stats.append(measures.ssim(input_tensor, Xk))
    lpips_stats.append(measures.lpips(input_tensor, Xk))
    nrmse_stats.append(measures.nrmse(input_tensor, Xk))

    wandb.log({"PSNR" : psnr_stats[0],
               "SSIM" : ssim_stats[0],
               "LPIPS" : lpips_stats[0],
               "Noisy image" : wandb.Image((noisy_im/torch.max(noisy_im)).cpu().squeeze(), caption="Noisy image"),
               "Posterior mean" : wandb.Image(Xk.cpu().squeeze(), caption="Rec. Image")})

    # init welford
    post_stats = welford(Xk)

    text_file = open(path_save + '/results.txt', "w+")
    text_file.write('Launch Reflected PnP-ULA with DRUNET!')
    text_file.write('Iterations: {}\n'.format(config.iterations))
    text_file.write('Stepsize: {}\n'.format(delta))
    text_file.write('alpha: {}\n'.format(config.alpha))
    text_file.write('level_noise_denoiser: {}\n'.format(config.noise_lvl_denoiser))
    text_file.write('Flip_n_Rotations: {}\n'.format(config.flip_n_rot))
    text_file.write('device: {}\n'.format(device))
    text_file.close()

    l = [0, 1, 2, 3]
    m = [0, 1, 2]

    #sampling loop
    for i in tqdm(range(N)):

        if config.flip_n_rot == True:
            # random rotate Xk
            # pick a random number 0-3
            select_rot = random.choice(l)
            # pick a random number 0-2
            select_flip = random.choice(m)
            # flip (or not)
            if select_flip: #not 0
                Xk_flip = torch.flip(Xk, dims=(-select_flip,))
            else:
                Xk_flip = Xk
            Xk_rot = torch.rot90(Xk_flip, k=select_rot, dims=(-1,-2))
            Dg_rot  = denoise(Xk_rot)
            # random rotate Dg back
            Dg_flip = torch.rot90(Dg_rot, k=-select_rot, dims=(-1,-2))
            # flip back
            if select_flip: #not 0
                Dg = torch.flip(Dg_flip, dims=(-select_flip,))
            else:
                Dg = Dg_flip
        else:
            Dg  = denoise(Xk)

        # CHANGE HERE
        if config.PPnP_method==True and config.PnP_method==False:
            Xk = projbox(Xk - delta * grad_f(Xk) - (alpha*delta/eps)*Dg + torch.sqrt(2 * delta) * torch.randn_like(Xk),C_lower_lim,C_upper_lim)
        elif config.PPnP_method==False and config.PnP_method==True:
            Xk = torch.abs(Xk - delta * grad_f(Xk) - (alpha*delta/eps)*Dg + delta/lambd*(projbox(Xk,C_lower_lim,C_upper_lim)-Xk) + torch.sqrt(2 * delta) * torch.randn_like(Xk))         
        else:
            raise TypeError('Choose a method PnP or PPnP, dear!')
        
        post_stats.update(Xk)
        psnr_stats.append(measures.psnr(input_tensor, post_stats.get_mean()))
        ssim_stats.append(measures.ssim(input_tensor, post_stats.get_mean()))
        lpips_stats.append(measures.lpips(input_tensor, post_stats.get_mean()))
        nrmse_stats.append(measures.nrmse(input_tensor, post_stats.get_mean()))

        if thinned_trace_counter == thinning_step-1:
            samples.append(Xk.cpu().numpy())
            thinned_trace_counter = 0
        else:
            thinned_trace_counter += 1


        if i%1000==0:
            wandb.log({"PSNR" : psnr_stats[-1],
                    "SSIM" : ssim_stats[-1],
                    "LPIPS" : lpips_stats[-1],
                    "Posterior mean" : wandb.Image(post_stats.get_mean().squeeze().cpu(), caption="Rec. Image"),
                    "Posterior std" : wandb.Image(np.sqrt(var2gray(post_stats.get_var())), caption="Posterior std")})
            # print("PSNR: ", psnr_stats[i+1])
            # print("SSIM: ", ssim_stats[i+1])
            # print("LPIPS: ", lpips_stats[i+1])

        if (i+1)%10000==0:

        
            text_file = open(path_save + '/results.txt', "a")
            text_file.write('Iteration [{}/{}] PSNR: {:.2f}, SSIM : {:.2f}, LPIPS: {:.2f} \n'.format(i + 1, N, psnr_stats[-1], ssim_stats[-1], lpips_stats[-1]))
            text_file.close()
            
            save_progress(psnr_stats, ssim_stats, lpips_stats, nrmse_stats, post_stats, np.asarray(samples), path_save)
            save_plots(psnr_stats, ssim_stats, lpips_stats, nrmse_stats, post_stats, path_save)

            # plt.figure(0)
            # plt.imshow(ur.tensor2uint(post_stats.get_mean()))
            # plt.axis("off")
            # plt.savefig(config.path_save + image_name + ".png", bbox_inches=None)
            # plt.close()

            # plt.figure(0)
            # plt.plot(psnr_stats[:(i+1)])
            # plt.savefig(config.path_save + "PSNR" + ".png", bbox_inches=None)
            # plt.close()

    wandb.finish()

#%%

if __name__ == '__main__': 

    PPnP_LMMO()