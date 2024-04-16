import torch

pth_ckpt_orig = 'logs/DNCNN/lightning_logs/DNCNN_R_nch_3_seed_0_ljr_0.01_jt_nonsymmetric_nit_20_loss_l1_lr_0.0001_sigma_0-50_single/'
pth_target = 'logs/DNCNN/lightning_logs/'

# Initial checkpoint file (.ckpt extension)
pth_ckpt_orig_spe = pth_ckpt_orig+'checkpoints/epoch=2999-step=51000.ckpt'

# Load checkpoint
ckpt_orig = torch.load(pth_ckpt_orig_spe, map_location=torch.device('cpu'))

# Remove 'denoiser.model.' in the keys of the dict
new_weights = dict((key.replace('denoiser.model.', ''), value) for (key, value) in ckpt_orig['state_dict'].items())
new_dict = {'state_dict': new_weights}

# Save the new dictionary
torch.save(new_dict, pth_target+'out.ckpt')
