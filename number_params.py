import torch

# Load the checkpoint file
checkpoint = torch.load("logs/DNCNN/lightning_logs/out.ckpt", map_location="cpu")

# Check the keys inside the checkpoint
# print(checkpoint.keys())

state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

total_params = sum(p.numel() for p in state_dict.values())
print(f"Total number of parameters: {total_params}")