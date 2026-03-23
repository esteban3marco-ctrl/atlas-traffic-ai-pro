import torch
import os

path = r'D:\ATLAS\atlas-traffic-ai-pro\checkpoints_extended\atlas_best.pt'
try:
    checkpoint = torch.load(path, map_location='cpu')
    print("State Dict Keys:")
    if 'online_net' in checkpoint:
        keys = sorted(checkpoint['online_net'].keys())
        for key in keys:
            print(f"{key}: {checkpoint['online_net'][key].shape}")
    else:
        keys = sorted(checkpoint.keys())
        for key in keys:
            print(f"{key}: {checkpoint[key].shape}")
except Exception as e:
    print(f"Error loading {path}: {e}")
