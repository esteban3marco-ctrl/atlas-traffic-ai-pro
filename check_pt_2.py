import torch
import os

path = r'D:\ATLAS\atlas-traffic-ai-pro\checkpoints_extended\atlas_best.pt'
try:
    checkpoint = torch.load(path, map_location='cpu')
    net = checkpoint.get('online_net', checkpoint)
    print("SHARED KEYS:")
    for k in sorted(net.keys()):
        if k.startswith('shared'):
            print(f"{k}: {net[k].shape}")
    print("STREAMS:")
    for k in sorted(net.keys()):
        if 'stream' in k:
            print(f"{k}: {net[k].shape}")
except Exception as e:
    print(f"Error: {e}")
