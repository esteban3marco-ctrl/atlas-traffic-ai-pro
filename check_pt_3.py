import torch
import os

path = r'D:\ATLAS\atlas-traffic-ai-pro\checkpoints_extended\atlas_best.pt'
try:
    checkpoint = torch.load(path, map_location='cpu')
    print("CONFIG:")
    if 'config' in checkpoint:
        print(f"  {checkpoint['config']}")
    else:
        print("  No config in checkpoint")
    
    net = checkpoint.get('online_net', checkpoint)
    print("KEYS (First 10):")
    for k in sorted(net.keys())[:20]:
        print(f"  {k}: {net[k].shape}")
except Exception as e:
    print(f"Error: {e}")
