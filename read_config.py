import torch
path = r'D:\ATLAS\atlas-traffic-ai-pro\checkpoints_extended\atlas_best.pt'
checkpoint = torch.load(path, map_location='cpu')
if 'config' in checkpoint:
    c = checkpoint['config']
    print(f"H_DIMS:{c.get('hidden_dims')}")
    print(f"USE_NOISY:{c.get('use_noisy_nets')}")
else:
    print("NO_CONFIG")
