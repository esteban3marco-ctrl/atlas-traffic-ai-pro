import torch
import torch.nn as nn
from atlas.networks import NoisyLinear

path = r'D:\ATLAS\atlas-traffic-ai-pro\checkpoints_extended\atlas_best.pt'
checkpoint = torch.load(path, map_location='cpu')
sd = checkpoint.get('online_net', checkpoint)

# Group keys by module prefix
modules = {}
for k in sd.keys():
    prefix = '.'.join(k.split('.')[:-1])
    if prefix not in modules:
        modules[prefix] = []
    modules[prefix].append(k.split('.')[-1])

print("Detected Modules:")
for m in sorted(modules.keys()):
    print(f"  {m}: {modules[m]}")
