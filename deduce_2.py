import torch
import torch.nn as nn

path = r'D:\ATLAS\atlas-traffic-ai-pro\checkpoints_extended\atlas_best.pt'
checkpoint = torch.load(path, map_location='cpu')
sd = checkpoint.get('online_net', checkpoint)

modules = {}
for k in sd.keys():
    if k == 'support': continue
    prefix = '.'.join(k.split('.')[:-1])
    if prefix not in modules:
        modules[prefix] = []
    modules[prefix].append(k.split('.')[-1])

with open(r'D:\ATLAS\atlas-traffic-ai-pro\deduced_clean.txt', 'w', encoding='utf-8') as f:
    f.write("DEDUCED ARCHITECTURE:\n")
    for m in sorted(modules.keys()):
        keys = sorted(modules[m])
        f.write(f"  {m}: {keys}\n")
