import torch
import os

root = r'D:\ATLAS\atlas-traffic-ai-pro'
path = os.path.join(root, 'checkpoints_extended', 'atlas_best.pt')
out = os.path.join(root, 'deduced_clean.txt')

checkpoint = torch.load(path, map_location='cpu')
sd = checkpoint.get('online_net', checkpoint)

with open(out, 'w', encoding='utf-8') as f:
    f.write("ARCHITECTURE KEYS\n")
    for k in sorted(sd.keys()):
        f.write(f"{k}\n")
