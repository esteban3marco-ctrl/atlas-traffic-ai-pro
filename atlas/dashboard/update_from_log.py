import re
from pathlib import Path

log = Path(__file__).parents[1] / 'checkpoints_extended' / 'training.log'
if not log.exists():
    print('log not found', log)
    raise SystemExit(1)

lines = log.read_text(encoding='utf-8', errors='ignore').splitlines()
last = None
for l in reversed(lines):
    if 'Ep' in l and '/' in l:
        last = l
        break
if last is None:
    print('no Ep line found')
    raise SystemExit(1)

m_ep = re.search(r"Ep\s+(\d+)\s*/\s*(\d+)", last)
m_r = re.search(r"R=\s*([\-\d\.]+)", last)
m_avg = re.search(r"Avg50=\s*([\-\d\.]+)", last)
m_best = re.search(r"Best=\s*([\-\d\.]+)", last)
m_t = re.search(r"T=\s*(\d+)", last)

if not m_ep:
    print('could not parse ep line')
    raise SystemExit(1)

episode = int(m_ep.group(1))
total = int(m_ep.group(2))
reward = float(m_r.group(1)) if m_r else 0.0
mean_reward = float(m_avg.group(1)) if m_avg else 0.0
best = float(m_best.group(1)) if m_best else 0.0
throughput = int(m_t.group(1)) if m_t else 0

print('Parsed:', episode, '/', total, 'R=', reward, 'Avg=', mean_reward, 'Best=', best, 'T=', throughput)

# push to dashboard
try:
    from atlas.dashboard.app import update_training_state
    update_training_state({
        'is_training': True,
        'episode': episode,
        'total_episodes': total,
        'reward': reward,
        'mean_reward': mean_reward,
        'best_reward': best,
        'throughput': throughput,
    })
    print('Dashboard updated')
except Exception as e:
    print('Failed to update dashboard:', e)
    raise
