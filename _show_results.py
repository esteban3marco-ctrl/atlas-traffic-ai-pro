import json
import numpy as np

print("=" * 65)
print("  ATLAS Pro - Multi-Scenario Training Results")
print("=" * 65)

# History
h = json.load(open('checkpoints_full/training_history.json'))
print(f"\n  Total Episodes Trained: {h['total_episodes']}")
print(f"  Best Reward:           {h['best_reward']:.1f}")
print(f"  Mean Reward:           {h['mean_reward']:.1f}")
print(f"  Training Time:         {h['training_time_minutes']:.1f} min")

print("\n  Stage Results:")
for k, v in h.get('stage_results', {}).items():
    print(f"    {k}: Mean={v['mean']:.1f} +/- {v['std']:.1f}, Best={v['best']:.1f}")

# Rewards progression
rewards = h.get('rewards', [])
if rewards:
    chunks = [rewards[i:i+50] for i in range(0, len(rewards), 50)]
    print("\n  Reward Progression (50-ep windows):")
    for i, c in enumerate(chunks):
        bar_len = max(0, int((np.mean(c) + 200) / 5))
        bar = "#" * min(bar_len, 40)
        print(f"    Ep {i*50+1:>4}-{min((i+1)*50, len(rewards)):>4}: Avg={np.mean(c):>7.1f} | {bar}")

# Benchmark
print(f"\n{'='*65}")
print("  BENCHMARK: ATLAS AI vs Traditional Methods")
print(f"{'='*65}")
b = json.load(open('checkpoints_full/benchmark_results.json'))
for name, vals in sorted(b.items(), key=lambda x: -x[1]['mean']):
    marker = ">>>" if name == "ATLAS_AI" else "   "
    print(f"  {marker} {name:20s}: {vals['mean']:>8.1f} +/- {vals['std']:.1f}")

print(f"\n  {'Improvement over baselines:'}")
ai = b['ATLAS_AI']['mean']
for bl in ['fixed_time', 'actuated', 'max_pressure', 'webster']:
    bv = b[bl]['mean']
    imp = ((ai - bv) / abs(bv)) * 100 if bv != 0 else 0
    symbol = "+" if imp > 0 else ""
    print(f"    vs {bl:20s}: {symbol}{imp:.1f}%")

print(f"\n{'='*65}")
print("  Models saved in: checkpoints_full/")
print("  - atlas_best.pt     (best performing)")
print("  - atlas_final.pt    (after all training)")
print("  - atlas_ep*.pt      (periodic checkpoints)")
print(f"{'='*65}")
