import json
import numpy as np

print("=" * 65)
print("  ATLAS Pro - FINAL TRAINING RESULTS")
print("=" * 65)

h = json.load(open("checkpoints_extended/training_history.json"))
print(f"\n  Total Episodes: {h['total_episodes']}")
print(f"  Best Reward:    {h['best_reward']:.1f}")
print(f"  Mean Reward:    {h['mean_reward']:.1f}")
print(f"  Training Time:  {h['training_time_minutes']:.1f} min")
print(f"  Failed Episodes:{h.get('failed_episodes', 0)}")

print("\n  Stage Results:")
for k, v in h.get("stage_results", {}).items():
    print(f"    {k}: Mean={v['mean']:.1f}, Best={v['best']:.1f}")

rewards = h.get("rewards", [])
if rewards:
    print("\n  Learning Curve (50-ep windows):")
    chunks = [rewards[i:i+50] for i in range(0, len(rewards), 50)]
    for i, c in enumerate(chunks):
        avg = np.mean(c)
        bar_len = max(0, int((avg + 300) / 8))
        bar = "#" * min(bar_len, 45)
        print(f"    Ep {i*50+1:>4}-{min((i+1)*50, len(rewards)):>4}: Avg={avg:>7.1f} | {bar}")

b = json.load(open("checkpoints_extended/benchmark_results.json"))
print(f"\n{'='*65}")
print("  FINAL BENCHMARK: ATLAS AI vs Traditional Methods")
print(f"{'='*65}")
for name, vals in sorted(b.items(), key=lambda x: -x[1]["mean"]):
    m = ">>>" if name == "ATLAS_AI" else "   "
    print(f"  {m} {name:20s}: {vals['mean']:>8.1f} +/- {vals['std']:.1f}")

if "ATLAS_AI" in b:
    ai = b["ATLAS_AI"]["mean"]
    print(f"\n  Improvements:")
    for bl in ["fixed_time", "actuated", "max_pressure", "webster"]:
        if bl in b:
            bv = b[bl]["mean"]
            imp = ((ai - bv) / abs(bv)) * 100 if bv != 0 else 0
            print(f"    vs {bl:20s}: {imp:+.1f}%")

print(f"\n{'='*65}")
