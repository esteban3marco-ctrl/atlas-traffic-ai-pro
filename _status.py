import json, os
import numpy as np

print("=" * 65)
print("  ATLAS Pro — Current Status Summary")
print("=" * 65)

# Check all checkpoint dirs
for cdir in ["checkpoints", "checkpoints_multi", "checkpoints_full", "checkpoints_extended"]:
    if not os.path.exists(cdir):
        continue
    print(f"\n--- {cdir} ---")
    
    # History
    hf = os.path.join(cdir, "training_history.json")
    if os.path.exists(hf):
        h = json.load(open(hf))
        print(f"  Episodes: {h['total_episodes']}")
        print(f"  Best Reward: {h['best_reward']:.1f}")
        print(f"  Mean Reward: {h['mean_reward']:.1f}")
        print(f"  Time: {h['training_time_minutes']:.1f} min")
        for k, v in h.get("stage_results", {}).items():
            print(f"    {k}: mean={v['mean']:.1f}, best={v['best']:.1f}")
    
    # Benchmark
    bf = os.path.join(cdir, "benchmark_results.json")
    if os.path.exists(bf):
        b = json.load(open(bf))
        print("  Benchmark:")
        for name, vals in sorted(b.items(), key=lambda x: -x[1]["mean"]):
            m = ">>>" if name == "ATLAS_AI" else "   "
            print(f"    {m} {name:20s}: {vals['mean']:>8.1f}")
    
    # Models
    models = [f for f in os.listdir(cdir) if f.endswith(".pt")]
    if models:
        print(f"  Models: {', '.join(sorted(models))}")

print(f"\n{'='*65}")
