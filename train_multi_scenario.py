"""
ATLAS Pro - Multi-Scenario Training Pipeline (v2)
===================================================
Full 1000-episode training across all SUMO scenarios.
Fixed: Windows encoding issues, proper stage progression.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.config import AgentConfig, EnvironmentConfig, RewardConfig
from atlas.agents import DuelingDDQNAgent
from atlas.sumo_env import ATLASTrafficEnv
from atlas.baselines import get_baseline, BASELINES

# ============================================================
TOTAL_EPISODES = 1000
DEVICE = "auto"
CHECKPOINT_DIR = "checkpoints_full"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STAGES = [
    (0,   200, ["simple"],                                  "Stage 1: Basic intersection"),
    (200, 400, ["simple_hora_punta", "simple_noche"],       "Stage 2: Demand variation"),
    (400, 600, ["hora_punta", "noche"],                     "Stage 3: High-stress timing"),
    (600, 800, ["complejo", "avenida"],                     "Stage 4: Complex geometry"),
    (800, 1000, ["emergencias", "evento"],                  "Stage 5: Emergency & events"),
]

# Safe logging (no emojis for Windows file handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
# File handler with UTF-8 encoding
fh = logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log"), encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(fh)

log = logging.getLogger("ATLAS")

# ============================================================

def create_env(scenario, episode=0):
    cfg = EnvironmentConfig(
        sumo_cfg_file=f"simulations/{scenario}/simulation.sumocfg",
        gui=False, max_steps=1000, delta_time=10,
        min_green_time=10, max_green_time=60,
        yellow_time=3, all_red_time=2, step_length=1.0,
        warmup_steps=50, use_curriculum=False,
        traffic_light_ids=["center"],
        directions=["N", "S", "E", "W"],
        num_lanes_per_direction=1,
        edges={"N": "north_in", "S": "south_in", "E": "east_in", "W": "west_in"},
    )
    rw = RewardConfig(
        queue_length_weight=-0.25, wait_time_weight=-0.15,
        throughput_weight=0.5, speed_weight=0.1,
        fairness_weight=-0.3, emissions_weight=-0.05,
        phase_change_penalty=-0.8, emergency_weight=-5.0,
        clip_min=-10.0, clip_max=10.0, normalize=False,
    )
    return ATLASTrafficEnv(env_config=cfg, reward_config=rw, episode_number=episode)


def create_agent():
    cfg = AgentConfig(
        algorithm="dueling_ddqn", state_dim=26, action_dim=4,
        hidden_dims=[256, 256, 128], lr=0.0003, gamma=0.99, tau=0.005,
        batch_size=64, buffer_size=200000, min_buffer_size=500,
        n_step=3, use_noisy_nets=True, noisy_sigma=0.5,
        per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100000,
        target_update_freq=4, max_grad_norm=10.0,
    )
    return DuelingDDQNAgent(state_dim=26, action_dim=4, config=cfg, device=DEVICE)


def run_episode(agent, scenario, episode, training=True):
    env = create_env(scenario, episode)
    try:
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=not training)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if training:
                agent.store_transition(obs, action, reward, next_obs, done)
                agent.train_step()
            ep_reward += reward
            ep_steps += 1
            obs = next_obs
        metrics = info.get("episode_metrics", {})
        return ep_reward, ep_steps, metrics
    finally:
        env.close()


def benchmark(agent):
    log.info("=" * 65)
    log.info("FINAL BENCHMARK: ATLAS AI vs Traditional Methods")
    log.info("=" * 65)
    
    results = {}
    
    # AI agent
    rs = [run_episode(agent, "simple", 0, training=False)[0] for _ in range(5)]
    results["ATLAS_AI"] = {"mean": float(np.mean(rs)), "std": float(np.std(rs))}
    log.info(f"  >>> ATLAS AI          : {np.mean(rs):>8.1f} +/- {np.std(rs):.1f}")
    
    # Baselines
    for bname in ["fixed_time", "actuated", "max_pressure", "webster"]:
        bl = get_baseline(bname)
        bl_rs = []
        for _ in range(5):
            env = create_env("simple")
            try:
                obs, _ = env.reset()
                bl.reset()
                r = 0.0
                done = False
                while not done:
                    a = bl.select_action(obs)
                    obs, reward, term, trunc, _ = env.step(a)
                    r += reward
                    done = term or trunc
                bl_rs.append(r)
            finally:
                env.close()
        results[bname] = {"mean": float(np.mean(bl_rs)), "std": float(np.std(bl_rs))}
        log.info(f"      {bname:20s}: {np.mean(bl_rs):>8.1f} +/- {np.std(bl_rs):.1f}")
    
    log.info("-" * 65)
    ai = results["ATLAS_AI"]["mean"]
    for bl in ["fixed_time", "actuated", "max_pressure", "webster"]:
        bv = results[bl]["mean"]
        imp = ((ai - bv) / abs(bv)) * 100 if bv != 0 else 0
        log.info(f"  Improvement vs {bl}: {imp:+.1f}%")
    log.info("=" * 65)
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    agent = create_agent()
    
    # Resume from prior best if exists
    prior = "checkpoints/atlas_dueling_ddqn_best.pt"
    if os.path.exists(prior):
        agent.load(prior)
        log.info(f"Resumed from: {prior}")
    
    log.info("=" * 65)
    log.info("ATLAS Pro - Multi-Scenario Training")
    log.info(f"  Episodes: {TOTAL_EPISODES} | Stages: {len(STAGES)}")
    log.info("=" * 65)
    
    t0 = time.time()
    best_reward = float("-inf")
    reward_window = deque(maxlen=50)
    all_rewards = []
    stage_summary = {}
    
    try:
        for s_start, s_end, scenarios, desc in STAGES:
            log.info(f"\n{'='*65}")
            log.info(f"{desc} | Episodes {s_start+1}-{s_end} | Scenarios: {scenarios}")
            log.info(f"{'='*65}")
            
            stage_rw = []
            
            for ep in range(s_start, s_end):
                scenario = scenarios[ep % len(scenarios)]
                
                try:
                    ep_r, ep_s, ep_m = run_episode(agent, scenario, ep, training=True)
                except Exception as e:
                    log.warning(f"Episode {ep+1} error on {scenario}: {e}")
                    continue
                
                reward_window.append(ep_r)
                all_rewards.append(ep_r)
                stage_rw.append(ep_r)
                
                if ep_r > best_reward:
                    best_reward = ep_r
                    agent.save(os.path.join(CHECKPOINT_DIR, "atlas_best.pt"))
                
                if (ep + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    avg = np.mean(reward_window)
                    eps_min = (ep + 1) / (elapsed / 60)
                    eta = (TOTAL_EPISODES - ep - 1) / max(eps_min, 0.1)
                    wt = ep_m.get("total_wait_time", 0)
                    th = ep_m.get("total_throughput", 0)
                    log.info(
                        f"Ep {ep+1:4d}/{TOTAL_EPISODES} | {scenario:20s} | "
                        f"R={ep_r:>7.1f} | Avg50={avg:>7.1f} | Best={best_reward:>7.1f} | "
                        f"Wait={wt:>6.0f} | Thru={th:>4d} | ETA={eta:.0f}min"
                    )
                
                if (ep + 1) % 100 == 0:
                    agent.save(os.path.join(CHECKPOINT_DIR, f"atlas_ep{ep+1}.pt"))
                    agent.save(os.path.join(CHECKPOINT_DIR, "atlas_latest.pt"))
            
            if stage_rw:
                sm = np.mean(stage_rw)
                ss = np.std(stage_rw)
                sb = max(stage_rw)
                stage_summary[desc] = {"mean": float(sm), "std": float(ss), "best": float(sb)}
                log.info(f"\n>> {desc} COMPLETE | Mean={sm:.1f} +/- {ss:.1f} | Best={sb:.1f}")
    
    except KeyboardInterrupt:
        log.info("\nTraining interrupted. Saving...")
    
    # Save final
    agent.save(os.path.join(CHECKPOINT_DIR, "atlas_final.pt"))
    
    elapsed = time.time() - t0
    log.info(f"\n{'='*65}")
    log.info(f"Training Complete | {len(all_rewards)} episodes | {elapsed/60:.1f} min")
    log.info(f"Best Reward: {best_reward:.1f} | Final Avg50: {np.mean(list(reward_window)):.1f}")
    log.info(f"{'='*65}")
    
    # Save history
    history = {
        "total_episodes": len(all_rewards),
        "best_reward": float(best_reward),
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0,
        "training_time_minutes": elapsed / 60,
        "stage_results": stage_summary,
        "rewards": [float(r) for r in all_rewards],
    }
    with open(os.path.join(CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Benchmark
    log.info("\nRunning benchmark...")
    bench = benchmark(agent)
    with open(os.path.join(CHECKPOINT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(bench, f, indent=2)
    
    # ONNX export
    try:
        agent.export_onnx(os.path.join(CHECKPOINT_DIR, "atlas_production.onnx"))
    except Exception as e:
        log.warning(f"ONNX export skipped: {e}")
    
    log.info(f"\nAll models saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
