"""
ATLAS Pro - Extended Training (Resume + All Scenarios)
=======================================================
Resumes from best checkpoint and trains 1000 more episodes.
Includes retry logic for SUMO connection stability.
"""

import os, sys, time, json, logging, traceback
import numpy as np
import torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.config import AgentConfig, EnvironmentConfig, RewardConfig
from atlas.agents import DuelingDDQNAgent
from atlas.sumo_env import ATLASTrafficEnv
from atlas.baselines import get_baseline

TOTAL_EPISODES = 1200
DEVICE = "auto"
CHECKPOINT_DIR = "checkpoints_extended"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 6 stages, all 12 scenarios covered
STAGES = [
    (0,    150, ["simple"],                                       "Stage 1: Basic warmup"),
    (150,  300, ["simple_hora_punta", "simple_noche"],            "Stage 2: Simple variants"),
    (300,  500, ["hora_punta", "noche"],                          "Stage 3: Rush & Night"),
    (500,  700, ["simple_emergencias", "emergencias"],            "Stage 4: Emergencies"),
    (700,  900, ["complejo", "avenida", "cruce_t"],               "Stage 5: Complex geometry"),
    (900, 1200, ["evento", "doble", "hora_punta", "emergencias"],"Stage 6: Ultimate stress"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("ATLAS")


def make_env(scenario, ep=0):
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
    return ATLASTrafficEnv(env_config=cfg, reward_config=rw, episode_number=ep)


def make_agent():
    cfg = AgentConfig(
        algorithm="dueling_ddqn", state_dim=26, action_dim=4,
        hidden_dims=[256, 256, 128], lr=0.0002, gamma=0.99, tau=0.005,
        batch_size=64, buffer_size=200000, min_buffer_size=500,
        n_step=3, use_noisy_nets=True, noisy_sigma=0.5,
        per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100000,
        target_update_freq=4, max_grad_norm=10.0,
    )
    return DuelingDDQNAgent(state_dim=26, action_dim=4, config=cfg, device=DEVICE)


def run_episode(agent, scenario, ep, training=True, max_retries=3):
    """Run one episode with retry logic for SUMO stability."""
    for attempt in range(max_retries):
        env = None
        try:
            # Small delay between SUMO launches to avoid port conflicts
            time.sleep(0.3)
            env = make_env(scenario, ep)
            obs, info = env.reset()
            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = agent.select_action(obs, evaluate=not training)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if training:
                    agent.store_transition(obs, action, reward, next_obs, done)
                    agent.train_step()
                ep_reward += reward
                steps += 1
                obs = next_obs

            metrics = info.get("episode_metrics", {})
            return ep_reward, steps, metrics

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0)  # Wait before retry
            else:
                return None, 0, {}
        finally:
            if env is not None:
                try:
                    env.close()
                except:
                    pass


def benchmark_final(agent):
    log.info("=" * 65)
    log.info("FINAL BENCHMARK: ATLAS AI vs Baselines")
    log.info("=" * 65)

    results = {}

    # AI
    rs = []
    for _ in range(5):
        r, _, _ = run_episode(agent, "simple", 0, training=False)
        if r is not None:
            rs.append(r)
    if rs:
        results["ATLAS_AI"] = {"mean": float(np.mean(rs)), "std": float(np.std(rs))}
        log.info(f"  >>> ATLAS AI          : {np.mean(rs):>8.1f} +/- {np.std(rs):.1f}")

    for bname in ["fixed_time", "actuated", "max_pressure", "webster"]:
        bl = get_baseline(bname)
        bl_rs = []
        for _ in range(5):
            time.sleep(0.3)
            env = None
            try:
                env = make_env("simple")
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
            except:
                pass
            finally:
                if env:
                    try: env.close()
                    except: pass

        if bl_rs:
            results[bname] = {"mean": float(np.mean(bl_rs)), "std": float(np.std(bl_rs))}
            log.info(f"      {bname:20s}: {np.mean(bl_rs):>8.1f} +/- {np.std(bl_rs):.1f}")

    log.info("-" * 65)
    if "ATLAS_AI" in results:
        ai = results["ATLAS_AI"]["mean"]
        for bl in ["fixed_time", "actuated", "max_pressure", "webster"]:
            if bl in results:
                bv = results[bl]["mean"]
                imp = ((ai - bv) / abs(bv)) * 100 if bv != 0 else 0
                log.info(f"  Improvement vs {bl}: {imp:+.1f}%")
    log.info("=" * 65)
    return results


def main():
    agent = make_agent()

    # Resume from BEST available checkpoint
    checkpoints_to_try = [
        "checkpoints_full/atlas_best.pt",
        "checkpoints_full/atlas_final.pt",
        "checkpoints_multi/atlas_best.pt",
        "checkpoints/atlas_dueling_ddqn_best.pt",
    ]
    for cp in checkpoints_to_try:
        if os.path.exists(cp):
            agent.load(cp)
            log.info(f"Resumed from: {cp}")
            break

    log.info("=" * 65)
    log.info("ATLAS Pro - Extended Multi-Scenario Training")
    log.info(f"  Episodes: {TOTAL_EPISODES} | Stages: {len(STAGES)} | All 12 scenarios")
    log.info("=" * 65)

    t0 = time.time()
    best_reward = float("-inf")
    window = deque(maxlen=50)
    all_rewards = []
    stage_results = {}
    failed_episodes = 0

    try:
        for s_start, s_end, scenarios, desc in STAGES:
            log.info(f"\n{'='*65}")
            log.info(f"{desc} | Ep {s_start+1}-{s_end} | {scenarios}")
            log.info(f"{'='*65}")
            stage_rw = []

            for ep in range(s_start, s_end):
                scenario = scenarios[ep % len(scenarios)]
                ep_r, ep_s, ep_m = run_episode(agent, scenario, ep, training=True)

                if ep_r is None:
                    failed_episodes += 1
                    continue

                window.append(ep_r)
                all_rewards.append(ep_r)
                stage_rw.append(ep_r)

                if ep_r > best_reward:
                    best_reward = ep_r
                    agent.save(os.path.join(CHECKPOINT_DIR, "atlas_best.pt"))

                if (ep + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    avg = np.mean(window)
                    eps_min = len(all_rewards) / (elapsed / 60)
                    eta = (TOTAL_EPISODES - ep - 1) / max(eps_min, 0.1)
                    wt = ep_m.get("total_wait_time", 0)
                    th = ep_m.get("total_throughput", 0)
                    log.info(
                        f"Ep {ep+1:4d}/{TOTAL_EPISODES} | {scenario:22s} | "
                        f"R={ep_r:>7.1f} | Avg50={avg:>7.1f} | Best={best_reward:>7.1f} | "
                        f"W={wt:>5.0f} T={th:>3d} | ETA={eta:.0f}m"
                    )

                if (ep + 1) % 100 == 0:
                    agent.save(os.path.join(CHECKPOINT_DIR, f"atlas_ep{ep+1}.pt"))
                    agent.save(os.path.join(CHECKPOINT_DIR, "atlas_latest.pt"))
                    log.info(f"  >> Checkpoint saved: atlas_ep{ep+1}.pt")

            if stage_rw:
                sm, ss, sb = np.mean(stage_rw), np.std(stage_rw), max(stage_rw)
                stage_results[desc] = {"mean": float(sm), "std": float(ss), "best": float(sb)}
                log.info(f"\n>> {desc} DONE | Mean={sm:.1f} +/- {ss:.1f} | Best={sb:.1f}")
            else:
                log.warning(f">> {desc} - No episodes completed")

    except KeyboardInterrupt:
        log.info("\nInterrupted. Saving progress...")

    agent.save(os.path.join(CHECKPOINT_DIR, "atlas_final.pt"))

    elapsed = time.time() - t0
    log.info(f"\n{'='*65}")
    log.info(f"Training Complete | {len(all_rewards)} episodes | {elapsed/60:.1f} min | {failed_episodes} failed")
    log.info(f"Best Reward: {best_reward:.1f} | Final Avg50: {np.mean(list(window)):.1f}")
    log.info(f"{'='*65}")

    history = {
        "total_episodes": len(all_rewards),
        "best_reward": float(best_reward),
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0,
        "training_time_minutes": elapsed / 60,
        "failed_episodes": failed_episodes,
        "stage_results": stage_results,
        "rewards": [float(r) for r in all_rewards],
    }
    with open(os.path.join(CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Benchmark
    log.info("\nRunning final benchmark...")
    bench = benchmark_final(agent)
    with open(os.path.join(CHECKPOINT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(bench, f, indent=2)

    try:
        agent.export_onnx(os.path.join(CHECKPOINT_DIR, "atlas_production.onnx"))
        log.info("ONNX model exported")
    except Exception as e:
        log.warning(f"ONNX export skipped: {e}")

    log.info(f"\nAll done! Models in: {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
