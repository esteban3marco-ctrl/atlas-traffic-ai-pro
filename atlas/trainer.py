"""
ATLAS Pro — Professional Training Pipeline
=============================================
Training engine with TensorBoard logging, checkpointing, evaluation, and
curriculum learning orchestration.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

from atlas.config import TrainingConfig
from atlas.agents import DuelingDDQNAgent, PPOAgent
from atlas.rewards import MultiObjectiveReward
from atlas.baselines import get_baseline, BASELINES

logger = logging.getLogger("ATLAS.Trainer")

# Optional dashboard integration
try:
    from atlas.dashboard.app import update_training_state
    _DASHBOARD_AVAILABLE = True
except Exception:
    _DASHBOARD_AVAILABLE = False
# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


class TrainingMetrics:
    """Tracks and aggregates training metrics."""
    
    def __init__(self, window: int = 50):
        self.window = window
        self.episode_rewards = deque(maxlen=window)
        self.episode_lengths = deque(maxlen=window)
        self.losses = deque(maxlen=1000)
        self.eval_rewards = []
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.total_episodes = 0
        self.total_steps = 0
        
        # Traffic-specific
        self.episode_wait_times = deque(maxlen=window)
        self.episode_throughputs = deque(maxlen=window)
        self.episode_max_queues = deque(maxlen=window)
    
    def log_episode(self, reward, length, wait_time=0, throughput=0, max_queue=0):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_wait_times.append(wait_time)
        self.episode_throughputs.append(throughput)
        self.episode_max_queues.append(max_queue)
        self.total_episodes += 1
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.episodes_without_improvement = 0
            return True  # New best
        else:
            self.episodes_without_improvement += 1
            return False
    
    def log_loss(self, loss):
        if loss is not None:
            self.losses.append(loss)
    
    @property
    def mean_reward(self):
        return np.mean(self.episode_rewards) if self.episode_rewards else 0
    
    @property
    def mean_loss(self):
        return np.mean(self.losses) if self.losses else 0
    
    @property
    def mean_wait(self):
        return np.mean(self.episode_wait_times) if self.episode_wait_times else 0
    
    @property
    def mean_throughput(self):
        return np.mean(self.episode_throughputs) if self.episode_throughputs else 0
    
    def summary_dict(self) -> Dict:
        return {
            "episodes": self.total_episodes,
            "mean_reward": float(self.mean_reward),
            "best_reward": float(self.best_reward),
            "mean_loss": float(self.mean_loss),
            "mean_wait_time": float(self.mean_wait),
            "mean_throughput": float(self.mean_throughput),
        }


class Trainer:
    """
    Professional RL training pipeline for ATLAS.
    
    Features:
        - TensorBoard logging with detailed metrics
        - Automatic checkpointing (best + periodic)
        - Early stopping based on reward plateau
        - Evaluation against baseline policies
        - Curriculum learning orchestration
        - Training resume from checkpoint
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # Seed
        self._set_seed(self.config.seed)
        
        # Directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = None
        if self.config.use_tensorboard and _TB_AVAILABLE:
            run_name = f"atlas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(os.path.join(self.config.log_dir, run_name))
            logger.info(f"📊 TensorBoard: {os.path.join(self.config.log_dir, run_name)}")
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Create agent
        self.agent = self._create_agent()
        
        logger.info(f"🚀 Trainer initialized | device={self.device} | "
                     f"algorithm={self.config.agent.algorithm}")
    
    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_agent(self):
        algo = self.config.agent.algorithm
        if algo == "dueling_ddqn":
            return DuelingDDQNAgent(
                state_dim=self.config.agent.state_dim,
                action_dim=self.config.agent.action_dim,
                config=self.config.agent,
                device=self.device,
            )
        elif algo == "ppo":
            return PPOAgent(
                state_dim=self.config.agent.state_dim,
                action_dim=self.config.agent.action_dim,
                config=self.config.agent,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
    
    def _create_environment(self, episode: int = 0):
        """Create training environment (SUMO or Mock)."""
        cfg_file = self.config.environment.sumo_cfg_file
        
        if os.path.exists(cfg_file):
            from atlas.sumo_env import ATLASTrafficEnv
            return ATLASTrafficEnv(
                env_config=self.config.environment,
                reward_config=self.config.reward,
                episode_number=episode,
            )
        else:
            logger.warning(f"SUMO config not found: {cfg_file}. Using MockTrafficEnv.")
            from atlas.sumo_env import MockTrafficEnv
            return MockTrafficEnv(
                state_dim=self.config.agent.state_dim,
                action_dim=self.config.agent.action_dim,
            )
    
    def train(self, resume_from: str = None) -> Dict:
        """
        Run the full training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training summary dictionary
        """
        if resume_from:
            self.agent.load(resume_from)
            logger.info(f"📂 Resumed from {resume_from}")
        
        total_episodes = self.config.total_episodes
        
        logger.info("=" * 70)
        logger.info("🧠 ATLAS Pro — Training Start")
        logger.info(f"   Algorithm: {self.config.agent.algorithm.upper()}")
        logger.info(f"   Episodes: {total_episodes}")
        logger.info(f"   Device: {self.device}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            for episode in range(total_episodes):
                # Train one episode
                ep_reward, ep_length, ep_metrics = self._train_episode(episode)
                
                # Log metrics
                is_best = self.metrics.log_episode(
                    ep_reward, ep_length,
                    wait_time=ep_metrics.get("total_wait_time", 0),
                    throughput=ep_metrics.get("total_throughput", 0),
                    max_queue=ep_metrics.get("max_queue", 0),
                )
                
                # Update dashboard (best-effort)
                if _DASHBOARD_AVAILABLE:
                    try:
                        update_training_state({
                            "is_training": True,
                            "episode": episode + 1,
                            "total_episodes": total_episodes,
                            "reward": float(ep_reward),
                            "mean_reward": float(self.metrics.mean_reward),
                            "best_reward": float(self.metrics.best_reward),
                            "loss": float(self.metrics.mean_loss),
                            "wait_time": float(ep_metrics.get("total_wait_time", 0)),
                            "throughput": int(ep_metrics.get("total_throughput", 0)),
                        })
                    except Exception:
                        pass
                # TensorBoard logging
                self._log_tensorboard(episode, ep_reward, ep_length, ep_metrics)
                
                # Save best model
                if is_best:
                    self._save_checkpoint("best", episode)
                    logger.info(f"   🏆 New best reward: {ep_reward:.1f}")
                
                # Periodic save
                if (episode + 1) % self.config.save_interval == 0:
                    self._save_checkpoint("latest", episode)
                
                # Periodic evaluation
                if (episode + 1) % self.config.eval_interval == 0:
                    self._evaluate(episode)
                
                # Print progress
                if (episode + 1) % 5 == 0:
                    elapsed = time.time() - start_time
                    eps_per_sec = (episode + 1) / elapsed
                    eta = (total_episodes - episode - 1) / max(eps_per_sec, 1e-6)
                    
                    logger.info(
                        f"📍 Ep {episode+1}/{total_episodes} | "
                        f"R={ep_reward:.1f} | "
                        f"Avg={self.metrics.mean_reward:.1f} | "
                        f"Best={self.metrics.best_reward:.1f} | "
                        f"Loss={self.metrics.mean_loss:.4f} | "
                        f"ETA={eta/60:.1f}min"
                    )
                
                # Early stopping
                if (self.config.early_stopping and 
                    self.metrics.episodes_without_improvement >= self.config.patience):
                    logger.info(f"⏹️ Early stopping at episode {episode+1} "
                                f"(no improvement for {self.config.patience} episodes)")
                    break
        
        except KeyboardInterrupt:
            logger.info("⚠️ Training interrupted by user")
        
        finally:
            self._save_checkpoint("final", self.metrics.total_episodes)
            if self.writer:
                self.writer.close()
        
        elapsed = time.time() - start_time
        summary = self.metrics.summary_dict()
        summary["training_time_minutes"] = elapsed / 60
        summary["device"] = self.device
        summary["algorithm"] = self.config.agent.algorithm
        
        # Save training summary
        summary_path = os.path.join(self.config.checkpoint_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 70)
        logger.info("🏁 Training Complete")
        logger.info(f"   Episodes: {summary['episodes']}")
        logger.info(f"   Best Reward: {summary['best_reward']:.1f}")
        logger.info(f"   Mean Reward: {summary['mean_reward']:.1f}")
        logger.info(f"   Time: {elapsed/60:.1f} min")
        logger.info(f"   Summary: {summary_path}")
        logger.info("=" * 70)
        
        return summary
    
    def _train_episode(self, episode: int) -> Tuple[float, int, Dict]:
        """Train for one episode."""
        env = self._create_environment(episode)
        
        try:
            obs, info = env.reset()
            
            episode_reward = 0.0
            episode_length = 0
            episode_metrics = {}
            
            done = False
            
            while not done:
                # Select action
                action = self.agent.select_action(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(obs, action, reward, next_obs, done)
                
                # Train
                loss = self.agent.train_step()
                self.metrics.log_loss(loss)
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            # Get episode metrics
            episode_metrics = info.get("episode_metrics", {})
            
        finally:
            env.close()
        
        return episode_reward, episode_length, episode_metrics
    
    def _evaluate(self, episode: int):
        """Evaluate agent against baselines."""
        logger.info(f"📊 Evaluation at episode {episode + 1}...")
        
        eval_rewards = []
        for _ in range(self.config.eval_episodes):
            env = self._create_environment(episode)
            try:
                obs, _ = env.reset()
                total_reward = 0.0
                done = False
                
                while not done:
                    action = self.agent.select_action(obs, evaluate=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
            finally:
                env.close()
            
            eval_rewards.append(total_reward)
        
        mean_eval = np.mean(eval_rewards)
        self.metrics.eval_rewards.append(mean_eval)
        
        if self.writer:
            self.writer.add_scalar("eval/mean_reward", mean_eval, episode)
        
        logger.info(f"   Eval reward: {mean_eval:.1f} ± {np.std(eval_rewards):.1f}")
    
    def _log_tensorboard(self, episode, reward, length, ep_metrics):
        """Log metrics to TensorBoard."""
        if not self.writer:
            return
        
        self.writer.add_scalar("train/episode_reward", reward, episode)
        self.writer.add_scalar("train/episode_length", length, episode)
        self.writer.add_scalar("train/mean_reward", self.metrics.mean_reward, episode)
        self.writer.add_scalar("train/best_reward", self.metrics.best_reward, episode)
        
        if self.metrics.losses:
            self.writer.add_scalar("train/loss", self.metrics.mean_loss, episode)
        
        # Agent-specific metrics
        agent_metrics = self.agent.get_metrics()
        for key, val in agent_metrics.items():
            self.writer.add_scalar(f"agent/{key}", val, episode)
        
        # Traffic metrics
        for key in ["total_wait_time", "total_throughput", "max_queue", "phase_changes"]:
            if key in ep_metrics:
                self.writer.add_scalar(f"traffic/{key}", ep_metrics[key], episode)
        
        self.writer.flush()
    
    def _save_checkpoint(self, tag: str, episode: int):
        """Save agent checkpoint."""
        algo = self.config.agent.algorithm
        path = os.path.join(self.config.checkpoint_dir, f"atlas_{algo}_{tag}.pt")
        self.agent.save(path)
        
        # Save config alongside checkpoint
        config_path = os.path.join(self.config.checkpoint_dir, f"config_{tag}.yaml")
        self.config.save(config_path)
    
    def benchmark_baselines(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Benchmark all baselines and the trained agent.
        
        Returns:
            Dictionary mapping policy name to mean reward.
        """
        results = {}
        
        # Evaluate trained agent
        agent_rewards = []
        for ep in range(n_episodes):
            env = self._create_environment(ep)
            try:
                obs, _ = env.reset()
                total_reward = 0.0
                done = False
                while not done:
                    action = self.agent.select_action(obs, evaluate=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
            finally:
                env.close()
            agent_rewards.append(total_reward)
        
        results["atlas_ai"] = float(np.mean(agent_rewards))
        
        # Evaluate baselines
        for name in BASELINES:
            baseline = get_baseline(name)
            baseline_rewards = []
            
            for ep in range(n_episodes):
                env = self._create_environment(ep)
                try:
                    obs, _ = env.reset()
                    baseline.reset()
                    total_reward = 0.0
                    done = False
                    while not done:
                        action = baseline.select_action(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward
                        done = terminated or truncated
                finally:
                    env.close()
                baseline_rewards.append(total_reward)
            
            results[name] = float(np.mean(baseline_rewards))
        
        # Log results
        logger.info("=" * 50)
        logger.info("📊 Benchmark Results")
        logger.info("=" * 50)
        for name, reward in sorted(results.items(), key=lambda x: -x[1]):
            marker = "🏆" if name == "atlas_ai" else "  "
            logger.info(f"  {marker} {name:20s}: {reward:.1f}")
        logger.info("=" * 50)
        
        if self.writer:
            for name, reward in results.items():
                self.writer.add_scalar(f"benchmark/{name}", reward, 
                                        self.metrics.total_episodes)
        
        return results
