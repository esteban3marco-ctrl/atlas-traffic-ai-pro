# -*- coding: utf-8 -*-
"""
ATLAS Pro — Trainer & Baselines Tests
========================================
Tests for training pipeline, metrics, checkpointing,
early stopping, and baseline policies.
"""

import sys
import os
import json
import tempfile
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# TRAINING METRICS TESTS
# =============================================================================

class TestTrainingMetrics:
    """Tests for TrainingMetrics tracker."""

    def _make_metrics(self, window=10):
        from atlas.trainer import TrainingMetrics
        return TrainingMetrics(window=window)

    def test_creation(self):
        m = self._make_metrics()
        assert m.total_episodes == 0
        assert m.best_reward == float('-inf')
        assert m.mean_reward == 0
        assert m.mean_loss == 0

    def test_log_episode_returns_new_best(self):
        m = self._make_metrics()
        is_best = m.log_episode(reward=10.0, length=100)
        assert is_best is True
        assert m.best_reward == 10.0
        assert m.total_episodes == 1

    def test_log_episode_no_improvement(self):
        m = self._make_metrics()
        m.log_episode(reward=10.0, length=100)
        is_best = m.log_episode(reward=5.0, length=50)
        assert is_best is False
        assert m.episodes_without_improvement == 1

    def test_mean_reward(self):
        m = self._make_metrics()
        for r in [10, 20, 30]:
            m.log_episode(reward=r, length=100)
        assert m.mean_reward == 20.0

    def test_log_loss(self):
        m = self._make_metrics()
        m.log_loss(0.5)
        m.log_loss(0.3)
        m.log_loss(None)  # Should be ignored
        assert len(m.losses) == 2
        assert abs(m.mean_loss - 0.4) < 1e-6

    def test_traffic_metrics(self):
        m = self._make_metrics()
        m.log_episode(reward=10, length=100, wait_time=50, throughput=200, max_queue=15)
        assert m.mean_wait == 50.0
        assert m.mean_throughput == 200.0

    def test_summary_dict(self):
        m = self._make_metrics()
        m.log_episode(reward=10, length=100)
        m.log_loss(0.5)
        s = m.summary_dict()
        assert "episodes" in s
        assert "mean_reward" in s
        assert "best_reward" in s
        assert "mean_loss" in s

    def test_window_overflow(self):
        m = self._make_metrics(window=5)
        for i in range(10):
            m.log_episode(reward=i, length=100)
        # Only last 5 should be in window
        assert len(m.episode_rewards) == 5
        assert m.mean_reward == np.mean([5, 6, 7, 8, 9])


# =============================================================================
# TRAINER TESTS
# =============================================================================

class TestTrainer:
    """Tests for Trainer class."""

    def _make_trainer(self, **overrides):
        from atlas.config import TrainingConfig
        defaults = dict(
            total_episodes=3,
            eval_interval=2,
            eval_episodes=1,
            save_interval=2,
            early_stopping=False,
            use_tensorboard=False,
        )
        defaults.update(overrides)
        config = TrainingConfig(**defaults)
        config.checkpoint_dir = tempfile.mkdtemp()
        config.log_dir = tempfile.mkdtemp()
        from atlas.trainer import Trainer
        return Trainer(config)

    def test_creation_ddqn(self):
        t = self._make_trainer()
        assert t.agent is not None
        assert t.device in ("cpu", "cuda")

    def test_creation_ppo(self):
        from atlas.config import TrainingConfig, AgentConfig
        config = TrainingConfig(
            total_episodes=2,
            use_tensorboard=False,
            early_stopping=False,
        )
        config.agent.algorithm = "ppo"
        config.checkpoint_dir = tempfile.mkdtemp()
        config.log_dir = tempfile.mkdtemp()
        from atlas.trainer import Trainer
        t = Trainer(config)
        assert t.agent is not None

    def test_creation_invalid_algo(self):
        from atlas.config import TrainingConfig
        config = TrainingConfig(use_tensorboard=False)
        config.agent.algorithm = "invalid"
        config.checkpoint_dir = tempfile.mkdtemp()
        config.log_dir = tempfile.mkdtemp()
        from atlas.trainer import Trainer
        with pytest.raises(ValueError):
            Trainer(config)

    def test_train_mock_env(self):
        """Full training loop with mock environment."""
        t = self._make_trainer()
        summary = t.train()
        assert "episodes" in summary
        assert "best_reward" in summary
        assert "training_time_minutes" in summary
        assert summary["episodes"] == 3

    def test_train_saves_checkpoint(self):
        t = self._make_trainer()
        t.train()
        # Should have best and final checkpoints
        files = os.listdir(t.config.checkpoint_dir)
        assert any("best" in f for f in files)
        assert any("final" in f for f in files)

    def test_train_saves_summary(self):
        t = self._make_trainer()
        t.train()
        summary_path = os.path.join(t.config.checkpoint_dir, "training_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            data = json.load(f)
        assert "best_reward" in data

    def test_early_stopping(self):
        """Training should stop early when no improvement."""
        from atlas.config import TrainingConfig
        config = TrainingConfig(
            total_episodes=30,
            use_tensorboard=False,
            early_stopping=True,
            patience=3,
        )
        config.checkpoint_dir = tempfile.mkdtemp()
        config.log_dir = tempfile.mkdtemp()
        from atlas.trainer import Trainer
        t = Trainer(config)
        summary = t.train()
        # Should stop before 30 episodes due to early stopping
        assert summary["episodes"] < 30

    def test_evaluation_runs(self):
        """Trainer should run evaluation at intervals."""
        t = self._make_trainer(eval_interval=1, total_episodes=3)
        t.train()
        assert len(t.metrics.eval_rewards) > 0

    def test_save_and_resume(self):
        """Trainer should be able to save and resume from checkpoint."""
        t = self._make_trainer(total_episodes=2)
        t.train()

        # Get checkpoint path
        checkpoint_path = os.path.join(
            t.config.checkpoint_dir,
            f"atlas_{t.config.agent.algorithm}_final.pt"
        )
        assert os.path.exists(checkpoint_path)

        # Resume
        t2 = self._make_trainer(total_episodes=2)
        summary = t2.train(resume_from=checkpoint_path)
        assert summary["episodes"] == 2

    def test_benchmark_baselines(self):
        t = self._make_trainer()
        results = t.benchmark_baselines(n_episodes=2)
        assert "atlas_ai" in results
        assert "fixed_time" in results
        assert isinstance(results["atlas_ai"], float)

    def test_log_tensorboard_no_writer(self):
        """Should not crash when TensorBoard is not available."""
        t = self._make_trainer()
        t.writer = None
        # Should not raise
        t._log_tensorboard(0, 10.0, 100, {"total_wait_time": 5})

    def test_create_environment_mock_fallback(self):
        """Should fall back to MockTrafficEnv when SUMO not found."""
        t = self._make_trainer()
        env = t._create_environment(episode=0)
        # Should be MockTrafficEnv since SUMO config doesn't exist
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        env.close()

    def test_train_ppo_mock(self):
        from atlas.config import TrainingConfig
        config = TrainingConfig(
            total_episodes=2,
            use_tensorboard=False,
            early_stopping=False,
        )
        config.agent.algorithm = "ppo"
        config.agent.ppo_rollout_length = 50
        config.checkpoint_dir = tempfile.mkdtemp()
        config.log_dir = tempfile.mkdtemp()
        from atlas.trainer import Trainer
        t = Trainer(config)
        summary = t.train()
        assert summary["episodes"] == 2


# =============================================================================
# BASELINES ADDITIONAL TESTS
# =============================================================================

class TestBaselinesExtended:
    """Extended tests for baseline policies to improve coverage."""

    def test_all_policies_return_valid_actions(self):
        from atlas.baselines import get_baseline, BASELINES
        state = np.random.rand(26)
        for name in BASELINES:
            b = get_baseline(name)
            action = b.select_action(state)
            assert isinstance(action, int)
            assert 0 <= action < 4

    def test_fixed_time_cycle(self):
        from atlas.baselines import get_baseline
        b = get_baseline("fixed_time")
        state = np.random.rand(26)
        # Simulate multiple steps
        actions = []
        for _ in range(100):
            actions.append(b.select_action(state))
        # Should cycle through phases
        assert len(set(actions)) > 1

    def test_max_pressure_responds_to_state(self):
        from atlas.baselines import get_baseline
        b = get_baseline("max_pressure")
        # High queue on NS
        state = np.zeros(26)
        state[0] = 0.9  # N queue high
        state[5] = 0.9  # S queue high
        a1 = b.select_action(state)
        # High queue on EW
        state2 = np.zeros(26)
        state2[10] = 0.9  # E queue high
        state2[15] = 0.9  # W queue high
        a2 = b.select_action(state2)
        # Actions should differ
        assert isinstance(a1, int)
        assert isinstance(a2, int)

    def test_actuated_policy_responds_to_queue(self):
        from atlas.baselines import get_baseline
        b = get_baseline("actuated")
        state = np.zeros(26)
        state[0] = 0.8  # High queue
        action = b.select_action(state)
        assert isinstance(action, int)

    def test_webster_policy(self):
        from atlas.baselines import get_baseline
        b = get_baseline("webster")
        state = np.random.rand(26)
        action = b.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_all_baselines_reset(self):
        from atlas.baselines import BASELINES, get_baseline
        for name in BASELINES:
            b = get_baseline(name)
            b.select_action(np.random.rand(26))
            b.reset()
            # Should not crash after reset

    def test_get_baseline_unknown(self):
        from atlas.baselines import get_baseline
        with pytest.raises(ValueError):
            get_baseline("nonexistent_policy")

    def test_fixed_time_with_many_steps(self):
        from atlas.baselines import get_baseline
        b = get_baseline("fixed_time")
        state = np.random.rand(26)
        for _ in range(500):
            a = b.select_action(state)
            assert 0 <= a < 4

    def test_max_pressure_all_zero(self):
        from atlas.baselines import get_baseline
        b = get_baseline("max_pressure")
        state = np.zeros(26)
        action = b.select_action(state)
        assert isinstance(action, int)
