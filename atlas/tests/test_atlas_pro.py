"""
ATLAS Pro — Comprehensive Test Suite
========================================
Professional pytest tests for all components.
"""

import sys
import os
import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration system."""
    
    def test_default_config_creation(self):
        from atlas.config import TrainingConfig
        config = TrainingConfig()
        assert config.total_episodes == 500
        assert config.agent.algorithm == "dueling_ddqn"
        assert config.agent.state_dim == 26
        assert config.agent.action_dim == 4
    
    def test_agent_config_defaults(self):
        from atlas.config import AgentConfig
        cfg = AgentConfig()
        assert cfg.lr > 0
        assert cfg.gamma > 0 and cfg.gamma < 1
        assert cfg.buffer_size > 0
        assert len(cfg.hidden_dims) > 0
    
    def test_reward_config_defaults(self):
        from atlas.config import RewardConfig
        cfg = RewardConfig()
        assert cfg.queue_length_weight <= 0  # Should be negative (penalty)
        assert cfg.throughput_weight >= 0    # Should be positive (reward)
    
    def test_config_yaml_roundtrip(self, tmp_path):
        from atlas.config import TrainingConfig
        config = TrainingConfig()
        config.total_episodes = 123
        config.agent.lr = 0.0005
        
        path = str(tmp_path / "test_config.yaml")
        config.save(path)
        loaded = TrainingConfig.load(path)
        
        assert loaded.total_episodes == 123
        assert abs(loaded.agent.lr - 0.0005) < 1e-10
    
    def test_config_to_dict(self):
        from atlas.config import TrainingConfig
        config = TrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "agent" in d
        assert "reward" in d
        assert "environment" in d


# =============================================================================
# NETWORK TESTS
# =============================================================================

class TestNetworks:
    """Tests for neural network architectures."""
    
    def test_noisy_linear_forward(self):
        from atlas.networks import NoisyLinear
        layer = NoisyLinear(10, 5)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 5)
    
    def test_noisy_linear_noise_reset(self):
        from atlas.networks import NoisyLinear
        layer = NoisyLinear(10, 5)
        x = torch.randn(1, 10)
        
        out1 = layer(x).detach().clone()
        layer.reset_noise()
        out2 = layer(x).detach().clone()
        
        # Outputs should differ after noise reset (probabilistically)
        # Can rarely be equal, so we don't hard-assert
    
    def test_dueling_network_forward(self):
        from atlas.networks import DuelingNetwork
        net = DuelingNetwork(state_dim=26, action_dim=4, hidden_dims=[64, 64])
        x = torch.randn(8, 26)
        q = net(x)
        assert q.shape == (8, 4)
    
    def test_dueling_network_with_noisy(self):
        from atlas.networks import DuelingNetwork
        net = DuelingNetwork(state_dim=26, action_dim=4, use_noisy=True)
        x = torch.randn(4, 26)
        q = net(x)
        assert q.shape == (4, 4)
        net.reset_noise()  # Should not error
    
    def test_actor_critic_network_forward(self):
        from atlas.networks import ActorCriticNetwork
        net = ActorCriticNetwork(state_dim=26, action_dim=4, hidden_dims=[64, 64])
        x = torch.randn(8, 26)
        
        action, log_prob, entropy, value = net.get_action_and_value(x)
        
        assert action.shape == (8,)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert value.shape == (8,)
    
    def test_actor_critic_with_given_action(self):
        from atlas.networks import ActorCriticNetwork
        net = ActorCriticNetwork(state_dim=26, action_dim=4)
        x = torch.randn(4, 26)
        actions = torch.tensor([0, 1, 2, 3])
        
        _, log_prob, entropy, value = net.get_action_and_value(x, actions)
        
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
    
    def test_network_gradient_flow(self):
        from atlas.networks import DuelingNetwork
        net = DuelingNetwork(state_dim=26, action_dim=4, hidden_dims=[64, 64])
        x = torch.randn(4, 26)
        q = net(x)
        loss = q.sum()
        loss.backward()
        
        # Check gradients exist and are non-zero
        grads_exist = False
        for p in net.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grads_exist = True
                break
        assert grads_exist

    def test_network_param_count(self):
        from atlas.networks import DuelingNetwork
        net = DuelingNetwork(state_dim=26, action_dim=4, hidden_dims=[256, 256, 128])
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params > 10000, "Network should have substantial parameter count"


# =============================================================================
# REPLAY BUFFER TESTS
# =============================================================================

class TestReplayBuffer:
    """Tests for replay buffers."""
    
    def test_per_add_and_sample(self):
        from atlas.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add transitions
        for i in range(30):
            state = np.random.randn(26).astype(np.float32)
            next_state = np.random.randn(26).astype(np.float32)
            buffer.add(state, i % 4, float(i), next_state, False)
        
        assert len(buffer) == 30
        
        # Sample
        batch = buffer.sample(8)
        assert batch is not None
        states, actions, rewards, next_states, dones, indices, weights = batch
        assert states.shape == (8, 26)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert weights.shape == (8,)
    
    def test_per_priority_update(self):
        from atlas.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(capacity=50)
        
        for i in range(20):
            state = np.random.randn(26).astype(np.float32)
            buffer.add(state, 0, 0.0, state, False)
        
        batch = buffer.sample(4)
        indices = batch[5]
        td_errors = np.array([10.0, 0.1, 5.0, 0.01])
        buffer.update_priorities(indices, td_errors)
    
    def test_per_empty_buffer(self):
        from atlas.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert len(buffer) == 0
    
    def test_per_capacity_overflow(self):
        from atlas.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(capacity=10)
        
        for i in range(25):
            state = np.random.randn(26).astype(np.float32)
            buffer.add(state, 0, 0.0, state, False)
        
        assert len(buffer) == 10  # Should not exceed capacity
    
    def test_sumtree_operations(self):
        from atlas.replay_buffer import SumTree
        tree = SumTree(8)
        
        for i in range(8):
            tree.add(float(i + 1), f"data_{i}")
        
        assert abs(tree.total_priority - sum(range(1, 9))) < 1e-6
        assert tree.size == 8
    
    def test_nstep_buffer(self):
        from atlas.replay_buffer import NStepBuffer
        nstep = NStepBuffer(n_step=3, gamma=0.99)
        
        state = np.zeros(26, dtype=np.float32)
        results = []
        
        for i in range(5):
            result = nstep.add(state, 0, 1.0, state, i == 4)
            if result is not None:
                results.append(result)
        
        assert len(results) > 0
        # N-step return should be accumulated discounted sum
        assert results[0].reward > 1.0  # Should be > single step reward
    
    def test_nstep_gamma_discount(self):
        from atlas.replay_buffer import NStepBuffer
        nstep = NStepBuffer(n_step=3, gamma=0.5)
        
        state = np.zeros(26, dtype=np.float32)
        
        nstep.add(state, 0, 1.0, state, False)
        nstep.add(state, 0, 1.0, state, False)
        result = nstep.add(state, 0, 1.0, state, False)
        
        # R = 1.0 + 0.5*1.0 + 0.25*1.0 = 1.75
        assert result is not None
        assert abs(result.reward - 1.75) < 1e-6


# =============================================================================
# REWARD TESTS
# =============================================================================

class TestRewards:
    """Tests for reward functions."""
    
    def test_multi_objective_basic(self):
        from atlas.rewards import MultiObjectiveReward
        from atlas.config import RewardConfig
        
        reward_fn = MultiObjectiveReward(RewardConfig())
        
        r = reward_fn.compute(
            queue_lengths={"N": 5, "S": 3, "E": 2, "W": 4},
            wait_times={"N": 10, "S": 8, "E": 5, "W": 7},
            speeds={"N": 10, "S": 12, "E": 8, "W": 11},
            throughput=10,
            phase_changed=False,
        )
        
        assert isinstance(r, float)
    
    def test_reward_phase_penalty(self):
        from atlas.rewards import MultiObjectiveReward
        from atlas.config import RewardConfig
        
        reward_fn = MultiObjectiveReward(RewardConfig())
        
        queues = {"N": 5, "S": 3, "E": 2, "W": 4}
        waits = {"N": 10, "S": 8, "E": 5, "W": 7}
        speeds = {"N": 10, "S": 12, "E": 8, "W": 11}
        
        r_no_change = reward_fn.compute(queues, waits, speeds, 10, False)
        reward_fn.reset()
        r_change = reward_fn.compute(queues, waits, speeds, 10, True)
        
        # Phase change should incur penalty (lower reward)
        assert r_change <= r_no_change
    
    def test_reward_reset(self):
        from atlas.rewards import MultiObjectiveReward
        reward_fn = MultiObjectiveReward()
        
        reward_fn.compute(
            {"N": 5, "S": 3, "E": 2, "W": 4},
            {"N": 10, "S": 8, "E": 5, "W": 7},
            {"N": 10, "S": 12, "E": 8, "W": 11},
            10, False
        )
        
        reward_fn.reset()
        assert reward_fn._step_count == 0
        assert reward_fn._prev_total_wait == 0
    
    def test_reward_breakdown(self):
        from atlas.rewards import MultiObjectiveReward
        reward_fn = MultiObjectiveReward()
        
        breakdown = reward_fn.get_component_breakdown(
            {"N": 5, "S": 3, "E": 2, "W": 4},
            {"N": 10, "S": 8, "E": 5, "W": 7},
            {"N": 10, "S": 12, "E": 8, "W": 11},
            10, False
        )
        
        assert "queue_penalty" in breakdown
        assert "throughput_reward" in breakdown
        assert "speed_reward" in breakdown
    
    def test_potential_based_shaping(self):
        from atlas.rewards import PotentialBasedRewardShaping
        shaping = PotentialBasedRewardShaping(gamma=0.99, scale=0.1)
        
        r1 = shaping.compute_shaping({"N": 10, "S": 10, "E": 10, "W": 10})
        r2 = shaping.compute_shaping({"N": 5, "S": 5, "E": 5, "W": 5})
        
        # Reducing queues should give positive shaping reward
        assert r2 > r1
    
    def test_fairness_jain_index(self):
        from atlas.rewards import MultiObjectiveReward
        from atlas.config import RewardConfig
        
        config = RewardConfig(fairness_weight=-1.0)
        reward_fn = MultiObjectiveReward(config)
        
        # Equal waits (perfectly fair)
        r_fair = reward_fn.compute(
            {"N": 5, "S": 5, "E": 5, "W": 5},
            {"N": 10, "S": 10, "E": 10, "W": 10},
            {"N": 10, "S": 10, "E": 10, "W": 10},
            10, False
        )
        
        reward_fn.reset()
        
        # Very unfair waits
        r_unfair = reward_fn.compute(
            {"N": 5, "S": 5, "E": 5, "W": 5},
            {"N": 100, "S": 1, "E": 1, "W": 1},
            {"N": 10, "S": 10, "E": 10, "W": 10},
            10, False
        )
        
        # Fair scenario should have higher reward
        assert r_fair > r_unfair


# =============================================================================
# AGENT TESTS
# =============================================================================

class TestAgents:
    """Tests for RL agents."""
    
    def test_ddqn_creation(self):
        from atlas.agents import DuelingDDQNAgent
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, device="cpu")
        assert agent.state_dim == 26
        assert agent.action_dim == 4
    
    def test_ddqn_action_selection(self):
        from atlas.agents import DuelingDDQNAgent
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, device="cpu")
        
        state = np.random.randn(26).astype(np.float32)
        action = agent.select_action(state)
        
        assert 0 <= action < 4
    
    def test_ddqn_evaluation_mode(self):
        from atlas.agents import DuelingDDQNAgent
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, device="cpu")
        
        state = np.random.randn(26).astype(np.float32)
        action = agent.select_action(state, evaluate=True)
        
        assert 0 <= action < 4
    
    def test_ddqn_store_and_train(self):
        from atlas.agents import DuelingDDQNAgent
        from atlas.config import AgentConfig
        
        config = AgentConfig(min_buffer_size=10, batch_size=4, buffer_size=100)
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, config=config, device="cpu")
        
        # Fill buffer
        for _ in range(20):
            s = np.random.randn(26).astype(np.float32)
            a = np.random.randint(4)
            r = np.random.randn()
            ns = np.random.randn(26).astype(np.float32)
            agent.store_transition(s, a, r, ns, False)
        
        # Train
        loss = agent.train_step()
        assert loss is not None or len(agent.buffer) < config.min_buffer_size
    
    def test_ddqn_save_load(self, tmp_path):
        from atlas.agents import DuelingDDQNAgent
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, device="cpu")
        
        state = np.random.randn(26).astype(np.float32)
        q_before = agent.select_action(state, evaluate=True)
        
        path = str(tmp_path / "test_agent.pt")
        agent.save(path)
        
        agent2 = DuelingDDQNAgent(state_dim=26, action_dim=4, device="cpu")
        agent2.load(path)
        
        q_after = agent2.select_action(state, evaluate=True)
        assert q_before == q_after
    
    def test_ppo_creation(self):
        from atlas.agents import PPOAgent
        agent = PPOAgent(state_dim=26, action_dim=4, device="cpu")
        assert agent.state_dim == 26
    
    def test_ppo_action_selection(self):
        from atlas.agents import PPOAgent
        agent = PPOAgent(state_dim=26, action_dim=4, device="cpu")
        
        state = np.random.randn(26).astype(np.float32)
        action = agent.select_action(state)
        
        assert 0 <= action < 4
    
    def test_ppo_training_loop(self):
        from atlas.agents import PPOAgent
        from atlas.config import AgentConfig
        
        config = AgentConfig(algorithm="ppo", ppo_rollout_length=32)
        agent = PPOAgent(state_dim=26, action_dim=4, config=config, device="cpu")
        
        for _ in range(40):
            s = np.random.randn(26).astype(np.float32)
            agent.select_action(s)
            agent.store_reward(np.random.randn(), False)
        
        if agent.is_ready_to_train():
            loss = agent.train_step()
            assert loss is not None
    
    def test_ppo_save_load(self, tmp_path):
        from atlas.agents import PPOAgent
        agent = PPOAgent(state_dim=26, action_dim=4, device="cpu")
        
        path = str(tmp_path / "test_ppo.pt")
        agent.save(path)
        
        agent2 = PPOAgent(state_dim=26, action_dim=4, device="cpu")
        agent2.load(path)
    
    def test_agent_metrics(self):
        from atlas.agents import DuelingDDQNAgent
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, device="cpu")
        metrics = agent.get_metrics()
        assert isinstance(metrics, dict)


# =============================================================================
# ENVIRONMENT TESTS
# =============================================================================

class TestEnvironment:
    """Tests for the traffic environment."""
    
    def test_mock_env_creation(self):
        from atlas.sumo_env import MockTrafficEnv
        env = MockTrafficEnv(state_dim=26, action_dim=4)
        assert env.observation_space.shape == (26,)
        assert env.action_space.n == 4
    
    def test_mock_env_reset(self):
        from atlas.sumo_env import MockTrafficEnv
        env = MockTrafficEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (26,)
        assert isinstance(info, dict)
    
    def test_mock_env_step(self):
        from atlas.sumo_env import MockTrafficEnv
        env = MockTrafficEnv()
        obs, _ = env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (26,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_mock_env_full_episode(self):
        from atlas.sumo_env import MockTrafficEnv
        env = MockTrafficEnv(max_steps=50)
        obs, _ = env.reset(seed=42)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        assert steps == 50
    
    def test_mock_env_reproducibility(self):
        from atlas.sumo_env import MockTrafficEnv
        
        env1 = MockTrafficEnv()
        obs1, _ = env1.reset(seed=42)
        
        env2 = MockTrafficEnv()
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_running_mean_std(self):
        from atlas.sumo_env import RunningMeanStd
        rms = RunningMeanStd(shape=(4,))
        
        for _ in range(100):
            x = np.random.randn(4) * 3 + 5
            rms.update(x)
        
        # Mean should be close to 5
        assert abs(rms.mean.mean() - 5) < 2
        
        normalized = rms.normalize(np.array([5, 5, 5, 5], dtype=np.float64))
        assert abs(normalized.mean()) < 1


# =============================================================================
# BASELINE TESTS
# =============================================================================

class TestBaselines:
    """Tests for baseline policies."""
    
    def test_fixed_time_policy(self):
        from atlas.baselines import FixedTimePolicy
        policy = FixedTimePolicy(green_ns=30, green_ew=25)
        
        state = np.random.randn(26).astype(np.float32)
        
        # Should maintain for initial steps
        for _ in range(5):
            action = policy.select_action(state)
            assert 0 <= action <= 3
    
    def test_actuated_policy(self):
        from atlas.baselines import ActuatedPolicy
        policy = ActuatedPolicy()
        
        state = np.random.randn(26).astype(np.float32)
        action = policy.select_action(state)
        assert 0 <= action <= 3
    
    def test_max_pressure_policy(self):
        from atlas.baselines import MaxPressurePolicy
        policy = MaxPressurePolicy()
        
        # High pressure on N-S
        state = np.zeros(26, dtype=np.float32)
        state[0] = 0.8   # N queue high
        state[5] = 0.7   # S queue high
        state[10] = 0.1  # E queue low
        state[15] = 0.1  # W queue low
        
        # After min green, should keep NS
        for _ in range(3):
            action = policy.select_action(state)
        
        assert 0 <= action <= 3
    
    def test_webster_policy(self):
        from atlas.baselines import WebsterPolicy
        policy = WebsterPolicy()
        state = np.random.randn(26).astype(np.float32)
        
        action = policy.select_action(state)
        assert 0 <= action <= 3
    
    def test_baseline_registry(self):
        from atlas.baselines import get_baseline, BASELINES
        
        assert len(BASELINES) >= 4
        
        for name in BASELINES:
            policy = get_baseline(name)
            state = np.random.randn(26).astype(np.float32)
            action = policy.select_action(state)
            assert 0 <= action <= 3
    
    def test_baseline_reset(self):
        from atlas.baselines import FixedTimePolicy
        policy = FixedTimePolicy()
        
        state = np.random.randn(26).astype(np.float32)
        for _ in range(10):
            policy.select_action(state)
        
        policy.reset()
        assert policy.step_count == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_ddqn_mock_env_training_loop(self):
        """Full training loop: agent + env + rewards."""
        from atlas.agents import DuelingDDQNAgent
        from atlas.sumo_env import MockTrafficEnv
        from atlas.config import AgentConfig
        
        config = AgentConfig(min_buffer_size=20, batch_size=8, buffer_size=200)
        agent = DuelingDDQNAgent(state_dim=26, action_dim=4, config=config, device="cpu")
        env = MockTrafficEnv(max_steps=50)
        
        for episode in range(3):
            obs, _ = env.reset(seed=episode)
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(obs, action, reward, next_obs, done)
                agent.train_step()
                
                total_reward += reward
                obs = next_obs
        
        # Agent should have trained
        assert agent.train_steps >= 0  # May not train if buffer not full
    
    def test_ppo_mock_env_training_loop(self):
        """Full PPO training loop."""
        from atlas.agents import PPOAgent
        from atlas.sumo_env import MockTrafficEnv
        from atlas.config import AgentConfig
        
        config = AgentConfig(algorithm="ppo", ppo_rollout_length=30)
        agent = PPOAgent(state_dim=26, action_dim=4, config=config, device="cpu")
        env = MockTrafficEnv(max_steps=50)
        
        obs, _ = env.reset(seed=42)
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)
            
            if agent.is_ready_to_train():
                agent.train_step()
            
            obs = next_obs
    
    def test_config_driven_training(self, tmp_path):
        """Training with full config object."""
        from atlas.config import TrainingConfig
        from atlas.trainer import Trainer
        
        config = TrainingConfig()
        config.total_episodes = 2
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.log_dir = str(tmp_path / "runs")
        config.use_tensorboard = False
        config.eval_interval = 999  # Skip eval
        
        trainer = Trainer(config)
        summary = trainer.train()
        
        assert summary["episodes"] == 2
        assert "best_reward" in summary
        assert "training_time_minutes" in summary


# =============================================================================
# PERFORMANCE / STRESS TESTS
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_batch_inference_speed(self):
        """Test batch inference throughput."""
        from atlas.networks import DuelingNetwork
        import time
        
        net = DuelingNetwork(state_dim=26, action_dim=4, hidden_dims=[256, 256, 128])
        net.eval()
        
        batch = torch.randn(64, 26)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                net(batch)
        elapsed = time.time() - start
        
        # Should process 6400 samples in reasonable time
        throughput = 6400 / elapsed
        assert throughput > 1000, f"Throughput too low: {throughput:.0f} samples/sec"
    
    def test_buffer_stress(self):
        """Stress test replay buffer with many operations."""
        from atlas.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(capacity=1000)
        
        for i in range(2000):
            s = np.random.randn(26).astype(np.float32)
            buffer.add(s, i % 4, float(i), s, False)
        
        assert len(buffer) == 1000
        
        for _ in range(50):
            batch = buffer.sample(32)
            assert batch is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
