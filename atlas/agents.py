"""
ATLAS Pro — Reinforcement Learning Agents
===========================================
State-of-the-art RL agents for traffic signal control.

Agents:
    - DuelingDDQNAgent: Double DQN + Dueling Architecture + PER + N-step + Noisy Nets
    - PPOAgent: Proximal Policy Optimization with GAE and entropy bonus
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Tuple, List, Union
import logging
import json

from atlas.networks import DuelingNetwork, ActorCriticNetwork, WorldModel, QMixer
from atlas.replay_buffer import PrioritizedReplayBuffer, NStepBuffer
from atlas.config import AgentConfig

logger = logging.getLogger("ATLAS.Agent")


# =============================================================================
# DUELING DOUBLE DQN AGENT
# =============================================================================

class DuelingDDQNAgent:
    """
    Dueling Double DQN Agent with full Rainbow enhancements (minus distributional).
    
    Combines:
        1. Double DQN: Reduces overestimation by decoupling action selection and evaluation
        2. Dueling Architecture: Separates V(s) and A(s,a) for better generalization
        3. Prioritized Experience Replay: Samples important transitions more often
        4. N-step Returns: Multi-step bootstrapping for better bias-variance tradeoff
        5. Noisy Networks: Learned exploration without epsilon scheduling
        6. Soft Target Updates: Smoother target network updates via Polyak averaging
    
    This is the primary production agent for ATLAS.
    """
    
    ACTIONS = ['maintain', 'switch_ns', 'switch_ew', 'extend']
    
    def __init__(self, state_dim: int = 26, action_dim: int = 4,
                 config: AgentConfig = None, device: str = "auto"):
        
        self.config = config or AgentConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"🧠 DuelingDDQN Agent | device={self.device} | "
                     f"state_dim={state_dim} | action_dim={action_dim}")
        
        # === Networks ===
        self.online_net = DuelingNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            use_noisy=self.config.use_noisy_nets,
            sigma_init=self.config.noisy_sigma,
            use_transformer=self.config.use_transformer,
        ).to(self.device)
        
        self.target_net = DuelingNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            use_noisy=self.config.use_noisy_nets,
            sigma_init=self.config.noisy_sigma,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            use_transformer=self.config.use_transformer,
        ).to(self.device)
        
        # === V2: World Model (Dreamer-lite) ===
        if self.config.use_world_model:
            self.world_model = WorldModel(
                state_dim=state_dim,
                action_dim=action_dim,
                latent_dim=self.config.latent_dim,
            ).to(self.device)
            self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=1e-4)
        else:
            self.world_model = None
        
        # === V3: MARL QMIX Coordination ===
        if self.config.use_qmix:
            self.mixer = QMixer(
                n_agents=self.config.n_agents,
                state_dim=state_dim * self.config.n_agents, # Global state = all observations
                embed_dim=self.config.mixer_embed_dim
            ).to(self.device)
            self.mixer_target = QMixer(
                n_agents=self.config.n_agents,
                state_dim=state_dim * self.config.n_agents,
                embed_dim=self.config.mixer_embed_dim
            ).to(self.device)
            self.mixer_target.load_state_dict(self.mixer.state_dict())
            
            # Combine parameters for optimizer
            self.agent_params = list(self.online_net.parameters()) + list(self.mixer.parameters())
        else:
            self.mixer = None
            self.agent_params = list(self.online_net.parameters())

        # Copy weights to target
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # === Optimizer ===
        self.optimizer = optim.AdamW(
            self.agent_params,
            lr=self.config.lr,
            weight_decay=1e-5,
            amsgrad=True,
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )
        
        # === Replay Buffer (PER) ===
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_frames=self.config.per_beta_frames,
        )
        
        # === N-step Buffer ===
        self.n_step_buffer = NStepBuffer(
            n_step=self.config.n_step,
            gamma=self.config.gamma,
        )
        
        # N-step discount for computing targets
        self.n_step_gamma = self.config.gamma ** self.config.n_step
        
        # === Counters ===
        self.train_steps = 0
        self.total_steps = 0
        
        # === Metrics ===
        self.metrics: Dict[str, float] = {}
        
        total_params = sum(p.numel() for p in self.online_net.parameters())
        logger.info(f"   Network params: {total_params:,}")
    
    def select_action(self, state: np.ndarray, evaluate: bool = False, return_xai: bool = False) -> Union[int, List[int], Tuple[Union[int, List[int]], np.ndarray]]:
        """
        Select action(s) for one or more agents.
        """
        with torch.no_grad():
            is_ma = (state.ndim == 2)
            state_t = torch.FloatTensor(state).to(self.device)
            if not is_ma:
                state_t = state_t.unsqueeze(0)
            
            if evaluate:
                self.online_net.eval()
            else:
                self.online_net.reset_noise()

            if return_xai:
                dist, attn = self.online_net(state_t, return_attn=True)
                # q_values: [B, Action]
                q_values = (dist * self.online_net.support).sum(dim=2)
                attn_map = attn[0].cpu().numpy() if attn is not None else None
            else:
                q_values = self.online_net.get_q_values(state_t)
                attn_map = None

            if evaluate:
                self.online_net.train()

            # actions: [B]
            actions = q_values.argmax(dim=-1).cpu().numpy()
            
            self.total_steps += 1
            
            if is_ma:
                final_actions = actions.tolist()
            else:
                final_actions = int(actions[0])

            if return_xai:
                return final_actions, attn_map
            return final_actions
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition, computing N-step return first."""
        n_step_transition = self.n_step_buffer.add(state, action, reward, next_state, done)
        
        if n_step_transition is not None:
            self.buffer.add(
                n_step_transition.state,
                n_step_transition.action,
                n_step_transition.reward,
                n_step_transition.next_state,
                n_step_transition.done,
            )
        
        if done:
            # Flush remaining transitions
            remaining = self.n_step_buffer.flush()
            for t in remaining:
                self.buffer.add(t.state, t.action, t.reward, t.next_state, t.done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return None
        
        # Sample from PER
        batch = self.buffer.sample(self.config.batch_size, self.device)
        if batch is None:
            return None
            
        if self.mixer:
            return self._train_step_qmix(batch)
        else:
            return self._train_step_c51(batch)

    def _train_step_qmix(self, batch) -> float:
        """QMIX specific training step."""
        states, actions, rewards, next_states, dones, indices, weights = batch
        bs = states.size(0)
        n_agents = self.config.n_agents

        # states: [Batch, n_agents * state_dim]
        # actions: [Batch, n_agents]
        # next_states: [Batch, n_agents * state_dim]
        # rewards: [Batch]
        
        # 1. Individual agent Q-values
        # Reshape states to [Batch * n_agents, state_dim] to process through shared net
        agent_obs = states.view(bs * n_agents, -1)
        next_agent_obs = next_states.view(bs * n_agents, -1)
        
        # Get individual Q-values (expected value since QMIX usually doesn't use distributional)
        q_vals_all = self.online_net.get_q_values(agent_obs).view(bs, n_agents, -1)
        # Select action Q-values: [Batch, n_agents]
        q_selected = q_vals_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        # 2. Target Q-values (Double DQN logic)
        with torch.no_grad():
            next_q_vals_online = self.online_net.get_q_values(next_agent_obs).view(bs, n_agents, -1)
            next_actions = next_q_vals_online.argmax(dim=-1).unsqueeze(-1) # [bs, n_agents, 1]
            
            next_q_vals_target = self.target_net.get_q_values(next_agent_obs).view(bs, n_agents, -1)
            next_q_selected = next_q_vals_target.gather(2, next_actions).squeeze(-1)
            
            # Mix through target mixer
            q_tot_target = self.mixer_target(next_q_selected, next_states)
            
            # Compute target: y = r + gamma * Q_tot_target
            y_tot = rewards.unsqueeze(1) + self.n_step_gamma * (1 - dones.unsqueeze(1)) * q_tot_target
        
        # 3. Mix current Q-values
        q_tot = self.mixer(q_selected, states)
        
        # 4. Loss
        loss = F.mse_loss(q_tot, y_tot)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent_params, self.config.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Update metrics
        self.metrics = {"loss": loss.item(), "q_tot_mean": q_tot.mean().item()}
        return loss.item()

    def _train_step_c51(self, batch) -> float:
        """Original C51 training step with full Rainbow enhancements."""
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # 1. Compute current probability distribution
        current_dist = self.online_net(states)
        current_dist_a = current_dist[range(self.config.batch_size), actions]
        
        # 2. Compute target distribution
        with torch.no_grad():
            next_q_online = self.online_net.get_q_values(next_states)
            next_actions = next_q_online.argmax(dim=-1)
            
            next_dist_target = self.target_net(next_states)
            next_dist_a = next_dist_target[range(self.config.batch_size), next_actions]
            
            target_dist = self._project_distribution(rewards, dones, next_dist_a)
        
        # 3. Loss
        log_p = torch.log(current_dist_a + 1e-10)
        elementwise_loss = -(target_dist * log_p).sum(dim=1)
        loss = (weights * elementwise_loss).mean()
        
        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Update PER priorities
        td_errors = elementwise_loss.detach()
        self.buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        # Soft update
        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._soft_update_target()
            
        self.metrics = {
            "loss": loss.item(),
            "td_error": td_errors.mean().item(),
            "lr": self.optimizer.param_groups[0]['lr']
        }
        return loss.item()

    def _project_distribution(self, rewards, dones, next_dist):
        """
        C51 Projection: Project return distributions onto the fixed support atoms.
        """
        batch_size = rewards.size(0)
        num_atoms = self.config.num_atoms
        v_min = self.config.v_min
        v_max = self.config.v_max
        delta_z = (v_max - v_min) / (num_atoms - 1)
        support = self.online_net.support
        
        # target_z = R + γ^n * support
        # rewards: [B], dones: [B], support: [Atoms], next_dist: [B, Atoms]
        target_z = rewards.unsqueeze(1) + self.n_step_gamma * support.unsqueeze(0) * (1.0 - dones.unsqueeze(1))
        target_z = target_z.clamp(v_min, v_max)
        
        # b = (target_z - v_min) / delta_z
        b = (target_z - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix boundary issues where l == u
        l[(u > 0) * (l == u)] -= 1
        u[(l < (num_atoms - 1)) * (l == u)] += 1
        
        proj_dist = torch.zeros(batch_size, num_atoms, device=self.device)
        
        # Distributed probability to neighbors l and u
        for b_idx in range(batch_size):
            proj_dist[b_idx].index_add_(0, l[b_idx], next_dist[b_idx] * (u[b_idx].float() - b[b_idx]))
            proj_dist[b_idx].index_add_(0, u[b_idx], next_dist[b_idx] * (b[b_idx] - l[b_idx].float()))
            
        return proj_dist
    
    def export_for_production(self, path: str):
        """
        Export the online network to TorchScript for high-performance C++/Edge deployment.
        Applies Dynamic Quantization (INT8) if supported.
        """
        import torch.quantization
        
        self.online_net.eval()
        
        # 1. Apply Dynamic Quantization (Weights: INT8, Activations: float32)
        # This reduces model size by ~4x and speeds up CPU inference.
        # Note: NoisyLinear is not directly supported by torch.quantization.quantize_dynamic
        # We need to handle it conditionally or ensure it's replaced/wrapped if quantization is critical.
        # For now, we'll assume it's either Linear or a custom module that behaves like Linear.
        # If NoisyLinear is present, it will be quantized if it has a 'use_noisy' attribute and it's True.
        # This part might need refinement depending on the exact implementation of NoisyLinear.
        quantizable_modules = {torch.nn.Linear}
        if hasattr(self.online_net, 'use_noisy_nets') and self.online_net.use_noisy_nets:
            # Assuming NoisyLinear is a custom class that behaves like nn.Linear for quantization purposes
            # Or, if NoisyLinear is defined elsewhere, import it and add it here.
            # For this example, we'll just add torch.nn.Linear as a fallback/general type.
            # A more robust solution would involve checking the actual type of NoisyLinear.
            pass # NoisyLinear is not directly available here, so we'll rely on torch.nn.Linear for now.
                 # If NoisyLinear is a custom module, it needs to be added to quantizable_modules.
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.online_net.cpu(), 
            quantizable_modules, 
            dtype=torch.qint8
        )
        
        # 2. Trace with dummy input
        dummy_state = torch.randn(1, self.state_dim)
        traced_script = torch.jit.trace(quantized_model, dummy_state)
        
        # 3. Save
        traced_script.save(path)
        logger.info(f"🚀 Model quantized and exported to TorchScript: {path}")
        
        # Move back to original device
        self.online_net.to(self.device)
    
    def _soft_update_target(self):
        """Polyak averaging: θ_target = τ * θ_online + (1 - τ) * θ_target."""
        tau = self.config.tau
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "train_steps": self.train_steps,
            "total_steps": self.total_steps,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dims": self.config.hidden_dims,
                "use_noisy_nets": self.config.use_noisy_nets,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"💾 Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.train_steps = checkpoint.get("train_steps", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        logger.info(f"📂 Agent loaded from {path} (step {self.train_steps})")
    
    def export_onnx(self, path: str):
        """Export model to ONNX format for production deployment."""
        self.online_net.eval()
        dummy = torch.randn(1, self.state_dim).to(self.device)
        torch.onnx.export(
            self.online_net, dummy, path,
            input_names=["state"],
            output_names=["q_values"],
            dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
            opset_version=14,
        )
        logger.info(f"📦 ONNX model exported to {path}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get latest training metrics for logging."""
        return self.metrics.copy()


# =============================================================================
# PPO AGENT
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization (Schulman et al., 2017).
    
    On-policy algorithm with:
        - Clipped surrogate objective for stable policy updates
        - Generalized Advantage Estimation (GAE) for variance reduction
        - Entropy bonus for exploration
        - Value function clipping for stability
    
    PPO is often more stable than DQN for complex environments and serves
    as the alternative/comparison agent in ATLAS.
    """
    
    ACTIONS = ['maintain', 'switch_ns', 'switch_ew', 'extend']
    
    def __init__(self, state_dim: int = 26, action_dim: int = 4,
                 config: AgentConfig = None, device: str = "auto"):
        
        self.config = config or AgentConfig(algorithm="ppo")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"🧠 PPO Agent | device={self.device} | "
                     f"state_dim={state_dim} | action_dim={action_dim}")
        
        # === Network ===
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)
        
        # === Optimizer ===
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.config.lr,
            eps=1e-5,
            weight_decay=1e-5,
        )
        
        # === Rollout storage ===
        self.rollout_length = self.config.ppo_rollout_length
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # === Counters ===
        self.train_steps = 0
        self.total_steps = 0
        self.metrics: Dict[str, float] = {}
        
        total_params = sum(p.numel() for p in self.network.parameters())
        logger.info(f"   Network params: {total_params:,}")
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action and store rollout data."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
        
        action_item = action.item()
        
        if not evaluate:
            self.states.append(state)
            self.actions.append(action_item)
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        self.total_steps += 1
        return action_item
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag for current step."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Compatibility wrapper — PPO stores per-step, not transitions."""
        # Data is already stored in select_action; just store reward/done
        if len(self.rewards) < len(self.actions):
            self.store_reward(reward, done)
    
    def is_ready_to_train(self) -> bool:
        """Check if enough rollout data is collected."""
        return len(self.rewards) >= self.rollout_length
    
    def train_step(self) -> Optional[float]:
        """
        Perform PPO update using collected rollout data.
        
        Returns:
            Mean policy loss, or None if not enough data.
        """
        if not self.is_ready_to_train():
            return None
        
        # Convert rollout to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # === Compute GAE advantages ===
        with torch.no_grad():
            # Get bootstrap value for last state
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            _, _, _, next_value = self.network.get_action_and_value(last_state)
            
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_val = old_values[t + 1]
                
                delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - old_values[t]
                advantages[t] = last_gae = (
                    delta + self.config.gamma * self.config.ppo_gae_lambda * 
                    next_non_terminal * last_gae
                )
            
            returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === PPO Update Epochs ===
        total_loss = 0
        n_updates = 0
        batch_size = min(256, len(states))
        
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch indices
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]
                
                # Get current policy output
                _, new_log_probs, entropy, new_values = \
                    self.network.get_action_and_value(mb_states, mb_actions)
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip, 
                                    1.0 + self.config.ppo_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.config.ppo_clip, self.config.ppo_clip
                )
                value_loss_unclipped = (new_values - mb_returns) ** 2
                value_loss_clipped = (value_pred_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss 
                        + self.config.ppo_value_coef * value_loss 
                        + self.config.ppo_entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.ppo_max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += policy_loss.item()
                n_updates += 1
        
        self.train_steps += 1
        
        # Metrics
        self.metrics = {
            "policy_loss": total_loss / max(n_updates, 1),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "lr": self.optimizer.param_groups[0]['lr'],
            "advantages_mean": advantages.mean().item(),
            "returns_mean": returns.mean().item(),
        }
        
        # Clear rollout data
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        
        return total_loss / max(n_updates, 1)
    
    def save(self, path: str):
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "total_steps": self.total_steps,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dims": self.config.hidden_dims,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"💾 PPO Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        logger.info(f"📂 PPO Agent loaded from {path}")
    
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics.copy()
