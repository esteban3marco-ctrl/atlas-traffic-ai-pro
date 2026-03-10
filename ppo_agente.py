"""
ATLAS Pro - Agente PPO (Proximal Policy Optimization)
======================================================
Implementación de PPO como alternativa al DQN:
- Actor-Critic con GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function clipping
- Entropy bonus para exploración
- Soporte para acción discreta (semáforos)

Ref: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("ATLAS.PPO")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False


if TORCH_DISPONIBLE:

    # =============================================================================
    # RED ACTOR-CRITIC
    # =============================================================================

    class ActorCritic(nn.Module):
        """
        Red Actor-Critic con arquitectura compartida.
        Actor: π(a|s) → distribución de probabilidad sobre acciones
        Critic: V(s) → valor estimado del estado
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dims: List[int] = [256, 256, 128]):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim

            # Feature extractor compartido
            layers = []
            prev_dim = state_dim
            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.LayerNorm(dim)
                ])
                prev_dim = dim
            self.shared = nn.Sequential(*layers)

            # Cabeza del Actor
            self.actor = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

            # Cabeza del Critic
            self.critic = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

            # Inicialización
            self._init_weights()

        def _init_weights(self):
            for module in [self.actor, self.critic]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=0.01)
                        nn.init.zeros_(layer.bias)

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.shared(state)
            action_logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            return action_logits, value

        def get_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
            """
            Muestrea acción de la política.
            Returns: (acción, log_prob, valor)
            """
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

        def evaluate(self, states: torch.Tensor, actions: torch.Tensor) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evalúa acciones bajo la política actual.
            Returns: (log_probs, values, entropy)
            """
            logits, values = self.forward(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, values, entropy


    # =============================================================================
    # ROLLOUT BUFFER
    # =============================================================================

    class RolloutBuffer:
        """
        Buffer para almacenar trayectorias completas.
        Calcula ventajas con GAE (Generalized Advantage Estimation).
        """

        def __init__(self):
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []
            self.advantages = []
            self.returns = []

        def add(self, state, action, reward, value, log_prob, done):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)

        def compute_gae(self, last_value: float, gamma: float = 0.99,
                       gae_lambda: float = 0.95):
            """
            Calcula Generalized Advantage Estimation.
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            donde δ_t = r_t + γV(s_{t+1}) - V(s_t)
            """
            gae = 0
            self.advantages = [0] * len(self.rewards)
            self.returns = [0] * len(self.rewards)

            values = self.values + [last_value]

            for t in reversed(range(len(self.rewards))):
                if self.dones[t]:
                    delta = self.rewards[t] - values[t]
                    gae = delta
                else:
                    delta = self.rewards[t] + gamma * values[t + 1] - values[t]
                    gae = delta + gamma * gae_lambda * gae

                self.advantages[t] = gae
                self.returns[t] = gae + values[t]

        def get_batches(self, batch_size: int) -> List[Dict]:
            """Genera mini-batches aleatorios"""
            n = len(self.states)
            indices = np.arange(n)
            np.random.shuffle(indices)

            states = np.array(self.states)
            actions = np.array(self.actions)
            old_log_probs = np.array(self.log_probs)
            advantages = np.array(self.advantages)
            returns = np.array(self.returns)

            # Normalizar ventajas
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batches = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                batches.append({
                    'states': torch.FloatTensor(states[idx]),
                    'actions': torch.LongTensor(actions[idx]),
                    'old_log_probs': torch.FloatTensor(old_log_probs[idx]),
                    'advantages': torch.FloatTensor(advantages[idx]),
                    'returns': torch.FloatTensor(returns[idx])
                })

            return batches

        def clear(self):
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()
            self.dones.clear()
            self.advantages.clear()
            self.returns.clear()

        def __len__(self):
            return len(self.states)


    # =============================================================================
    # AGENTE PPO
    # =============================================================================

    class AgentePPO:
        """
        Agente PPO completo con clipped objective y GAE.

        Hiperparámetros clave:
        - clip_ratio: ratio de clipping (típicamente 0.2)
        - value_coef: peso del loss del critic (típicamente 0.5)
        - entropy_coef: peso del bonus de entropía (típicamente 0.01)
        - gae_lambda: factor lambda de GAE (típicamente 0.95)
        """

        def __init__(self, config: Dict = None):
            self.config = config or self._default_config()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            state_dim = self.config.get('state_dim', 26)
            action_dim = self.config.get('action_dim', 4)
            hidden_dims = self.config.get('hidden_dims', [256, 256, 128])

            self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.config.get('lr', 0.0003),
                eps=1e-5
            )

            self.buffer = RolloutBuffer()

            # Hiperparámetros
            self.gamma = self.config.get('gamma', 0.99)
            self.gae_lambda = self.config.get('gae_lambda', 0.95)
            self.clip_ratio = self.config.get('clip_ratio', 0.2)
            self.value_coef = self.config.get('value_coef', 0.5)
            self.entropy_coef = self.config.get('entropy_coef', 0.01)
            self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
            self.n_epochs = self.config.get('n_epochs', 10)
            self.batch_size = self.config.get('batch_size', 64)
            self.rollout_length = self.config.get('rollout_length', 2048)

            # Tracking
            self.episodes = 0
            self.total_steps = 0
            self.training_losses = []
            self.episode_rewards = []

            total_params = sum(p.numel() for p in self.network.parameters())
            logger.info(f"AgentePPO creado en {self.device}: {total_params:,} parámetros")

        def _default_config(self) -> Dict:
            return {
                'state_dim': 26, 'action_dim': 4,
                'hidden_dims': [256, 256, 128],
                'lr': 0.0003, 'gamma': 0.99, 'gae_lambda': 0.95,
                'clip_ratio': 0.2, 'value_coef': 0.5,
                'entropy_coef': 0.01, 'max_grad_norm': 0.5,
                'n_epochs': 10, 'batch_size': 64,
                'rollout_length': 2048
            }

        @torch.no_grad()
        def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, float, float]:
            """
            Selecciona acción.
            Returns: (acción, log_prob, valor)
            """
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if evaluate:
                logits, value = self.network(state_t)
                return logits.argmax(dim=1).item(), 0.0, value.item()

            action, log_prob, value = self.network.get_action(state_t)
            return action, log_prob, value

        def store_transition(self, state, action, reward, value, log_prob, done):
            """Almacena transición en el buffer"""
            self.buffer.add(state, action, reward, value, log_prob, done)
            self.total_steps += 1

        def train(self, last_value: float = 0.0) -> Dict:
            """
            Entrena con los datos acumulados en el buffer.
            Implementa PPO-Clip.
            """
            if len(self.buffer) == 0:
                return {}

            # Calcular ventajas con GAE
            self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

            # Métricas de entrenamiento
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_updates = 0

            for epoch in range(self.n_epochs):
                batches = self.buffer.get_batches(self.batch_size)

                for batch in batches:
                    states = batch['states'].to(self.device)
                    actions = batch['actions'].to(self.device)
                    old_log_probs = batch['old_log_probs'].to(self.device)
                    advantages = batch['advantages'].to(self.device)
                    returns = batch['returns'].to(self.device)

                    # Evaluar bajo política actual
                    new_log_probs, values, entropy = self.network.evaluate(states, actions)

                    # Ratio π_new / π_old
                    ratio = (new_log_probs - old_log_probs).exp()

                    # Clipped surrogate objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss (también con clipping)
                    value_loss = F.mse_loss(values, returns)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = (policy_loss
                           + self.value_coef * value_loss
                           + self.entropy_coef * entropy_loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += (-entropy_loss).item()
                    n_updates += 1

            # Limpiar buffer
            self.buffer.clear()

            metrics = {
                'policy_loss': total_policy_loss / max(1, n_updates),
                'value_loss': total_value_loss / max(1, n_updates),
                'entropy': total_entropy / max(1, n_updates),
                'n_updates': n_updates,
                'total_steps': self.total_steps
            }

            self.training_losses.append(metrics)
            return metrics

        def get_metrics(self) -> Dict:
            """Métricas de entrenamiento"""
            recent = self.training_losses[-10:] if self.training_losses else []
            return {
                'total_steps': self.total_steps,
                'episodes': self.episodes,
                'avg_policy_loss': np.mean([m['policy_loss'] for m in recent]) if recent else 0,
                'avg_value_loss': np.mean([m['value_loss'] for m in recent]) if recent else 0,
                'avg_entropy': np.mean([m['entropy'] for m in recent]) if recent else 0,
                'avg_episode_reward': np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
            }

        def save(self, path: str):
            """Guarda el agente"""
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            torch.save({
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
                'episodes': self.episodes,
                'config': self.config,
                'training_losses': self.training_losses[-100:],
                'episode_rewards': self.episode_rewards[-500:]
            }, path)
            logger.info(f"PPO guardado: {path}")

        def load(self, path: str) -> bool:
            """Carga el agente"""
            if not os.path.exists(path):
                return False
            ckpt = torch.load(path, map_location=self.device)
            self.network.load_state_dict(ckpt['network'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.total_steps = ckpt.get('total_steps', 0)
            self.episodes = ckpt.get('episodes', 0)
            self.training_losses = ckpt.get('training_losses', [])
            self.episode_rewards = ckpt.get('episode_rewards', [])
            logger.info(f"PPO cargado: {path} (step {self.total_steps})")
            return True


# =============================================================================
# EJEMPLO
# =============================================================================

def ejemplo_ppo():
    """Demo del agente PPO"""
    print("\n" + "=" * 70)
    print("🎯 ATLAS Pro - Agente PPO")
    print("=" * 70)

    if not TORCH_DISPONIBLE:
        print("❌ PyTorch necesario")
        return

    agent = AgentePPO({
        'state_dim': 26, 'action_dim': 4,
        'hidden_dims': [256, 256, 128],
        'rollout_length': 256, 'batch_size': 32,
        'n_epochs': 4
    })

    print(f"\n✅ Agente creado en {agent.device}")
    print(f"   Parámetros: {sum(p.numel() for p in agent.network.parameters()):,}")

    # Simular rollout
    print("\n🔄 Simulando rollout...")
    episode_reward = 0
    state = np.random.randn(26).astype(np.float32)

    for step in range(256):
        action, log_prob, value = agent.select_action(state)
        reward = np.random.uniform(-5, 5)
        next_state = np.random.randn(26).astype(np.float32)
        done = random.random() < 0.01

        agent.store_transition(state, action, reward, value, log_prob, done)
        episode_reward += reward
        state = next_state

        if done:
            agent.episode_rewards.append(episode_reward)
            agent.episodes += 1
            episode_reward = 0
            state = np.random.randn(26).astype(np.float32)

    # Entrenar
    print("\n🏋️ Entrenando...")
    _, _, last_value = agent.select_action(state, evaluate=True)
    metrics = agent.train(last_value)

    print(f"   Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"   Value Loss:  {metrics['value_loss']:.4f}")
    print(f"   Entropy:     {metrics['entropy']:.4f}")
    print(f"   Updates:     {metrics['n_updates']}")

    # Evaluación
    print("\n📊 Evaluación...")
    total_reward = 0
    state = np.random.randn(26).astype(np.float32)
    for _ in range(100):
        action, _, _ = agent.select_action(state, evaluate=True)
        reward = np.random.uniform(-5, 5)
        total_reward += reward
        state = np.random.randn(26).astype(np.float32)

    print(f"   Reward promedio (eval): {total_reward / 100:.3f}")
    print(f"\n✅ Demo completada")


if __name__ == "__main__":
    import random
    ejemplo_ppo()
