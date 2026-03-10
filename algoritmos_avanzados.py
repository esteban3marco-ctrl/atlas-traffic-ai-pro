"""
ATLAS Pro - Algoritmos Avanzados de Reinforcement Learning
===========================================================
Implementaciones de:
- Dueling DDQN (Double Deep Q-Network con arquitectura Dueling)
- Noisy Networks (exploración paramétrica)
- Distributional RL (C51 - Categorical DQN)
- Prioritized Experience Replay (PER)
- N-Step Returns

Compatible con PyTorch y la config train_config.yaml
"""

import os
import math
import random
import logging
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger("ATLAS.AlgoritmosAvanzados")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False
    logger.warning("PyTorch no disponible. Instalar con: pip install torch")


# =============================================================================
# NOISY LINEAR LAYER
# =============================================================================

if TORCH_DISPONIBLE:

    class NoisyLinear(nn.Module):
        """
        Noisy Linear Layer para exploración paramétrica.
        Reemplaza epsilon-greedy con ruido aprendido en los pesos.
        Ref: "Noisy Networks for Exploration" (Fortunato et al., 2018)
        """

        def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.sigma_init = sigma_init

            # Pesos aprendibles
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

            # Bias aprendible
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))

            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            mu_range = 1 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

        def _scale_noise(self, size: int) -> torch.Tensor:
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign().mul_(x.abs().sqrt_())

        def reset_noise(self):
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)


    # =============================================================================
    # DUELING DDQN NETWORK
    # =============================================================================

    class DuelingDDQN(nn.Module):
        """
        Dueling Double DQN con soporte para Noisy Networks.

        Arquitectura:
        - Feature extractor compartido
        - Stream de Valor V(s)
        - Stream de Ventaja A(s,a)
        - Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

        Ref: "Dueling Network Architectures for Deep RL" (Wang et al., 2016)
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dims: List[int] = [256, 256, 128],
                     use_noisy: bool = True, noisy_sigma: float = 0.5):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.use_noisy = use_noisy

            Linear = NoisyLinear if use_noisy else nn.Linear

            # Feature extractor compartido
            layers = []
            prev_dim = state_dim
            for i, dim in enumerate(hidden_dims[:-1]):
                if use_noisy:
                    layers.append(NoisyLinear(prev_dim, dim, noisy_sigma))
                else:
                    layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dim))
                prev_dim = dim
            self.feature_extractor = nn.Sequential(*layers)

            last_hidden = hidden_dims[-1]

            # Value stream V(s)
            if use_noisy:
                self.value_stream = nn.Sequential(
                    NoisyLinear(prev_dim, last_hidden, noisy_sigma),
                    nn.ReLU(),
                    NoisyLinear(last_hidden, 1, noisy_sigma)
                )
            else:
                self.value_stream = nn.Sequential(
                    nn.Linear(prev_dim, last_hidden),
                    nn.ReLU(),
                    nn.Linear(last_hidden, 1)
                )

            # Advantage stream A(s,a)
            if use_noisy:
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(prev_dim, last_hidden, noisy_sigma),
                    nn.ReLU(),
                    NoisyLinear(last_hidden, action_dim, noisy_sigma)
                )
            else:
                self.advantage_stream = nn.Sequential(
                    nn.Linear(prev_dim, last_hidden),
                    nn.ReLU(),
                    nn.Linear(last_hidden, action_dim)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature_extractor(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q(s,a) = V(s) + (A(s,a) - mean(A))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values

        def reset_noise(self):
            """Reset noise en todas las NoisyLinear layers"""
            if self.use_noisy:
                for module in self.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()


    # =============================================================================
    # CATEGORICAL DQN (C51) NETWORK
    # =============================================================================

    class CategoricalDQN(nn.Module):
        """
        Distributional RL usando C51 (Categorical DQN).
        En vez de estimar E[Q(s,a)], modela la distribución completa de retornos.

        Ref: "A Distributional Perspective on RL" (Bellemare et al., 2017)
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dims: List[int] = [256, 256, 128],
                     n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0,
                     use_noisy: bool = True, noisy_sigma: float = 0.5):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.n_atoms = n_atoms
            self.v_min = v_min
            self.v_max = v_max
            self.use_noisy = use_noisy

            self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
            self.delta_z = (v_max - v_min) / (n_atoms - 1)

            # Feature extractor
            layers = []
            prev_dim = state_dim
            for dim in hidden_dims[:-1]:
                if use_noisy:
                    layers.append(NoisyLinear(prev_dim, dim, noisy_sigma))
                else:
                    layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dim))
                prev_dim = dim
            self.feature_extractor = nn.Sequential(*layers)

            last_hidden = hidden_dims[-1]

            # Output: action_dim * n_atoms logits
            if use_noisy:
                self.output_layer = nn.Sequential(
                    NoisyLinear(prev_dim, last_hidden, noisy_sigma),
                    nn.ReLU(),
                    NoisyLinear(last_hidden, action_dim * n_atoms, noisy_sigma)
                )
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(prev_dim, last_hidden),
                    nn.ReLU(),
                    nn.Linear(last_hidden, action_dim * n_atoms)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Retorna distribuciones de probabilidad para cada acción"""
            features = self.feature_extractor(x)
            logits = self.output_layer(features)
            logits = logits.view(-1, self.action_dim, self.n_atoms)
            probs = F.softmax(logits, dim=2)
            return probs

        def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
            """Calcula Q-values como esperanza de la distribución"""
            probs = self.forward(x)
            q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            return q_values

        def reset_noise(self):
            if self.use_noisy:
                for module in self.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()


    # =============================================================================
    # PRIORITIZED EXPERIENCE REPLAY
    # =============================================================================

    class SumTree:
        """Árbol de sumas para muestreo proporcional eficiente O(log n)"""

        def __init__(self, capacity: int):
            self.capacity = capacity
            self.tree = np.zeros(2 * capacity - 1)
            self.data = [None] * capacity
            self.write_idx = 0
            self.size = 0

        def _propagate(self, idx: int, change: float):
            parent = (idx - 1) // 2
            self.tree[parent] += change
            if parent != 0:
                self._propagate(parent, change)

        def _retrieve(self, idx: int, s: float) -> int:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                return self._retrieve(left, s)
            else:
                return self._retrieve(right, s - self.tree[left])

        def total(self) -> float:
            return self.tree[0]

        def add(self, priority: float, data):
            idx = self.write_idx + self.capacity - 1
            self.data[self.write_idx] = data
            self.update(idx, priority)
            self.write_idx = (self.write_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        def update(self, idx: int, priority: float):
            change = priority - self.tree[idx]
            self.tree[idx] = priority
            self._propagate(idx, change)

        def get(self, s: float) -> Tuple[int, float, any]:
            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1
            return idx, self.tree[idx], self.data[data_idx]


    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


    class PrioritizedReplayBuffer:
        """
        Experience Replay con prioridad basada en TD-error.
        Muestrea experiencias más "sorprendentes" con mayor frecuencia.

        Ref: "Prioritized Experience Replay" (Schaul et al., 2016)
        """

        def __init__(self, capacity: int, alpha: float = 0.6,
                     beta_start: float = 0.4, beta_frames: int = 100000):
            self.tree = SumTree(capacity)
            self.capacity = capacity
            self.alpha = alpha
            self.beta_start = beta_start
            self.beta_frames = beta_frames
            self.frame = 0
            self.max_priority = 1.0
            self.min_priority = 1e-6

        @property
        def beta(self) -> float:
            """Beta annealing: de beta_start a 1.0"""
            return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        def push(self, state, action, reward, next_state, done):
            """Añade transición con prioridad máxima"""
            transition = Transition(state, action, reward, next_state, done)
            priority = self.max_priority ** self.alpha
            self.tree.add(priority, transition)

        def sample(self, batch_size: int) -> Tuple:
            """Muestrea batch con prioridad proporcional"""
            self.frame += 1
            indices = []
            priorities = []
            transitions = []

            segment = self.tree.total() / batch_size

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                idx, priority, data = self.tree.get(s)
                if data is not None:
                    indices.append(idx)
                    priorities.append(priority)
                    transitions.append(data)

            # Importance sampling weights
            total = self.tree.total()
            probs = np.array(priorities) / total
            weights = (self.tree.size * probs) ** (-self.beta)
            weights /= weights.max()

            states = torch.FloatTensor(np.array([t.state for t in transitions]))
            actions = torch.LongTensor(np.array([t.action for t in transitions]))
            rewards = torch.FloatTensor(np.array([t.reward for t in transitions]))
            next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
            dones = torch.FloatTensor(np.array([t.done for t in transitions]))
            weights = torch.FloatTensor(weights)

            return states, actions, rewards, next_states, dones, indices, weights

        def update_priorities(self, indices: List[int], td_errors: np.ndarray):
            """Actualiza prioridades basadas en TD-error"""
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + self.min_priority) ** self.alpha
                self.max_priority = max(self.max_priority, priority)
                self.tree.update(idx, priority)

        def __len__(self):
            return self.tree.size


    # =============================================================================
    # N-STEP RETURN BUFFER
    # =============================================================================

    class NStepBuffer:
        """Buffer para calcular retornos n-step"""

        def __init__(self, n_step: int = 3, gamma: float = 0.99):
            self.n_step = n_step
            self.gamma = gamma
            self.buffer = deque(maxlen=n_step)

        def push(self, transition: Transition) -> Optional[Transition]:
            """
            Añade transición y retorna transición n-step si el buffer está lleno.
            """
            self.buffer.append(transition)
            if len(self.buffer) < self.n_step:
                return None
            return self._compute_nstep()

        def _compute_nstep(self) -> Transition:
            """Calcula retorno n-step: R = r_0 + γ*r_1 + γ²*r_2 + ... + γⁿ⁻¹*r_{n-1}"""
            reward = 0.0
            for i, t in enumerate(self.buffer):
                reward += (self.gamma ** i) * t.reward
                if t.done:
                    return Transition(
                        self.buffer[0].state,
                        self.buffer[0].action,
                        reward,
                        t.next_state,
                        True
                    )
            return Transition(
                self.buffer[0].state,
                self.buffer[0].action,
                reward,
                self.buffer[-1].next_state,
                self.buffer[-1].done
            )

        def flush(self) -> List[Transition]:
            """Vacía el buffer retornando transiciones parciales"""
            transitions = []
            while len(self.buffer) > 0:
                t = self._compute_nstep()
                transitions.append(t)
                self.buffer.popleft()
            return transitions


    # =============================================================================
    # AGENTE DUELING DDQN COMPLETO
    # =============================================================================

    class AgenteDuelingDDQN:
        """
        Agente completo con Dueling DDQN + PER + N-Step + Noisy Networks.
        """

        def __init__(self, config: Dict = None):
            self.config = config or self._default_config()
            self.device = self._select_device()

            state_dim = self.config.get('state_dim', 26)
            action_dim = self.config.get('action_dim', 4)
            hidden_dims = self.config.get('hidden_dims', [256, 256, 128])
            use_noisy = self.config.get('use_noisy_nets', True)
            noisy_sigma = self.config.get('noisy_sigma', 0.5)

            # Redes
            self.policy_net = DuelingDDQN(
                state_dim, action_dim, hidden_dims, use_noisy, noisy_sigma
            ).to(self.device)

            self.target_net = DuelingDDQN(
                state_dim, action_dim, hidden_dims, use_noisy, noisy_sigma
            ).to(self.device)

            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            # Optimizador
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.get('lr', 0.0003)
            )

            # Replay Buffer
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.get('buffer_size', 100000),
                alpha=self.config.get('per_alpha', 0.6),
                beta_start=self.config.get('per_beta_start', 0.4),
                beta_frames=self.config.get('per_beta_frames', 50000)
            )

            # N-Step
            self.n_step_buffer = NStepBuffer(
                n_step=self.config.get('n_step', 3),
                gamma=self.config.get('gamma', 0.99)
            )

            # Hiperparámetros
            self.gamma = self.config.get('gamma', 0.99)
            self.tau = self.config.get('tau', 0.005)
            self.batch_size = self.config.get('batch_size', 64)
            self.min_buffer_size = self.config.get('min_buffer_size', 500)
            self.max_grad_norm = self.config.get('max_grad_norm', 10.0)
            self.target_update_freq = self.config.get('target_update_freq', 4)
            self.use_noisy = use_noisy

            # Epsilon (solo si no usa noisy)
            if not use_noisy:
                self.epsilon = 1.0
                self.epsilon_min = 0.01
                self.epsilon_decay = 0.9995
            else:
                self.epsilon = 0.0

            # Contadores
            self.steps = 0
            self.episodes = 0
            self.training_losses = []

            logger.info(f"AgenteDuelingDDQN creado en {self.device}")
            logger.info(f"  Noisy: {use_noisy}, N-Step: {self.config.get('n_step', 3)}")
            total_params = sum(p.numel() for p in self.policy_net.parameters())
            logger.info(f"  Parámetros: {total_params:,}")

        def _default_config(self) -> Dict:
            return {
                'state_dim': 26, 'action_dim': 4,
                'hidden_dims': [256, 256, 128],
                'lr': 0.0003, 'gamma': 0.99, 'tau': 0.005,
                'batch_size': 64, 'buffer_size': 100000,
                'min_buffer_size': 500, 'n_step': 3,
                'use_noisy_nets': True, 'noisy_sigma': 0.5,
                'per_alpha': 0.6, 'per_beta_start': 0.4,
                'per_beta_frames': 50000,
                'target_update_freq': 4, 'max_grad_norm': 10.0
            }

        def _select_device(self) -> torch.device:
            device_str = self.config.get('device', 'auto')
            if device_str == 'auto':
                return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.device(device_str)

        @torch.no_grad()
        def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
            """Selecciona acción con Noisy o epsilon-greedy"""
            if not evaluate and not self.use_noisy and random.random() < self.epsilon:
                return random.randint(0, self.config.get('action_dim', 4) - 1)

            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if not evaluate:
                self.policy_net.reset_noise()
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

        def store_transition(self, state, action, reward, next_state, done):
            """Almacena transición con N-Step processing"""
            transition = Transition(state, action, reward, next_state, done)
            nstep_transition = self.n_step_buffer.push(transition)

            if nstep_transition:
                self.replay_buffer.push(
                    nstep_transition.state, nstep_transition.action,
                    nstep_transition.reward, nstep_transition.next_state,
                    nstep_transition.done
                )

            if done:
                for t in self.n_step_buffer.flush():
                    self.replay_buffer.push(t.state, t.action, t.reward, t.next_state, t.done)

        def train_step(self) -> Optional[float]:
            """Un paso de entrenamiento"""
            if len(self.replay_buffer) < self.min_buffer_size:
                return None

            self.steps += 1

            # Sample
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)

            # Reset noise
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN: select action with policy, evaluate with target
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                n_step_gamma = self.gamma ** self.config.get('n_step', 3)
                target_q = rewards + (1 - dones) * n_step_gamma * next_q

            # TD errors for PER
            td_errors = (target_q - current_q).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

            # Weighted loss
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Soft update target network
            if self.steps % self.target_update_freq == 0:
                self._soft_update()

            # Decay epsilon
            if not self.use_noisy and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            loss_val = loss.item()
            self.training_losses.append(loss_val)
            return loss_val

        def _soft_update(self):
            """Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target"""
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )

        def get_metrics(self) -> Dict:
            """Métricas de entrenamiento"""
            recent_losses = self.training_losses[-100:] if self.training_losses else [0]
            return {
                'steps': self.steps,
                'episodes': self.episodes,
                'epsilon': self.epsilon,
                'buffer_size': len(self.replay_buffer),
                'avg_loss': np.mean(recent_losses),
                'beta': self.replay_buffer.beta
            }

        def save(self, path: str):
            """Guarda el agente completo"""
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps': self.steps,
                'episodes': self.episodes,
                'epsilon': self.epsilon,
                'config': self.config,
                'training_losses': self.training_losses[-1000:]
            }, path)
            logger.info(f"Agente guardado en {path}")

        def load(self, path: str) -> bool:
            """Carga el agente. Si la arquitectura cambio, ignora el modelo viejo."""
            if not os.path.exists(path):
                return False
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.steps = checkpoint.get('steps', 0)
                self.episodes = checkpoint.get('episodes', 0)
                self.epsilon = checkpoint.get('epsilon', 0.01)
                logger.info(f"Agente cargado desde {path} (step {self.steps})")
                return True
            except (RuntimeError, KeyError) as e:
                logger.warning(f"Modelo incompatible en {path} (arquitectura diferente). Entrenando desde cero.")
                return False


    # =============================================================================
    # AGENTE C51 (DISTRIBUTIONAL)
    # =============================================================================

    class AgenteC51:
        """
        Agente Categorical DQN (C51) con Dueling + PER + N-Step.
        Modela la distribución completa de retornos para mejor estimación.
        """

        def __init__(self, config: Dict = None):
            self.config = config or {}
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            state_dim = self.config.get('state_dim', 26)
            action_dim = self.config.get('action_dim', 4)
            self.action_dim = action_dim
            self.n_atoms = self.config.get('n_atoms', 51)
            self.v_min = self.config.get('v_min', -10.0)
            self.v_max = self.config.get('v_max', 10.0)
            self.gamma = self.config.get('gamma', 0.99)

            self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
            self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

            # Redes
            self.policy_net = CategoricalDQN(
                state_dim, action_dim, n_atoms=self.n_atoms,
                v_min=self.v_min, v_max=self.v_max
            ).to(self.device)

            self.target_net = CategoricalDQN(
                state_dim, action_dim, n_atoms=self.n_atoms,
                v_min=self.v_min, v_max=self.v_max
            ).to(self.device)

            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.get('lr', 0.0003)
            )

            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.get('buffer_size', 100000)
            )
            self.n_step_buffer = NStepBuffer(
                n_step=self.config.get('n_step', 3), gamma=self.gamma
            )

            self.batch_size = self.config.get('batch_size', 64)
            self.steps = 0

            logger.info(f"AgenteC51 creado: {self.n_atoms} átomos, [{self.v_min}, {self.v_max}]")

        @torch.no_grad()
        def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if not evaluate:
                self.policy_net.reset_noise()
            q_values = self.policy_net.get_q_values(state_t)
            return q_values.argmax(dim=1).item()

        def store_transition(self, state, action, reward, next_state, done):
            transition = Transition(state, action, reward, next_state, done)
            nstep = self.n_step_buffer.push(transition)
            if nstep:
                self.replay_buffer.push(nstep.state, nstep.action, nstep.reward, nstep.next_state, nstep.done)
            if done:
                for t in self.n_step_buffer.flush():
                    self.replay_buffer.push(t.state, t.action, t.reward, t.next_state, t.done)

        def train_step(self) -> Optional[float]:
            if len(self.replay_buffer) < self.batch_size * 4:
                return None

            self.steps += 1
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)

            # Projection of Tz onto support
            with torch.no_grad():
                self.target_net.reset_noise()
                self.policy_net.reset_noise()

                # Double DQN action selection
                next_q = self.policy_net.get_q_values(next_states)
                next_actions = next_q.argmax(dim=1)

                next_dist = self.target_net(next_states)
                next_dist = next_dist[torch.arange(self.batch_size), next_actions]

                n_gamma = self.gamma ** self.config.get('n_step', 3)
                Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * n_gamma * self.support.unsqueeze(0)
                Tz = Tz.clamp(self.v_min, self.v_max)

                b = (Tz - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                l = l.clamp(0, self.n_atoms - 1)
                u = u.clamp(0, self.n_atoms - 1)

                m = torch.zeros(self.batch_size, self.n_atoms, device=self.device)
                offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms,
                                       self.batch_size, device=self.device).long().unsqueeze(1)

                m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            # Current distribution
            current_dist = self.policy_net(states)
            current_dist = current_dist[torch.arange(self.batch_size), actions]

            # Cross-entropy loss
            loss = -(m * (current_dist + 1e-8).log()).sum(dim=1)

            # PER weights
            td_errors = loss.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

            loss = (weights * loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()

            # Soft update
            if self.steps % 4 == 0:
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.copy_(0.005 * pp.data + 0.995 * tp.data)

            return loss.item()

        def save(self, path: str):
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps': self.steps,
                'config': self.config
            }, path)

        def load(self, path: str) -> bool:
            if not os.path.exists(path):
                return False
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(ckpt['policy_net'])
                self.target_net.load_state_dict(ckpt['target_net'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.steps = ckpt.get('steps', 0)
                return True
            except (RuntimeError, KeyError) as e:
                logger.warning(f"Modelo incompatible en {path}. Entrenando desde cero.")
                return False


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def ejemplo_algoritmos_avanzados():
    """Demo de los algoritmos avanzados"""
    print("\n" + "=" * 70)
    print("🧠 ATLAS Pro - Algoritmos Avanzados de RL")
    print("=" * 70)

    if not TORCH_DISPONIBLE:
        print("❌ PyTorch no disponible")
        return

    print(f"\n✅ PyTorch {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")

    # Test Dueling DDQN
    print("\n📦 Creando AgenteDuelingDDQN...")
    agent = AgenteDuelingDDQN({
        'state_dim': 26, 'action_dim': 4,
        'hidden_dims': [256, 256, 128],
        'use_noisy_nets': True, 'buffer_size': 10000,
        'min_buffer_size': 100, 'batch_size': 32
    })

    print("🧪 Test de interacción...")
    for i in range(200):
        state = np.random.randn(26).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.uniform(-5, 5)
        next_state = np.random.randn(26).astype(np.float32)
        done = random.random() < 0.05

        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()

        if i % 50 == 0 and loss is not None:
            print(f"   Step {i}: loss={loss:.4f}, action={action}")

    metrics = agent.get_metrics()
    print(f"\n📊 Métricas: {metrics}")

    # Test C51
    print("\n📦 Creando AgenteC51...")
    c51 = AgenteC51({'state_dim': 26, 'action_dim': 4, 'buffer_size': 10000})
    state = np.random.randn(26).astype(np.float32)
    action = c51.select_action(state)
    print(f"   C51 acción: {action}")

    print("\n✅ Todos los algoritmos funcionando correctamente")


if __name__ == "__main__":
    ejemplo_algoritmos_avanzados()
