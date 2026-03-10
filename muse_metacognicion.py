"""
ATLAS Pro - MUSE: Metacognition for Unknown Situations and Environments
========================================================================
Implementacion del framework MUSE adaptado para control de trafico.
Basado en: Valiente & Pilly (2024) - arXiv:2411.13537

Componentes:
  1. Self-Assessment (Conciencia de Competencia)
     - World Model: predice el siguiente estado y recompensa
     - Competence Estimator: estima probabilidad de exito del agente
     - Novelty Detector: detecta situaciones fuera de distribucion (OOD)

  2. Self-Regulation (Seleccion de Estrategia)
     - Strategy Selector: elige entre agente RL, fallback conservador,
       o solicitar intervencion humana
     - Performance Monitor: monitoriza rendimiento en tiempo real
       y ajusta la confianza del sistema

Uso:
    from muse_metacognicion import MUSEController
    muse = MUSEController(agent, config)
    action = muse.act(state)  # decide con metacognicion
"""

import os
import sys
import time
import logging
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger("ATLAS_MUSE")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.warning("PyTorch no disponible - MUSE desactivado")


if TORCH_OK:

    # =========================================================================
    # 1. WORLD MODEL - Predice el siguiente estado y recompensa
    # =========================================================================

    class WorldModel(nn.Module):
        """
        Modelo del mundo que predice:
          - next_state = f(state, action)
          - reward = g(state, action)

        Usado para evaluar planes de accion SIN ejecutarlos en SUMO.
        """

        def __init__(self, state_dim: int = 26, action_dim: int = 4,
                     hidden_dim: int = 256):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim

            # Encoder: state + action_one_hot -> hidden
            input_dim = state_dim + action_dim

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

            # Predictor de siguiente estado (delta)
            self.state_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )

            # Predictor de recompensa
            self.reward_predictor = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

            # Predictor de "done"
            self.done_predictor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, state: torch.Tensor, action: torch.Tensor):
            """
            Args:
                state: (batch, state_dim)
                action: (batch,) indices de accion
            Returns:
                next_state_pred, reward_pred, done_pred
            """
            # One-hot encode action
            action_oh = F.one_hot(action.long(), self.action_dim).float()
            x = torch.cat([state, action_oh], dim=-1)

            h = self.encoder(x)

            # Predecir delta de estado (residual connection)
            state_delta = self.state_predictor(h)
            next_state = state + state_delta

            reward = self.reward_predictor(h).squeeze(-1)
            done = self.done_predictor(h).squeeze(-1)

            return next_state, reward, done

        def predict(self, state: np.ndarray, action: int,
                    device: torch.device = None) -> Tuple[np.ndarray, float, float]:
            """Prediccion simple para un estado y accion."""
            if device is None:
                device = next(self.parameters()).device
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                a = torch.tensor([action]).to(device)
                ns, r, d = self.forward(s, a)
            return ns.cpu().numpy()[0], r.item(), d.item()


    # =========================================================================
    # 2. COMPETENCE ESTIMATOR - Estima probabilidad de exito
    # =========================================================================

    class CompetenceEstimator(nn.Module):
        """
        Estima la competencia del agente para un estado dado.
        Output: probabilidad [0, 1] de que el agente manejara bien la situacion.

        Entrenado con las recompensas reales:
          - Recompensa alta -> competencia alta (label ~1)
          - Recompensa baja -> competencia baja (label ~0)
        """

        def __init__(self, state_dim: int = 26, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state).squeeze(-1)

        def estimate(self, state: np.ndarray,
                     device: torch.device = None) -> float:
            """Estima competencia para un estado."""
            if device is None:
                device = next(self.parameters()).device
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                return self.forward(s).item()


    # =========================================================================
    # 3. NOVELTY DETECTOR - Detecta situaciones fuera de distribucion
    # =========================================================================

    class NoveltyDetector:
        """
        Detecta si un estado es "novel" (fuera de la distribucion de entrenamiento).

        Metodo: mantiene estadisticas de estados vistos durante entrenamiento.
        Si un nuevo estado esta muy lejos de la distribucion conocida,
        se marca como novel -> el agente deberia ser mas cauteloso.

        Usa distancia de Mahalanobis simplificada (z-score por dimension).
        """

        def __init__(self, state_dim: int = 26, window_size: int = 10000,
                     novelty_threshold: float = 5.0):
            self.state_dim = state_dim
            self.window_size = window_size
            self.threshold = novelty_threshold

            # Estadisticas running
            self.count = 0
            self.mean = np.zeros(state_dim, dtype=np.float64)
            self.M2 = np.zeros(state_dim, dtype=np.float64)  # sum of squares
            self.std = np.ones(state_dim, dtype=np.float64)

            # Historial reciente para adaptacion online
            self.recent_states = deque(maxlen=window_size)

        def update(self, state: np.ndarray):
            """Actualiza estadisticas con un nuevo estado (Welford online)."""
            self.count += 1
            delta = state - self.mean
            self.mean += delta / self.count
            delta2 = state - self.mean
            self.M2 += delta * delta2

            if self.count > 1:
                self.std = np.sqrt(self.M2 / (self.count - 1))
                self.std = np.maximum(self.std, 1e-8)  # evitar division por 0

            self.recent_states.append(state.copy())

        def novelty_score(self, state: np.ndarray) -> float:
            """
            Calcula score de novedad [0, inf).
            0 = perfectamente conocido
            >threshold = situacion desconocida
            """
            if self.count < 10:
                return 0.0  # no hay suficiente data

            z_scores = np.abs((state - self.mean) / self.std)
            # Media de z-scores (cuantas dimensiones estan fuera de lo normal)
            return float(np.mean(z_scores))

        def is_novel(self, state: np.ndarray) -> bool:
            """True si el estado es significativamente diferente a lo conocido."""
            return self.novelty_score(state) > self.threshold

        def get_novel_dimensions(self, state: np.ndarray,
                                 threshold: float = 2.0) -> List[int]:
            """Retorna indices de las dimensiones que son anormales."""
            if self.count < 10:
                return []
            z_scores = np.abs((state - self.mean) / self.std)
            return [i for i, z in enumerate(z_scores) if z > threshold]

        def get_stats(self) -> Dict:
            return {
                "count": self.count,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
            }


    # =========================================================================
    # 4. PERFORMANCE MONITOR - Monitoriza rendimiento en tiempo real
    # =========================================================================

    class PerformanceMonitor:
        """
        Monitoriza el rendimiento del agente en tiempo real.
        Detecta degradacion de rendimiento (reward bajando) y
        ajusta la confianza del sistema.
        """

        def __init__(self, window_short: int = 10, window_long: int = 50,
                     degradation_threshold: float = -0.2):
            self.window_short = window_short
            self.window_long = window_long
            self.degradation_threshold = degradation_threshold

            self.rewards_history = deque(maxlen=window_long * 2)
            self.step_rewards = deque(maxlen=1000)
            self.confidence = 0.5  # [0, 1] - confianza actual

            # Contadores
            self.total_steps = 0
            self.fallback_activations = 0
            self.novel_situations = 0
            self.successful_recoveries = 0

        def record_step(self, reward: float, is_novel: bool = False,
                        competence: float = 0.5):
            """Registra un paso de simulacion."""
            self.total_steps += 1
            self.step_rewards.append(reward)
            if is_novel:
                self.novel_situations += 1

            # Actualizar confianza con exponential moving average
            alpha = 0.01
            step_confidence = min(1.0, max(0.0, (reward + 20) / 40))
            self.confidence = (1 - alpha) * self.confidence + alpha * step_confidence

        def record_episode(self, total_reward: float):
            """Registra un episodio completo."""
            self.rewards_history.append(total_reward)

        def is_degrading(self) -> bool:
            """Detecta si el rendimiento esta bajando."""
            if len(self.rewards_history) < self.window_long:
                return False

            recent = list(self.rewards_history)
            short_avg = np.mean(recent[-self.window_short:])
            long_avg = np.mean(recent[-self.window_long:])

            if long_avg == 0:
                return False

            ratio = (short_avg - long_avg) / abs(long_avg)
            return ratio < self.degradation_threshold

        def get_confidence(self) -> float:
            """Retorna confianza actual del sistema [0, 1]."""
            return self.confidence

        def get_summary(self) -> Dict:
            """Resumen de rendimiento."""
            recent = list(self.step_rewards)[-100:] if self.step_rewards else [0]
            return {
                "confidence": round(self.confidence, 3),
                "total_steps": self.total_steps,
                "avg_recent_reward": round(float(np.mean(recent)), 2),
                "fallback_activations": self.fallback_activations,
                "novel_situations": self.novel_situations,
                "successful_recoveries": self.successful_recoveries,
                "is_degrading": self.is_degrading(),
            }


    # =========================================================================
    # 5. STRATEGY SELECTOR - Seleccion de estrategia metacognitiva
    # =========================================================================

    class StrategySelector:
        """
        Selecciona la estrategia basandose en la metacognicion.

        Estrategias disponibles:
          - "rl_agent": usar el agente entrenado (modo normal)
          - "conservative": semaforo con timing fijo seguro
          - "emergency_priority": priorizar vehiculos de emergencia
          - "balanced_rotation": rotacion equitativa entre fases
          - "request_human": solicitar intervencion humana

        La seleccion se basa en:
          - Competencia estimada del agente
          - Novedad de la situacion
          - Confianza del sistema
          - Presencia de emergencias
        """

        # Umbrales de decision (v2 - corregidos tras Ronda 3)
        # Ronda 3 mostro que MUSE intervenia demasiado (5736 fallbacks en evento)
        # El agente RL de Ronda 2 ya es bueno -> MUSE debe intervenir SOLO en extremos
        COMPETENCE_HIGH = 0.85   # el agente sabe lo que hace (antes: 0.7)
        COMPETENCE_LOW = 0.12    # el agente esta MUY perdido (antes: 0.3)
        NOVELTY_HIGH = 5.0       # situacion MUY diferente (antes: 3.0)
        CONFIDENCE_LOW = 0.15    # sistema en serios problemas (antes: 0.25)

        def __init__(self):
            self.current_strategy = "rl_agent"
            self.strategy_history = deque(maxlen=1000)
            self.strategy_counts = {
                "rl_agent": 0,
                "conservative": 0,
                "emergency_priority": 0,
                "balanced_rotation": 0,
                "request_human": 0,
            }

        def select(self, competence: float, novelty_score: float,
                   confidence: float, has_emergency: bool,
                   is_degrading: bool) -> str:
            """
            Selecciona la estrategia optima dada la metacognicion.

            Returns:
                Nombre de la estrategia a usar
            """
            # Regla 1: Emergencia + agente MUY perdido -> priorizar emergencias
            # (antes: competence < COMPETENCE_HIGH, ahora solo si realmente no sabe)
            if has_emergency and competence < self.COMPETENCE_LOW:
                strategy = "emergency_priority"

            # Regla 2: Situacion EXTREMADAMENTE novel + muy baja competencia -> conservador
            elif (novelty_score > self.NOVELTY_HIGH * 2.0
                  and competence < self.COMPETENCE_LOW):
                strategy = "request_human"

            # Regla 3: Situacion muy novel + baja competencia -> conservador
            elif (novelty_score > self.NOVELTY_HIGH
                  and competence < self.COMPETENCE_LOW):
                strategy = "conservative"

            # Regla 4: Rendimiento degradandose severamente -> conservador
            elif is_degrading and confidence < self.CONFIDENCE_LOW:
                strategy = "conservative"

            # Regla 5: Competencia extremadamente baja -> rotacion equilibrada
            elif competence < self.COMPETENCE_LOW * 0.5:
                strategy = "balanced_rotation"

            # Default: SIEMPRE usar agente RL (el agente ya es bueno)
            else:
                strategy = "rl_agent"

            self.current_strategy = strategy
            self.strategy_history.append(strategy)
            self.strategy_counts[strategy] += 1

            return strategy

        def get_stats(self) -> Dict:
            total = sum(self.strategy_counts.values()) or 1
            return {
                "current": self.current_strategy,
                "counts": dict(self.strategy_counts),
                "percentages": {
                    k: round(v / total * 100, 1)
                    for k, v in self.strategy_counts.items()
                },
            }


    # =========================================================================
    # 6. FALLBACK STRATEGIES - Estrategias de respaldo
    # =========================================================================

    class FallbackStrategies:
        """
        Estrategias de respaldo cuando el agente RL no es competente.
        Cada estrategia devuelve una accion (0-3) basada en reglas simples.
        """

        def __init__(self):
            self.phase_counter = 0
            self.rotation_phase = 0

        def conservative_action(self, state: np.ndarray) -> int:
            """
            Estrategia conservadora: dar verde a la direccion con mas cola.
            Acciones: 0=Mantener, 1=Fase N-S, 2=Fase E-O, 3=Extender
            """
            # Extraer colas del estado (indices 0, 6, 12, 18 en 26D)
            # N=0, S=6, E=12, W=18
            queue_ns = state[0] + state[6]   # colas N + S (normalizadas)
            queue_ew = state[12] + state[18]  # colas E + W (normalizadas)

            # Fase actual (indice 24)
            current_phase = state[24] * 3.0

            if queue_ns > queue_ew * 1.3:
                return 1  # cambiar a N-S (mas congestion)
            elif queue_ew > queue_ns * 1.3:
                return 2  # cambiar a E-O
            else:
                return 0  # mantener

        def emergency_action(self, state: np.ndarray) -> int:
            """
            Priorizar la direccion que tiene vehiculo de emergencia.
            Emergencias estan en indices 5, 11, 17, 23.
            """
            emerg_n = state[5]
            emerg_s = state[11]
            emerg_e = state[17]
            emerg_w = state[23]

            if emerg_n > 0 or emerg_s > 0:
                return 1  # verde para N-S
            elif emerg_e > 0 or emerg_w > 0:
                return 2  # verde para E-O
            else:
                return self.conservative_action(state)

        def balanced_rotation(self, state: np.ndarray) -> int:
            """
            Rotacion inteligente: alterna entre N-S y E-O basandose en colas.
            Si las colas son similares, alterna proporcional.
            """
            self.phase_counter += 1
            # Mirar colas para decidir proporcion
            queue_ns = state[0] + state[6]
            queue_ew = state[12] + state[18]
            total_q = queue_ns + queue_ew + 1e-8
            ratio_ns = queue_ns / total_q

            # Si NS tiene mas del 60% del trafico, dar verde NS
            if ratio_ns > 0.6:
                return 1
            elif ratio_ns < 0.4:
                return 2
            else:
                # Similar carga: alternar
                if self.phase_counter % 20 < 10:
                    return 1
                else:
                    return 2

        def get_action(self, strategy: str, state: np.ndarray) -> int:
            """Obtiene accion segun la estrategia."""
            if strategy == "conservative":
                return self.conservative_action(state)
            elif strategy == "emergency_priority":
                return self.emergency_action(state)
            elif strategy == "balanced_rotation":
                return self.balanced_rotation(state)
            else:
                return self.conservative_action(state)


    # =========================================================================
    # 7. WORLD MODEL TRAINER - Entrenamiento del modelo del mundo
    # =========================================================================

    class WorldModelTrainer:
        """Entrena el World Model con experiencias del agente."""

        def __init__(self, world_model: WorldModel, lr: float = 0.001,
                     buffer_size: int = 50000, batch_size: int = 64):
            self.model = world_model
            self.optimizer = optim.Adam(world_model.parameters(), lr=lr)
            self.batch_size = batch_size

            # Buffer simple
            self.states = deque(maxlen=buffer_size)
            self.actions = deque(maxlen=buffer_size)
            self.rewards = deque(maxlen=buffer_size)
            self.next_states = deque(maxlen=buffer_size)
            self.dones = deque(maxlen=buffer_size)

        def store(self, state, action, reward, next_state, done):
            """Almacena transicion para entrenar el world model."""
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(float(done))

        def train_step(self) -> Optional[float]:
            """Un paso de entrenamiento del world model."""
            if len(self.states) < self.batch_size:
                return None

            # Sample aleatorio
            indices = np.random.choice(len(self.states), self.batch_size, replace=False)

            device = next(self.model.parameters()).device

            states = torch.FloatTensor(np.array([self.states[i] for i in indices])).to(device)
            actions = torch.LongTensor(np.array([self.actions[i] for i in indices])).to(device)
            rewards = torch.FloatTensor(np.array([self.rewards[i] for i in indices])).to(device)
            next_states = torch.FloatTensor(np.array([self.next_states[i] for i in indices])).to(device)
            dones = torch.FloatTensor(np.array([self.dones[i] for i in indices])).to(device)

            # Forward
            pred_ns, pred_r, pred_d = self.model(states, actions)

            # Losses
            state_loss = F.mse_loss(pred_ns, next_states)
            reward_loss = F.mse_loss(pred_r, rewards)
            done_loss = F.binary_cross_entropy(pred_d, dones)

            total_loss = state_loss + reward_loss + 0.1 * done_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            return total_loss.item()


    # =========================================================================
    # 8. COMPETENCE TRAINER - Entrenamiento del estimador de competencia
    # =========================================================================

    class CompetenceTrainer:
        """
        Entrena el CompetenceEstimator.

        Label = sigmoid((reward - reward_mean) / reward_std)
        Recompensa alta relativa -> competencia alta
        """

        def __init__(self, estimator: CompetenceEstimator,
                     lr: float = 0.0005, buffer_size: int = 20000,
                     batch_size: int = 64):
            self.estimator = estimator
            self.optimizer = optim.Adam(estimator.parameters(), lr=lr)
            self.batch_size = batch_size

            self.states = deque(maxlen=buffer_size)
            self.rewards = deque(maxlen=buffer_size)

            # Estadisticas de recompensa para normalizar labels
            self.reward_mean = 0.0
            self.reward_std = 1.0
            self.reward_count = 0

        def store(self, state: np.ndarray, reward: float):
            """Almacena estado con su recompensa resultante."""
            self.states.append(state)
            self.rewards.append(reward)

            # Actualizar estadisticas (online)
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            if self.reward_count > 1:
                delta2 = reward - self.reward_mean
                # Running variance
                self.reward_std = max(1e-8, np.sqrt(
                    ((self.reward_count - 1) * self.reward_std**2 + delta * delta2)
                    / self.reward_count
                ))

        def train_step(self) -> Optional[float]:
            """Un paso de entrenamiento."""
            if len(self.states) < self.batch_size:
                return None

            indices = np.random.choice(len(self.states), self.batch_size, replace=False)

            device = next(self.estimator.parameters()).device

            states = torch.FloatTensor(
                np.array([self.states[i] for i in indices])
            ).to(device)

            rewards = np.array([self.rewards[i] for i in indices])

            # Labels: normalizar recompensas a [0, 1]
            normalized = (rewards - self.reward_mean) / max(self.reward_std, 1e-8)
            labels = 1.0 / (1.0 + np.exp(-normalized))  # sigmoid
            labels = torch.FloatTensor(labels).to(device)

            # Forward
            predictions = self.estimator(states)
            loss = F.binary_cross_entropy(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()


    # =========================================================================
    # 9. MUSE CONTROLLER - Controlador principal
    # =========================================================================

    class MUSEController:
        """
        Controlador principal de MUSE.
        Integra todos los componentes metacognitivos y envuelve al agente RL.

        Uso:
            agent = AgenteDuelingDDQN(config)
            muse = MUSEController(agent, config)

            state = env.reset()
            action = muse.act(state)       # accion metacognitiva
            muse.observe(state, action, reward, next_state, done)
            muse.train_metacognition()     # entrena componentes MUSE
        """

        def __init__(self, agent, config: dict = None):
            self.agent = agent
            self.config = config or {}

            agent_cfg = self.config.get("agent", {})
            state_dim = agent_cfg.get("state_dim", 26)
            action_dim = agent_cfg.get("action_dim", 4)

            # Dispositivo
            self.device = agent.device if hasattr(agent, 'device') else torch.device('cpu')

            # Componentes MUSE
            self.world_model = WorldModel(state_dim, action_dim).to(self.device)
            self.competence_estimator = CompetenceEstimator(state_dim).to(self.device)
            self.novelty_detector = NoveltyDetector(state_dim)
            self.performance_monitor = PerformanceMonitor()
            self.strategy_selector = StrategySelector()
            self.fallback = FallbackStrategies()

            # Trainers
            self.wm_trainer = WorldModelTrainer(self.world_model)
            self.comp_trainer = CompetenceTrainer(self.competence_estimator)

            # Config MUSE
            muse_cfg = self.config.get("muse", {})
            self.enabled = muse_cfg.get("enabled", True)
            self.train_interval = muse_cfg.get("train_interval", 10)
            self.log_interval = muse_cfg.get("log_interval", 100)

            # Warmup: MUSE solo observa durante los primeros N pasos (no interviene)
            # Esto permite calibrar competence estimator y novelty detector
            self.warmup_steps = muse_cfg.get("warmup_steps", 5000)

            # Cooldown: minimo de pasos entre fallbacks consecutivos
            # Evita que MUSE override al agente en rafagas
            self.fallback_cooldown = muse_cfg.get("fallback_cooldown", 50)
            self._last_fallback_step = -9999

            # Contadores
            self.step_count = 0
            self.episode_count = 0

            # Ultimo diagnostico
            self.last_diagnosis = {
                "competence": 0.5,
                "novelty": 0.0,
                "confidence": 0.5,
                "strategy": "rl_agent",
                "has_emergency": False,
                "is_novel": False,
            }

            logger.info("MUSE Controller inicializado")
            logger.info(f"  World Model params: {sum(p.numel() for p in self.world_model.parameters()):,}")
            logger.info(f"  Competence Estimator params: {sum(p.numel() for p in self.competence_estimator.parameters()):,}")

        def act(self, state: np.ndarray) -> int:
            """
            Selecciona accion con metacognicion.

            1. Evalua competencia del agente en este estado
            2. Detecta si la situacion es novel
            3. Selecciona estrategia (RL agent vs fallback)
            4. Retorna la accion

            Respeta warmup (solo observa al inicio) y cooldown entre fallbacks.
            """
            if not self.enabled:
                return self.agent.select_action(state)

            # Self-Assessment
            competence = self.competence_estimator.estimate(state, self.device)
            novelty = self.novelty_detector.novelty_score(state)
            is_novel = self.novelty_detector.is_novel(state)
            confidence = self.performance_monitor.get_confidence()
            is_degrading = self.performance_monitor.is_degrading()

            # Detectar emergencias (indices 5, 11, 17, 23 en estado 26D)
            has_emergency = (state[5] > 0 or state[11] > 0 or
                             state[17] > 0 or state[23] > 0)

            # --- WARMUP: durante los primeros N pasos, MUSE solo observa ---
            # No interviene, deja que el agente RL actue normal
            # Esto calibra competence estimator y novelty detector
            in_warmup = (self.step_count < self.warmup_steps)

            if in_warmup:
                strategy = "rl_agent"
            else:
                # Self-Regulation: seleccionar estrategia
                strategy = self.strategy_selector.select(
                    competence=competence,
                    novelty_score=novelty,
                    confidence=confidence,
                    has_emergency=has_emergency,
                    is_degrading=is_degrading,
                )

                # --- COOLDOWN: evitar rafagas de fallbacks ---
                # Si ya hizo un fallback hace poco, dejar al agente RL
                if strategy != "rl_agent":
                    steps_since_fallback = self.step_count - self._last_fallback_step
                    if steps_since_fallback < self.fallback_cooldown:
                        strategy = "rl_agent"

            # Guardar diagnostico
            self.last_diagnosis = {
                "competence": round(competence, 3),
                "novelty": round(novelty, 3),
                "confidence": round(confidence, 3),
                "strategy": strategy,
                "has_emergency": has_emergency,
                "is_novel": is_novel,
                "in_warmup": in_warmup,
            }

            # Ejecutar estrategia
            if strategy == "rl_agent":
                action = self.agent.select_action(state)
            elif strategy == "request_human":
                logger.warning("MUSE: Situacion muy desconocida - usando fallback conservador")
                action = self.fallback.get_action("conservative", state)
                self.performance_monitor.fallback_activations += 1
                self._last_fallback_step = self.step_count
            else:
                action = self.fallback.get_action(strategy, state)
                self.performance_monitor.fallback_activations += 1
                self._last_fallback_step = self.step_count

            return action

        def observe(self, state: np.ndarray, action: int, reward: float,
                    next_state: np.ndarray, done: bool):
            """
            Observa el resultado de una accion.
            Alimenta todos los componentes metacognitivos.
            """
            self.step_count += 1

            # Actualizar novelty detector
            self.novelty_detector.update(state)

            # Actualizar performance monitor
            is_novel = self.novelty_detector.is_novel(state)
            competence = self.last_diagnosis["competence"]
            self.performance_monitor.record_step(reward, is_novel, competence)

            # Almacenar para entrenamiento
            self.wm_trainer.store(state, action, reward, next_state, done)
            self.comp_trainer.store(state, reward)

            # Entrenar metacognicion periodicamente
            if self.step_count % self.train_interval == 0:
                self.train_metacognition()

            # Log periodico
            if self.step_count % self.log_interval == 0:
                diag = self.last_diagnosis
                logger.debug(
                    f"MUSE step {self.step_count}: "
                    f"comp={diag['competence']:.2f} "
                    f"nov={diag['novelty']:.2f} "
                    f"conf={diag['confidence']:.2f} "
                    f"strat={diag['strategy']}"
                )

            if done:
                self.episode_count += 1
                self.performance_monitor.record_episode(
                    sum(list(self.performance_monitor.step_rewards)[-100:])
                )

        def train_metacognition(self):
            """Entrena los componentes metacognitivos (world model + competence)."""
            wm_loss = self.wm_trainer.train_step()
            comp_loss = self.comp_trainer.train_step()
            return wm_loss, comp_loss

        def get_diagnosis(self) -> Dict:
            """Retorna el diagnostico completo del sistema."""
            return {
                "metacognition": self.last_diagnosis,
                "performance": self.performance_monitor.get_summary(),
                "strategy_stats": self.strategy_selector.get_stats(),
                "novelty_stats": {
                    "states_seen": self.novelty_detector.count,
                    "threshold": self.novelty_detector.threshold,
                },
                "step_count": self.step_count,
                "episode_count": self.episode_count,
            }

        def save(self, path: str):
            """Guarda los modelos MUSE."""
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save({
                "world_model": self.world_model.state_dict(),
                "competence_estimator": self.competence_estimator.state_dict(),
                "novelty_mean": self.novelty_detector.mean,
                "novelty_M2": self.novelty_detector.M2,
                "novelty_count": self.novelty_detector.count,
                "novelty_std": self.novelty_detector.std,
                "performance_confidence": self.performance_monitor.confidence,
                "strategy_counts": self.strategy_selector.strategy_counts,
                "step_count": self.step_count,
                "episode_count": self.episode_count,
            }, path)
            logger.info(f"MUSE guardado en {path}")

        def load(self, path: str) -> bool:
            """Carga los modelos MUSE."""
            if not os.path.exists(path):
                return False
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.world_model.load_state_dict(ckpt["world_model"])
                self.competence_estimator.load_state_dict(ckpt["competence_estimator"])
                self.novelty_detector.mean = ckpt["novelty_mean"]
                self.novelty_detector.M2 = ckpt["novelty_M2"]
                self.novelty_detector.count = ckpt["novelty_count"]
                self.novelty_detector.std = ckpt["novelty_std"]
                self.performance_monitor.confidence = ckpt.get("performance_confidence", 0.5)
                self.strategy_selector.strategy_counts = ckpt.get("strategy_counts", self.strategy_selector.strategy_counts)
                self.step_count = ckpt.get("step_count", 0)
                self.episode_count = ckpt.get("episode_count", 0)
                logger.info(f"MUSE cargado desde {path} (step {self.step_count})")
                return True
            except Exception as e:
                logger.warning(f"No se pudo cargar MUSE desde {path}: {e}")
                return False


    # =========================================================================
    # 10. DEMO
    # =========================================================================

    def demo_muse():
        """Demo de MUSE con datos sinteticos."""
        print("\n" + "=" * 70)
        print("  ATLAS Pro - MUSE: Metacognicion Demo")
        print("=" * 70)

        # Simular un agente dummy
        class DummyAgent:
            device = torch.device("cpu")
            def select_action(self, state):
                return np.random.randint(0, 4)

        agent = DummyAgent()
        config = {"agent": {"state_dim": 26, "action_dim": 4}}
        muse = MUSEController(agent, config)

        print("\n  Simulando 500 pasos con datos sinteticos...")

        for i in range(500):
            # Estado aleatorio
            state = np.random.randn(26).astype(np.float32) * 0.5

            # Simular emergencia ocasional
            if i % 100 == 50:
                state[5] = 1.0  # emergencia en N

            # Simular situacion novel
            if i > 400:
                state = state * 5.0  # valores extremos

            action = muse.act(state)
            reward = np.random.uniform(-5, 5)
            next_state = state + np.random.randn(26).astype(np.float32) * 0.1
            done = (i % 100 == 99)

            muse.observe(state, action, reward, next_state, done)

            if i % 100 == 99:
                diag = muse.get_diagnosis()
                print(f"\n  Step {i+1}:")
                print(f"    Competencia: {diag['metacognition']['competence']:.3f}")
                print(f"    Novedad:     {diag['metacognition']['novelty']:.3f}")
                print(f"    Confianza:   {diag['metacognition']['confidence']:.3f}")
                print(f"    Estrategia:  {diag['metacognition']['strategy']}")
                print(f"    Fallbacks:   {diag['performance']['fallback_activations']}")

        # Resumen final
        diag = muse.get_diagnosis()
        print("\n  Resumen de estrategias:")
        for strat, pct in diag["strategy_stats"]["percentages"].items():
            if pct > 0:
                print(f"    {strat}: {pct}%")

        # Test save/load
        muse.save("modelos/muse_test.pt")
        muse2 = MUSEController(agent, config)
        muse2.load("modelos/muse_test.pt")
        print(f"\n  Save/Load: OK (steps recuperados: {muse2.step_count})")

        print("\n" + "=" * 70)
        print("  MUSE Demo completada")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    if TORCH_OK:
        demo_muse()
    else:
        print("PyTorch necesario para MUSE")
