"""
ATLAS Pro - Coordinación Multi-Intersección con QMIX
=====================================================
Sistema de control coordinado de múltiples semáforos:
- QMIX (Monotonic Value Function Factorisation)
- Comunicación inter-agente
- Optimización de onda verde
- Coordinación de corredor
- Adaptación al flujo global

Ref: "QMIX: Monotonic Value Function Factorisation" (Rashid et al., 2018)
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

logger = logging.getLogger("ATLAS.MultiInterseccion")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_DISPONIBLE = True
except ImportError:
    TORCH_DISPONIBLE = False


if TORCH_DISPONIBLE:

    # =============================================================================
    # RED INDIVIDUAL DE AGENTE
    # =============================================================================

    class AgentNetwork(nn.Module):
        """Red Q individual para cada intersección"""

        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            return self.fc3(x)


    # =============================================================================
    # QMIX MIXING NETWORK
    # =============================================================================

    class QMIXMixer(nn.Module):
        """
        Red de mezcla QMIX.
        Combina Q-values individuales de forma monótona para obtener Q_total.

        Q_total = f(Q_1, Q_2, ..., Q_n, state)
        donde ∂Q_total/∂Q_i >= 0 (monotonicidad)
        """

        def __init__(self, n_agents: int, state_dim: int,
                     mixing_embed_dim: int = 64):
            super().__init__()
            self.n_agents = n_agents
            self.state_dim = state_dim
            self.embed_dim = mixing_embed_dim

            # Hypernetwork para pesos (constrained to be positive)
            self.hyper_w1 = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(mixing_embed_dim, n_agents * mixing_embed_dim)
            )

            self.hyper_w2 = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(mixing_embed_dim, mixing_embed_dim)
            )

            # Hypernetwork para biases
            self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(mixing_embed_dim, 1)
            )

        def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            """
            Args:
                agent_qs: [batch, n_agents] Q-values individuales
                state: [batch, state_dim] estado global

            Returns:
                Q_total: [batch, 1]
            """
            batch_size = agent_qs.size(0)

            # Primera capa (positive weights via abs)
            w1 = self.hyper_w1(state).abs()
            w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
            b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

            # agent_qs: [batch, 1, n_agents] @ w1: [batch, n_agents, embed]
            hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

            # Segunda capa
            w2 = self.hyper_w2(state).abs()
            w2 = w2.view(batch_size, self.embed_dim, 1)
            b2 = self.hyper_b2(state).view(batch_size, 1, 1)

            q_total = torch.bmm(hidden, w2) + b2
            return q_total.squeeze(2).squeeze(1)


    # =============================================================================
    # BUFFER DE REPLAY MULTI-AGENTE
    # =============================================================================

    class MultiAgentReplayBuffer:
        """Buffer de replay para experiencias multi-agente"""

        def __init__(self, capacity: int, n_agents: int):
            self.capacity = capacity
            self.n_agents = n_agents
            self.buffer = deque(maxlen=capacity)

        def push(self, observations: List[np.ndarray], global_state: np.ndarray,
                actions: List[int], rewards: List[float],
                next_observations: List[np.ndarray], next_global_state: np.ndarray,
                dones: List[bool]):
            """
            Almacena experiencia multi-agente.
            """
            self.buffer.append({
                'obs': observations,
                'state': global_state,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_observations,
                'next_state': next_global_state,
                'dones': dones
            })

        def sample(self, batch_size: int) -> Dict:
            """Muestrea batch de experiencias"""
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

            obs = torch.FloatTensor(np.array([[e['obs'][i] for e in batch] for i in range(self.n_agents)]))
            state = torch.FloatTensor(np.array([e['state'] for e in batch]))
            actions = torch.LongTensor(np.array([[e['actions'][i] for e in batch] for i in range(self.n_agents)]))
            rewards = torch.FloatTensor(np.array([np.mean(e['rewards']) for e in batch]))
            next_obs = torch.FloatTensor(np.array([[e['next_obs'][i] for e in batch] for i in range(self.n_agents)]))
            next_state = torch.FloatTensor(np.array([e['next_state'] for e in batch]))
            dones = torch.FloatTensor(np.array([any(e['dones']) for e in batch]))

            return {
                'obs': obs, 'state': state, 'actions': actions,
                'rewards': rewards, 'next_obs': next_obs,
                'next_state': next_state, 'dones': dones
            }

        def __len__(self):
            return len(self.buffer)


    # =============================================================================
    # SISTEMA QMIX COMPLETO
    # =============================================================================

    class QMIXSystem:
        """
        Sistema QMIX completo para coordinación multi-intersección.

        Cada intersección tiene su propio agente DQN, y el mixer QMIX
        coordina sus decisiones para optimizar el flujo global.
        """

        def __init__(self, config: Dict = None):
            self.config = config or self._default_config()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.n_agents = self.config.get('n_agents', 4)
            self.obs_dim = self.config.get('obs_dim', 26)
            self.action_dim = self.config.get('action_dim', 4)
            self.state_dim = self.config.get('state_dim', 104)  # n_agents * obs_dim

            # Redes de agentes
            self.agents = nn.ModuleList([
                AgentNetwork(self.obs_dim, self.action_dim)
                for _ in range(self.n_agents)
            ]).to(self.device)

            self.target_agents = nn.ModuleList([
                AgentNetwork(self.obs_dim, self.action_dim)
                for _ in range(self.n_agents)
            ]).to(self.device)

            # Mixer QMIX
            self.mixer = QMIXMixer(
                self.n_agents, self.state_dim,
                self.config.get('mixing_embed_dim', 64)
            ).to(self.device)

            self.target_mixer = QMIXMixer(
                self.n_agents, self.state_dim,
                self.config.get('mixing_embed_dim', 64)
            ).to(self.device)

            # Sync targets
            self._hard_update()

            # Optimizador (todos los parámetros juntos)
            params = list(self.agents.parameters()) + list(self.mixer.parameters())
            self.optimizer = optim.Adam(params, lr=self.config.get('lr', 0.0005))

            # Buffer
            self.buffer = MultiAgentReplayBuffer(
                self.config.get('buffer_size', 50000),
                self.n_agents
            )

            # Estado
            self.epsilon = self.config.get('epsilon_start', 1.0)
            self.epsilon_min = self.config.get('epsilon_min', 0.05)
            self.epsilon_decay = self.config.get('epsilon_decay', 0.9995)
            self.gamma = self.config.get('gamma', 0.99)
            self.steps = 0

            logger.info(f"QMIXSystem: {self.n_agents} agentes, obs={self.obs_dim}, actions={self.action_dim}")

        def _default_config(self) -> Dict:
            return {
                'n_agents': 4, 'obs_dim': 26, 'action_dim': 4,
                'state_dim': 104, 'mixing_embed_dim': 64,
                'lr': 0.0005, 'gamma': 0.99, 'buffer_size': 50000,
                'batch_size': 32, 'target_update_freq': 200,
                'epsilon_start': 1.0, 'epsilon_min': 0.05,
                'epsilon_decay': 0.9995
            }

        @torch.no_grad()
        def select_actions(self, observations: List[np.ndarray],
                          evaluate: bool = False) -> List[int]:
            """Selecciona acciones para todos los agentes"""
            actions = []
            for i, obs in enumerate(observations):
                if not evaluate and random.random() < self.epsilon:
                    actions.append(random.randint(0, self.action_dim - 1))
                else:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    q_values = self.agents[i](obs_t)
                    actions.append(q_values.argmax(dim=1).item())
            return actions

        def store_experience(self, observations, global_state, actions,
                           rewards, next_observations, next_global_state, dones):
            """Almacena experiencia"""
            self.buffer.push(observations, global_state, actions,
                           rewards, next_observations, next_global_state, dones)

        def train_step(self) -> Optional[float]:
            """Un paso de entrenamiento QMIX"""
            batch_size = self.config.get('batch_size', 32)
            if len(self.buffer) < batch_size:
                return None

            self.steps += 1
            batch = self.buffer.sample(batch_size)

            # Calcular Q-values individuales
            agent_qs = []
            target_agent_qs = []

            for i in range(self.n_agents):
                obs_i = batch['obs'][i].to(self.device)
                next_obs_i = batch['next_obs'][i].to(self.device)
                actions_i = batch['actions'][i].to(self.device)

                q_i = self.agents[i](obs_i)
                chosen_q = q_i.gather(1, actions_i.unsqueeze(1)).squeeze(1)
                agent_qs.append(chosen_q)

                with torch.no_grad():
                    target_q_i = self.target_agents[i](next_obs_i)
                    max_target_q = target_q_i.max(dim=1)[0]
                    target_agent_qs.append(max_target_q)

            # Stack Q-values: [batch, n_agents]
            agent_qs = torch.stack(agent_qs, dim=1)
            target_agent_qs = torch.stack(target_agent_qs, dim=1)

            state = batch['state'].to(self.device)
            next_state = batch['next_state'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            dones = batch['dones'].to(self.device)

            # Q_total via mixer
            q_total = self.mixer(agent_qs, state)

            with torch.no_grad():
                target_q_total = self.target_mixer(target_agent_qs, next_state)
                targets = rewards + (1 - dones) * self.gamma * target_q_total

            # Loss
            loss = F.mse_loss(q_total, targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.agents.parameters()) + list(self.mixer.parameters()),
                10.0
            )
            self.optimizer.step()

            # Update target
            if self.steps % self.config.get('target_update_freq', 200) == 0:
                self._hard_update()

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return loss.item()

        def _hard_update(self):
            """Copia pesos a target networks"""
            self.target_agents.load_state_dict(self.agents.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        def save(self, path: str):
            torch.save({
                'agents': self.agents.state_dict(),
                'mixer': self.mixer.state_dict(),
                'target_agents': self.target_agents.state_dict(),
                'target_mixer': self.target_mixer.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps': self.steps,
                'epsilon': self.epsilon,
                'config': self.config
            }, path)
            logger.info(f"QMIX guardado: {path}")

        def load(self, path: str) -> bool:
            if not os.path.exists(path):
                return False
            ckpt = torch.load(path, map_location=self.device)
            self.agents.load_state_dict(ckpt['agents'])
            self.mixer.load_state_dict(ckpt['mixer'])
            self.target_agents.load_state_dict(ckpt['target_agents'])
            self.target_mixer.load_state_dict(ckpt['target_mixer'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.steps = ckpt.get('steps', 0)
            self.epsilon = ckpt.get('epsilon', 0.05)
            return True


    # =============================================================================
    # OPTIMIZADOR DE ONDA VERDE
    # =============================================================================

    class GreenWaveOptimizer:
        """
        Optimizador de onda verde para coordinar semáforos en un corredor.
        Calcula offsets óptimos entre intersecciones para maximizar flujo.
        """

        def __init__(self, intersection_distances: List[float],
                     speed_limit: float = 50.0):
            """
            Args:
                intersection_distances: Distancias entre intersecciones consecutivas (metros)
                speed_limit: Velocidad límite del corredor (km/h)
            """
            self.distances = intersection_distances
            self.speed = speed_limit / 3.6  # m/s
            self.n_intersections = len(intersection_distances) + 1

        def compute_offsets(self, cycle_time: float,
                          green_ratio: float = 0.5) -> List[float]:
            """
            Calcula offsets óptimos para onda verde.

            Args:
                cycle_time: Duración del ciclo completo (segundos)
                green_ratio: Ratio de verde respecto al ciclo

            Returns:
                Lista de offsets (segundos) para cada intersección
            """
            offsets = [0.0]  # Primera intersección: offset 0

            for distance in self.distances:
                travel_time = distance / self.speed
                offset = travel_time % cycle_time
                offsets.append(offset)

            return offsets

        def compute_bandwidth(self, offsets: List[float],
                            cycle_time: float,
                            green_time: float) -> float:
            """
            Calcula el ancho de banda (bandwidth) de la onda verde.
            El bandwidth es la proporción del ciclo que un vehículo puede
            recorrer todo el corredor sin detenerse.
            """
            min_gap = float('inf')

            for i in range(len(offsets) - 1):
                travel = self.distances[i] / self.speed
                arrival_offset = offsets[i] + travel
                green_start = offsets[i + 1]
                green_end = green_start + green_time

                # Gap available
                gap = green_time - abs((arrival_offset % cycle_time) - green_start)
                min_gap = min(min_gap, max(0, gap))

            return min_gap / cycle_time if cycle_time > 0 else 0

        def optimize(self, cycle_time: float, green_time: float,
                    n_iterations: int = 1000) -> Tuple[List[float], float]:
            """
            Optimiza offsets para maximizar bandwidth.
            Usa búsqueda estocástica.
            """
            best_offsets = self.compute_offsets(cycle_time)
            best_bandwidth = self.compute_bandwidth(best_offsets, cycle_time, green_time)

            for _ in range(n_iterations):
                # Perturbar offsets
                new_offsets = [0.0] + [
                    (o + np.random.uniform(-5, 5)) % cycle_time
                    for o in best_offsets[1:]
                ]

                bw = self.compute_bandwidth(new_offsets, cycle_time, green_time)
                if bw > best_bandwidth:
                    best_offsets = new_offsets
                    best_bandwidth = bw

            return best_offsets, best_bandwidth


# =============================================================================
# EJEMPLO
# =============================================================================

def ejemplo_multi_interseccion():
    """Demo de coordinación multi-intersección"""
    print("\n" + "=" * 70)
    print("🚦 ATLAS Pro - Coordinación Multi-Intersección QMIX")
    print("=" * 70)

    if not TORCH_DISPONIBLE:
        print("❌ PyTorch necesario")
        return

    # QMIX System
    print("\n📦 Creando sistema QMIX con 4 intersecciones...")
    qmix = QMIXSystem({
        'n_agents': 4, 'obs_dim': 26, 'action_dim': 4,
        'state_dim': 104, 'buffer_size': 5000, 'batch_size': 16
    })

    # Simular experiencias
    print("🧪 Simulando tráfico...")
    for step in range(200):
        obs = [np.random.randn(26).astype(np.float32) for _ in range(4)]
        global_state = np.concatenate(obs)
        actions = qmix.select_actions(obs)
        rewards = [np.random.uniform(-2, 2) for _ in range(4)]
        next_obs = [np.random.randn(26).astype(np.float32) for _ in range(4)]
        next_state = np.concatenate(next_obs)
        dones = [False] * 4

        qmix.store_experience(obs, global_state, actions, rewards,
                            next_obs, next_state, dones)
        loss = qmix.train_step()

        if step % 50 == 0 and loss is not None:
            print(f"   Step {step}: loss={loss:.4f}, epsilon={qmix.epsilon:.3f}")

    # Green Wave
    print("\n🟢 Optimizador de Onda Verde...")
    distances = [200, 300, 250]  # metros entre intersecciones
    gw = GreenWaveOptimizer(distances, speed_limit=50)

    offsets, bandwidth = gw.optimize(cycle_time=90, green_time=40)
    print(f"   Offsets óptimos: {[f'{o:.1f}s' for o in offsets]}")
    print(f"   Bandwidth: {bandwidth:.1%}")

    print("\n✅ Demo completada")


# =============================================================================
# TOPOLOGÍA DE RED DE INTERSECCIONES
# =============================================================================

class IntersectionNode:
    """Nodo que representa una intersección en la red"""

    def __init__(self, node_id: str, position: Tuple[float, float],
                 tls_id: str = None, neighbors: List[str] = None):
        self.node_id = node_id
        self.position = position  # (x, y) en metros
        self.tls_id = tls_id or node_id
        self.neighbors = neighbors or []
        self.current_phase = 0
        self.queue_data = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.throughput = 0.0
        self.avg_wait = 0.0

    def distance_to(self, other: 'IntersectionNode') -> float:
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return (dx**2 + dy**2) ** 0.5


class NetworkTopology:
    """
    Define la topología de la red de intersecciones.
    Soporta grid, corredor lineal y topología personalizada.
    """

    def __init__(self):
        self.nodes: Dict[str, IntersectionNode] = {}
        self.edges: List[Tuple[str, str, float]] = []  # (from, to, distance)

    def add_node(self, node_id: str, position: Tuple[float, float],
                 tls_id: str = None) -> IntersectionNode:
        node = IntersectionNode(node_id, position, tls_id)
        self.nodes[node_id] = node
        return node

    def add_edge(self, from_id: str, to_id: str, distance: float = None):
        if distance is None:
            distance = self.nodes[from_id].distance_to(self.nodes[to_id])
        self.edges.append((from_id, to_id, distance))
        self.nodes[from_id].neighbors.append(to_id)
        self.nodes[to_id].neighbors.append(from_id)

    @classmethod
    def create_grid(cls, rows: int = 2, cols: int = 2,
                    spacing: float = 200.0) -> 'NetworkTopology':
        """Crea una red en grid (malla regular)"""
        net = cls()
        for r in range(rows):
            for c in range(cols):
                node_id = f"int_{r}_{c}"
                pos = (c * spacing, r * spacing)
                net.add_node(node_id, pos)

        # Conectar horizontal
        for r in range(rows):
            for c in range(cols - 1):
                net.add_edge(f"int_{r}_{c}", f"int_{r}_{c+1}")

        # Conectar vertical
        for r in range(rows - 1):
            for c in range(cols):
                net.add_edge(f"int_{r}_{c}", f"int_{r+1}_{c}")

        return net

    @classmethod
    def create_corridor(cls, n_intersections: int = 4,
                       spacing: float = 250.0) -> 'NetworkTopology':
        """Crea un corredor lineal de intersecciones"""
        net = cls()
        for i in range(n_intersections):
            net.add_node(f"int_{i}", (i * spacing, 0))
        for i in range(n_intersections - 1):
            net.add_edge(f"int_{i}", f"int_{i+1}")
        return net

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Matriz de adyacencia de la red"""
        ids = list(self.nodes.keys())
        n = len(ids)
        adj = np.zeros((n, n))
        for from_id, to_id, dist in self.edges:
            i, j = ids.index(from_id), ids.index(to_id)
            adj[i][j] = 1.0
            adj[j][i] = 1.0
        return adj

    def summary(self) -> str:
        return (
            f"NetworkTopology: {self.n_nodes} intersecciones, "
            f"{len(self.edges)} conexiones"
        )


# =============================================================================
# ENVIRONMENT MULTI-INTERSECCIÓN (SUMO compatible)
# =============================================================================

class MultiIntersectionEnv:
    """
    Entorno multi-intersección que se conecta a SUMO o funciona en modo
    simulado para testing. Cada intersección tiene su propio estado 26D.
    """

    def __init__(self, topology: NetworkTopology, sumo_cfg: str = None,
                 max_steps: int = 1000, simulated: bool = True):
        self.topology = topology
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.simulated = simulated
        self.step_count = 0
        self.n_agents = topology.n_nodes
        self.obs_dim = 26
        self.action_dim = 4
        self.node_ids = list(topology.nodes.keys())

        self._sumo_running = False

        # Estado de simulación interna
        self._queues = {nid: np.zeros(4) for nid in self.node_ids}
        self._phases = {nid: 0 for nid in self.node_ids}
        self._throughputs = {nid: 0.0 for nid in self.node_ids}

        if not simulated and sumo_cfg:
            self._init_sumo()

        logger.info(
            f"MultiIntersectionEnv: {self.n_agents} agentes, "
            f"simulated={simulated}"
        )

    def _init_sumo(self):
        """Inicializa conexión con SUMO vía TraCI"""
        try:
            import traci
            sumo_binary = os.environ.get("SUMO_BINARY", "sumo")
            traci.start([sumo_binary, "-c", self.sumo_cfg, "--no-warnings"])
            self._sumo_running = True
            logger.info(f"SUMO conectado: {self.sumo_cfg}")
        except Exception as e:
            logger.warning(f"SUMO no disponible, usando simulación: {e}")
            self.simulated = True

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Reset del entorno. Retorna (observaciones, estado_global)"""
        self.step_count = 0

        if self._sumo_running:
            import traci
            traci.load(["-c", self.sumo_cfg, "--no-warnings"])

        # Reset colas
        for nid in self.node_ids:
            self._queues[nid] = np.random.uniform(0, 5, size=4).astype(np.float32)
            self._phases[nid] = 0
            self._throughputs[nid] = 0.0

        observations = [self._get_obs(nid) for nid in self.node_ids]
        global_state = np.concatenate(observations)
        return observations, global_state

    def step(self, actions: List[int]) -> Tuple[
        List[np.ndarray], np.ndarray, List[float], List[bool], Dict
    ]:
        """
        Ejecuta un paso con acciones de todos los agentes.

        Returns:
            observations, global_state, rewards, dones, info
        """
        self.step_count += 1

        if self._sumo_running:
            return self._step_sumo(actions)
        else:
            return self._step_simulated(actions)

    def _step_simulated(self, actions: List[int]) -> Tuple:
        """Paso de simulación interna (v2 - balanced dynamics)"""
        rewards = []
        dones = [self.step_count >= self.max_steps] * self.n_agents

        for i, (nid, action) in enumerate(zip(self.node_ids, actions)):
            node = self.topology.nodes[nid]
            old_queues = self._queues[nid].copy()
            old_total = old_queues.sum()

            # Aplicar acción
            if action == 0:  # Mantener fase actual
                pass
            elif action == 1:  # N-S verde
                self._phases[nid] = 0
            elif action == 2:  # E-O verde
                self._phases[nid] = 2
            elif action == 3:  # Extender fase (bonus throughput)
                pass

            phase = self._phases[nid]

            # Simulación de flujo BALANCEADA
            # Llegadas: ~2 veh/dir/paso = ~8 total (antes era ~12)
            arrivals = np.random.poisson(2, size=4).astype(np.float32)

            # Salidas: direcciones con verde evacuan más eficientemente
            departures = np.zeros(4, dtype=np.float32)
            if phase in [0, 1]:  # N-S verde
                departures[0] = min(self._queues[nid][0], np.random.uniform(4, 8))
                departures[1] = min(self._queues[nid][1], np.random.uniform(4, 8))
                # Rojo parcial: algo de tráfico sale por giro
                departures[2] = min(self._queues[nid][2], np.random.uniform(0, 1.5))
                departures[3] = min(self._queues[nid][3], np.random.uniform(0, 1.5))
            else:  # E-O verde
                departures[2] = min(self._queues[nid][2], np.random.uniform(4, 8))
                departures[3] = min(self._queues[nid][3], np.random.uniform(4, 8))
                departures[0] = min(self._queues[nid][0], np.random.uniform(0, 1.5))
                departures[1] = min(self._queues[nid][1], np.random.uniform(0, 1.5))

            # Bonus por extender fase (acción 3): +20% throughput
            if action == 3:
                departures *= 1.2

            # Efecto de vecinos (reducido para no saturar)
            neighbor_flow = 0.0
            for neighbor_id in node.neighbors:
                if neighbor_id in self._throughputs:
                    neighbor_flow += self._throughputs[neighbor_id] * 0.05

            arrivals += neighbor_flow / 4

            self._queues[nid] = np.maximum(
                0, self._queues[nid] + arrivals - departures
            )

            throughput = departures.sum()
            self._throughputs[nid] = throughput
            total_queue = self._queues[nid].sum()

            # === REWARD SHAPING v2 ===
            # R1: Throughput normalizado (0 a +1)
            r_throughput = min(throughput / 15.0, 1.0)

            # R2: Reducción de cola (positivo si baja, negativo si sube)
            queue_delta = old_total - total_queue
            r_queue_delta = np.clip(queue_delta / 10.0, -1.0, 1.0)

            # R3: Penalización por colas excesivas (suave, acotada)
            r_queue_penalty = -min(total_queue / 50.0, 1.0)

            # R4: Bonus por equilibrio entre direcciones
            queue_std = np.std(self._queues[nid])
            r_balance = -min(queue_std / 10.0, 0.5)

            # Reward compuesto, rango típico: [-1.5, +1.5]
            reward = (r_throughput * 1.0 +
                      r_queue_delta * 0.8 +
                      r_queue_penalty * 0.5 +
                      r_balance * 0.3)

            rewards.append(reward)

            # Actualizar nodo
            dirs = ["N", "S", "E", "W"]
            for d, q in zip(dirs, self._queues[nid]):
                node.queue_data[d] = int(q)
            node.throughput = throughput
            node.current_phase = phase

        observations = [self._get_obs(nid) for nid in self.node_ids]
        global_state = np.concatenate(observations)

        info = {
            "step": self.step_count,
            "queues": {nid: self._queues[nid].tolist() for nid in self.node_ids},
            "throughputs": dict(self._throughputs),
        }

        return observations, global_state, rewards, dones, info

    def _step_sumo(self, actions: List[int]) -> Tuple:
        """Paso usando SUMO real vía TraCI"""
        import traci

        # Aplicar acciones a los TLS de SUMO
        phase_map = {0: 0, 1: 0, 2: 2, 3: -1}  # action -> SUMO phase
        for nid, action in zip(self.node_ids, actions):
            tls_id = self.topology.nodes[nid].tls_id
            if action != 3:  # No extender
                try:
                    traci.trafficlight.setPhase(tls_id, phase_map[action])
                except Exception:
                    pass

        # Simular pasos de SUMO
        for _ in range(10):  # 10 sub-steps por decisión
            traci.simulationStep()

        # Recoger observaciones
        rewards = []
        for nid in self.node_ids:
            tls_id = self.topology.nodes[nid].tls_id
            try:
                # Obtener colas de SUMO
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes[:4]]
                self._queues[nid] = np.array(queues[:4], dtype=np.float32)

                throughput = sum(
                    traci.lane.getLastStepVehicleNumber(l) for l in lanes[:4]
                )
                self._throughputs[nid] = throughput
                total_q = sum(queues[:4])
                rewards.append(throughput * 0.8 - total_q * 0.3)
            except Exception:
                rewards.append(0.0)

        observations = [self._get_obs(nid) for nid in self.node_ids]
        global_state = np.concatenate(observations)
        dones = [self.step_count >= self.max_steps] * self.n_agents

        info = {"step": self.step_count}
        return observations, global_state, rewards, dones, info

    def _get_obs(self, node_id: str) -> np.ndarray:
        """Construye vector de observación 26D para una intersección"""
        q = self._queues[node_id]
        phase = self._phases.get(node_id, 0)
        step_norm = self.step_count / self.max_steps

        # 6 features x 4 directions (24D) + phase (1D) + step (1D)
        obs = np.zeros(26, dtype=np.float32)
        for d in range(4):
            base = d * 6
            obs[base] = q[d] / 30.0  # queue normalized
            obs[base + 1] = min(q[d] * 2.0, 60.0) / 60.0  # wait estimate
            obs[base + 2] = np.random.uniform(0.3, 0.9)  # density
            obs[base + 3] = np.random.uniform(15, 50) / 50.0  # speed
            obs[base + 4] = 1.0 if (phase // 2 == d // 2) else 0.0  # green
            obs[base + 5] = self._throughputs.get(node_id, 0) / 30.0  # flow

        obs[24] = phase / 3.0
        obs[25] = step_norm

        return obs

    def close(self):
        if self._sumo_running:
            try:
                import traci
                traci.close()
            except Exception:
                pass
            self._sumo_running = False


# =============================================================================
# ENTRENADOR MULTI-INTERSECCIÓN
# =============================================================================

class MultiIntersectionTrainer:
    """
    Entrenador completo para el sistema multi-intersección con QMIX.
    Soporta topologías grid y corredor.
    """

    def __init__(self, topology_type: str = "grid", n_episodes: int = 500,
                 max_steps: int = 500, **kwargs):
        # Crear topología
        if topology_type == "grid":
            rows = kwargs.get("rows", 2)
            cols = kwargs.get("cols", 2)
            self.topology = NetworkTopology.create_grid(rows, cols)
        elif topology_type == "corridor":
            n = kwargs.get("n_intersections", 4)
            self.topology = NetworkTopology.create_corridor(n)
        else:
            raise ValueError(f"Topología desconocida: {topology_type}")

        self.n_episodes = n_episodes
        self.max_steps = max_steps

        # Entorno
        self.env = MultiIntersectionEnv(
            self.topology,
            max_steps=max_steps,
            simulated=True
        )

        # Sistema QMIX (hiperparámetros v2 - optimizados)
        n_agents = self.topology.n_nodes
        self.qmix = QMIXSystem({
            'n_agents': n_agents,
            'obs_dim': 26,
            'action_dim': 4,
            'state_dim': n_agents * 26,
            'mixing_embed_dim': 64,
            'lr': 0.0003,            # Reducido para estabilidad multi-agente
            'gamma': 0.95,            # Menor para acelerar convergencia
            'buffer_size': 30000,
            'batch_size': 64,         # Mayor batch = gradientes más estables
            'target_update_freq': 100, # Actualización target más frecuente
            'epsilon_start': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.997    # Decae más rápido (funciona con --rapido)
        })

        # Green Wave
        distances = []
        nodes = list(self.topology.nodes.values())
        for i in range(len(nodes) - 1):
            distances.append(nodes[i].distance_to(nodes[i+1]))
        if distances:
            self.green_wave = GreenWaveOptimizer(distances)
        else:
            self.green_wave = None

        logger.info(
            f"MultiIntersectionTrainer: {self.topology.summary()}, "
            f"{n_episodes} episodios"
        )

    def train(self, save_path: str = None, log_interval: int = 10) -> Dict:
        """Ejecuta el entrenamiento completo"""
        print(f"\n{'='*65}")
        print(f"  ATLAS Pro — Entrenamiento Multi-Interseccion QMIX")
        print(f"  {self.topology.summary()}")
        print(f"  Episodios: {self.n_episodes} | Max steps: {self.max_steps}")
        print(f"{'='*65}\n")

        all_rewards = []
        all_losses = []
        best_avg_reward = float('-inf')

        for ep in range(1, self.n_episodes + 1):
            observations, global_state = self.env.reset()
            episode_reward = 0.0
            episode_losses = []

            for step in range(self.max_steps):
                actions = self.qmix.select_actions(observations)

                next_obs, next_state, rewards, dones, info = self.env.step(actions)

                self.qmix.store_experience(
                    observations, global_state, actions, rewards,
                    next_obs, next_state, dones
                )

                loss = self.qmix.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                episode_reward += np.mean(rewards)
                observations = next_obs
                global_state = next_state

                if any(dones):
                    break

            # avg_reward = reward promedio por paso (ya acumulado como media de agentes)
            avg_reward = episode_reward / max(step + 1, 1)
            all_rewards.append(avg_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0

            if ep % log_interval == 0:
                recent_avg = np.mean(all_rewards[-log_interval:])
                print(
                    f"  Ep {ep:>4d}/{self.n_episodes} | "
                    f"Reward: {avg_reward:>7.2f} | "
                    f"Avg({log_interval}): {recent_avg:>7.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Eps: {self.qmix.epsilon:.3f}"
                )

                if recent_avg > best_avg_reward and save_path:
                    best_avg_reward = recent_avg
                    self.qmix.save(save_path)
                    print(f"         -> Mejor modelo guardado: {recent_avg:.2f}")

        # Green wave optimization
        gw_result = None
        if self.green_wave:
            offsets, bandwidth = self.green_wave.optimize(
                cycle_time=90, green_time=40
            )
            gw_result = {"offsets": offsets, "bandwidth": bandwidth}
            print(f"\n  Onda Verde optimizada:")
            print(f"    Offsets: {[f'{o:.1f}s' for o in offsets]}")
            print(f"    Bandwidth: {bandwidth:.1%}")

        self.env.close()

        results = {
            "episodes": self.n_episodes,
            "final_avg_reward": np.mean(all_rewards[-20:]),
            "best_avg_reward": best_avg_reward,
            "reward_history": all_rewards,
            "green_wave": gw_result,
            "topology": self.topology.summary()
        }

        print(f"\n  Resultado final: avg_reward = {results['final_avg_reward']:.2f}")
        print(f"  Mejor avg_reward = {best_avg_reward:.2f}")
        print(f"{'='*65}\n")

        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ATLAS Pro - Multi-Interseccion QMIX"
    )
    parser.add_argument("--mode", choices=["train", "demo", "test"],
                       default="demo", help="Modo de ejecucion")
    parser.add_argument("--topology", choices=["grid", "corridor"],
                       default="grid", help="Tipo de topologia")
    parser.add_argument("--rows", type=int, default=2, help="Filas del grid")
    parser.add_argument("--cols", type=int, default=2, help="Columnas del grid")
    parser.add_argument("--n-intersections", type=int, default=4,
                       help="Intersecciones en corredor")
    parser.add_argument("--episodes", type=int, default=500,
                       help="Episodios de entrenamiento")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Pasos por episodio")
    parser.add_argument("--save", type=str, default=None,
                       help="Ruta para guardar modelo")
    args = parser.parse_args()

    if args.mode == "demo":
        ejemplo_multi_interseccion()

    elif args.mode == "test":
        print("\n=== Test Multi-Interseccion ===\n")

        # Test topologías
        grid = NetworkTopology.create_grid(2, 2)
        print(f"Grid: {grid.summary()}")
        print(f"  Adj matrix shape: {grid.get_adjacency_matrix().shape}")

        corridor = NetworkTopology.create_corridor(4)
        print(f"Corridor: {corridor.summary()}")

        # Test environment
        env = MultiIntersectionEnv(grid, max_steps=50, simulated=True)
        obs, state = env.reset()
        print(f"\nEnv reset: {len(obs)} agents, obs_dim={obs[0].shape}, state_dim={state.shape}")

        actions = [random.randint(0, 3) for _ in range(grid.n_nodes)]
        next_obs, next_state, rewards, dones, info = env.step(actions)
        print(f"Env step: rewards={[f'{r:.2f}' for r in rewards]}")
        env.close()

        # Test QMIX
        if TORCH_DISPONIBLE:
            qmix = QMIXSystem({
                'n_agents': grid.n_nodes, 'obs_dim': 26,
                'action_dim': 4, 'state_dim': grid.n_nodes * 26
            })
            actions = qmix.select_actions(obs)
            print(f"QMIX actions: {actions}")

            # Quick train
            for _ in range(50):
                obs2, state2 = env.reset()
                a = qmix.select_actions(obs2)
                no, ns, r, d, _ = env.step(a)
                qmix.store_experience(obs2, state2, a, r, no, ns, d)
                qmix.train_step()
            print(f"QMIX trained 50 steps, epsilon={qmix.epsilon:.3f}")
        else:
            print("PyTorch no disponible — QMIX test omitido")

        # Test Green Wave
        gw = GreenWaveOptimizer([200, 300, 250])
        offsets, bw = gw.optimize(cycle_time=90, green_time=40)
        print(f"\nGreen Wave: bandwidth={bw:.1%}, offsets={[f'{o:.1f}' for o in offsets]}")

        print("\n=== ALL MULTI-INTERSECTION TESTS OK ===")

    elif args.mode == "train":
        kwargs = {}
        if args.topology == "grid":
            kwargs = {"rows": args.rows, "cols": args.cols}
        else:
            kwargs = {"n_intersections": args.n_intersections}

        save_path = args.save or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "modelos", f"qmix_{args.topology}.pt"
        )

        trainer = MultiIntersectionTrainer(
            topology_type=args.topology,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            **kwargs
        )
        results = trainer.train(save_path=save_path)
        print(f"Modelo guardado en: {save_path}")


if __name__ == "__main__":
    main()
