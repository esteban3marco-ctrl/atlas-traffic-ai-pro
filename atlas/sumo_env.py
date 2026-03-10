"""
ATLAS Pro — Gymnasium-Compatible SUMO Environment
====================================================
Professional traffic signal control environment wrapping SUMO simulator.

Features:
    - Gymnasium (gym) API compatible for use with any RL library
    - Enriched state space (26+ dimensions) with adaptive normalization
    - Multi-intersection support (MARL)
    - Curriculum learning with progressive difficulty
    - Configurable phase constraints and safety limits
"""

import os
import sys
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from atlas.config import EnvironmentConfig, RewardConfig
from atlas.rewards import MultiObjectiveReward, PotentialBasedRewardShaping

logger = logging.getLogger("ATLAS.Env")

# Attempt to import traci (SUMO's Python API)
_TRACI_AVAILABLE = False
try:
    import traci
    import sumolib
    _TRACI_AVAILABLE = True
except ImportError:
    logger.warning("traci/sumolib not found. SUMO simulation unavailable. "
                   "Install with: pip install traci sumolib")


# =============================================================================
# RUNNING STATISTICS FOR ADAPTIVE NORMALIZATION
# =============================================================================

class RunningMeanStd:
    """
    Tracks running mean and standard deviation for online normalization.
    Uses Welford's algorithm for numerical stability.
    """

    def __init__(self, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Small initial count for stability

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# =============================================================================
# ATLAS TRAFFIC ENVIRONMENT
# =============================================================================

class ATLASTrafficEnv(gym.Env):
    """
    ATLAS Traffic Signal Control Environment.

    Gymnasium-compatible environment wrapping SUMO for traffic signal
    optimization using deep reinforcement learning.

    State Space (26 dimensions for single intersection):
        Per direction (N, S, E, W) × 5 features = 20:
            - Queue length (normalized)
            - Average speed (normalized)
            - Accumulated waiting time (normalized)
            - Vehicle density (vehicles/lane length)
            - Number of halted vehicles (normalized)
        Global features (6):
            - Current phase (one-hot encoded, 4 dims)
            - Time in current phase (normalized)
            - Time of day (normalized, 0-1)

    Action Space (Discrete, 4 actions):
        0: Maintain current phase
        1: Switch to North-South green
        2: Switch to East-West green
        3: Extend current phase (same as maintain but signals intent)

    Reward:
        Multi-objective: queue + wait + throughput + fairness + emissions
    """

    metadata = {"render_modes": ["human", "sumo_gui"]}

    # Actions
    ACTION_MAINTAIN = 0
    ACTION_SWITCH_NS = 1
    ACTION_SWITCH_EW = 2
    ACTION_EXTEND = 3
    ACTION_NAMES = ['maintain', 'switch_ns', 'switch_ew', 'extend']

    def __init__(
        self,
        env_config: EnvironmentConfig = None,
        reward_config: RewardConfig = None,
        render_mode: Optional[str] = None,
        episode_number: int = 0,
    ):
        super().__init__()

        self.env_config = env_config or EnvironmentConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self._episode_number = episode_number

        # State dimension
        n_dirs = len(self.env_config.directions)
        self.features_per_dir = 5
        self.global_features = 6  # phase_onehot(4) + time_in_phase(1) + time_of_day(1)
        self.state_dim = n_dirs * self.features_per_dir + self.global_features

        # Define spaces
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # Reward function
        self._reward_fn = MultiObjectiveReward(self.reward_config)
        self._reward_shaping = PotentialBasedRewardShaping(
            gamma=0.99, scale=0.1
        )

        # State normalization
        self._obs_rms = RunningMeanStd(shape=(self.state_dim,))
        self._normalize_obs = True

        # Internal state
        self._connected = False
        self._current_phase = 0
        self._steps_in_phase = 0
        self._step_count = 0
        self._total_arrived = 0
        self._prev_arrived = 0

        # Phase timing (in simulation steps)
        self._min_green_steps = int(self.env_config.min_green_time / self.env_config.step_length)
        self._max_green_steps = int(self.env_config.max_green_time / self.env_config.step_length)
        self._yellow_steps = int(self.env_config.yellow_time / self.env_config.step_length)
        self._all_red_steps = int(self.env_config.all_red_time / self.env_config.step_length)
        self._decision_steps = int(self.env_config.delta_time / self.env_config.step_length)

        # Traffic light IDs
        self._tl_ids = self.env_config.traffic_light_ids
        self.n_agents = len(self._tl_ids) if self.env_config.is_multi_agent else 1
        
        # Internal state (per agent)
        self._tl_states = {tl_id: {"phase": 0, "steps": 0} for tl_id in self._tl_ids}

        # Curriculum learning
        self._flow_scale = 1.0
        if self.env_config.use_curriculum:
            self._update_curriculum(episode_number)

        # Metrics tracking
        self._episode_metrics = {
            "total_wait_time": 0.0,
            "total_throughput": 0,
            "max_queue": 0,
            "avg_speed": 0.0,
            "phase_changes": 0,
            "total_reward": 0.0,
        }

        logger.info(f"🚦 ATLASTrafficEnv initialized | state_dim={self.state_dim} | "
                     f"flow_scale={self._flow_scale:.1f}")

    def _update_curriculum(self, episode: int):
        """Update flow scale based on curriculum stage."""
        for stage in self.env_config.curriculum_stages:
            ep_range = stage["episode_range"]
            if ep_range[0] <= episode < ep_range[1]:
                self._flow_scale = stage["flow_scale"]
                logger.debug(f"📚 Curriculum: {stage['description']} "
                             f"(flow_scale={self._flow_scale})")
                break

    def set_episode(self, episode: int):
        """Update episode number for curriculum learning."""
        self._episode_number = episode
        if self.env_config.use_curriculum:
            self._update_curriculum(episode)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Disconnect previous simulation
        if self._connected:
            self._disconnect()

        # Connect to SUMO
        self._connect()

        # Reset internal state
        self._tl_states = {tl_id: {"phase": 0, "steps": 0} for tl_id in self._tl_ids}
        self._total_arrived = 0
        self._prev_arrived = 0
        self._reward_fn.reset()
        self._reward_shaping.reset()

        # Domain Randomization: Jitter timings
        if self.env_config.use_domain_randomization:
            self._yellow_steps = int((self.env_config.yellow_time + self.np_random.uniform(-1, 1)) / self.env_config.step_length)
            self._decision_steps = int((self.env_config.delta_time + self.np_random.uniform(-0.5, 0.5)) / self.env_config.step_length)
            logger.debug(f"🎲 Randomization: yellow={self._yellow_steps}, decision={self._decision_steps}")
        
        self._episode_metrics = {
            "total_wait_time": 0.0,
            "total_throughput": 0,
            "max_queue": 0,
            "avg_speed": 0.0,
            "phase_changes": 0,
            "total_reward": 0.0,
        }

        # Warmup: run simulation without agent decisions
        for _ in range(self.env_config.warmup_steps):
            traci.simulationStep()
            self._step_count += 1

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, actions: Union[int, np.ndarray, List[int]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one decision step for all agents.

        Args:
            actions: List of actions or a single action (if single agent)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if not isinstance(actions, (list, np.ndarray, torch.Tensor)):
            actions = [int(actions)]
        
        phase_changed_any = False
        per_agent_info = {}

        for i, (tl_id, action) in enumerate(zip(self._tl_ids, actions)):
            p_changed = False
            curr_state = self._tl_states[tl_id]
            curr_phase = curr_state["phase"]
            curr_steps = curr_state["steps"]

            # Execute action
            if action == self.ACTION_SWITCH_NS:
                if curr_phase != 0 and curr_steps >= self._min_green_steps:
                    self._change_phase(tl_id, 0)
                    p_changed = True
            elif action == self.ACTION_SWITCH_EW:
                if curr_phase != 2 and curr_steps >= self._min_green_steps:
                    self._change_phase(tl_id, 2)
                    p_changed = True

            # Force change if max green exceeded
            if self._tl_states[tl_id]["steps"] >= self._max_green_steps:
                new_phase = 2 if self._tl_states[tl_id]["phase"] == 0 else 0
                self._change_phase(tl_id, new_phase)
                p_changed = True
            
            phase_changed_any = phase_changed_any or p_changed
            per_agent_info[tl_id] = {"phase_changed": p_changed}

        # Advance simulation by delta_time steps
        for _ in range(self._decision_steps):
            if not self._is_active():
                break
            traci.simulationStep()
            self._step_count += 1
            for tl_id in self._tl_ids:
                self._tl_states[tl_id]["steps"] += 1

        # Compute state and reward
        obs = self._get_observation()
        traffic_data = self._get_traffic_data()

        # Joint Reward (Simplified: calculated from global traffic data)
        reward = self._reward_fn.compute(
            queue_lengths=traffic_data["queues"],
            wait_times=traffic_data["waits"],
            speeds=traffic_data["speeds"],
            throughput=traffic_data["throughput"],
            phase_changed=phase_changed_any,
            emergency_waiting=traffic_data.get("emergency", False),
            halted_vehicles=traffic_data.get("halted", None),
        )

        # Add potential-based shaping
        shaping = self._reward_shaping.compute_shaping(traffic_data["queues"])
        reward += shaping

        # Check termination
        terminated = not self._is_active()
        truncated = self._step_count >= (self.env_config.max_steps / self.env_config.step_length)

        # Update metrics (Joint metrics)
        self._episode_metrics["total_reward"] += reward
        self._episode_metrics["total_throughput"] += traffic_data["throughput"]
        self._episode_metrics["total_wait_time"] += sum(traffic_data["waits"].values())
        self._episode_metrics["max_queue"] = max(
            self._episode_metrics["max_queue"],
            max(traffic_data["queues"].values()) if traffic_data["queues"] else 0
        )
        if phase_changed_any:
            self._episode_metrics["phase_changes"] += 1

        speed_vals = list(traffic_data["speeds"].values())
        if speed_vals:
            self._episode_metrics["avg_speed"] = np.mean(speed_vals)

        info = self._get_info()
        info["traffic_data"] = traffic_data
        info["phase_changed"] = phase_changed_any
        info["per_agent"] = per_agent_info

        if terminated or truncated:
            info["episode_metrics"] = self._episode_metrics.copy()
            self._disconnect()

        return obs, reward, terminated, truncated, info

    def _connect(self):
        """Connect to SUMO simulator."""
        if not _TRACI_AVAILABLE:
            raise RuntimeError("SUMO (traci) is not installed!")

        cfg = self.env_config
        use_gui = cfg.gui or self.render_mode == "sumo_gui"
        sumo_binary = "sumo-gui" if use_gui else "sumo"

        cmd = [
            sumo_binary,
            "-c", cfg.sumo_cfg_file,
            "--step-length", str(cfg.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--random",
        ]

        # Use unique label for parallel envs
        label = f"atlas_{id(self)}"
        traci.start(cmd, label=label)
        self._conn = traci.getConnection(label)
        self._connected = True
        logger.debug("Connected to SUMO")

    def _disconnect(self):
        """Disconnect from SUMO."""
        if self._connected:
            try:
                self._conn.close()
            except Exception:
                pass
            self._connected = False

    def _is_active(self) -> bool:
        """Check if simulation has vehicles remaining."""
        try:
            return self._conn.simulation.getMinExpectedNumber() > 0
        except Exception:
            return False

    def _change_phase(self, tl_id: str, new_phase: int):
        """Change traffic light phase with yellow transition and all-red clearance."""
        conn = self._conn
        current_phase = self._tl_states[tl_id]["phase"]

        # Yellow phase
        yellow_phase = current_phase + 1
        try:
            conn.trafficlight.setPhase(tl_id, yellow_phase)
        except Exception:
            pass

        for _ in range(self._yellow_steps):
            if not self._is_active():
                break
            conn.simulationStep()
            self._step_count += 1
            # In multi-agent, other TLs might still be in green, but for simplicity
            # we count steps for all.
            for t in self._tl_ids:
                self._tl_states[t]["steps"] += 1

        # All-red clearance interval
        for _ in range(self._all_red_steps):
            if not self._is_active():
                break
            conn.simulationStep()
            self._step_count += 1
            for t in self._tl_ids:
                self._tl_states[t]["steps"] += 1

        # Set new phase
        try:
            conn.trafficlight.setPhase(tl_id, new_phase)
        except Exception:
            pass

        self._tl_states[tl_id]["phase"] = new_phase
        self._tl_states[tl_id]["steps"] = 0

    def _get_observation(self) -> np.ndarray:
        """Collect observations for all traffic lights."""
        obs_list = []
        for tl_id in self._tl_ids:
            obs_list.append(self._get_tl_observation(tl_id))
        
        obs = np.array(obs_list, dtype=np.float32)
        
        if self.env_config.is_multi_agent:
            return obs # [N_agents, State_dim]
        else:
            return obs[0] # [State_dim]

    def _get_tl_observation(self, tl_id: str) -> np.ndarray:
        """Build observation for a single traffic light."""
        conn = self._conn
        state = []
        edges = self.env_config.edges # For simplicity, assumes same edges mapping or handles it
        
        # Determine edges for this TL (If multi-agent, we might need a better mapping)
        # For now, we use a simple heuristic: if tl_id matches an edge prefix, use it.
        # But to be safe, we'll use the ones from config.
        
        for direction in self.env_config.directions:
            edge = edges.get(direction, f"{direction.lower()}_in")

            # Queue length
            queue = 0
            halted = 0
            try:
                for i in range(self.env_config.num_lanes_per_direction):
                    lane_id = f"{edge}_{i}"
                    queue += conn.lane.getLastStepHaltingNumber(lane_id)
                    halted += conn.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                pass
            state.append(queue / 25.0)  # Normalize

            # Average speed
            try:
                speed = conn.edge.getLastStepMeanSpeed(edge)
            except Exception:
                speed = 0
            state.append(speed / 15.0)

            # Waiting time
            try:
                wait = conn.edge.getWaitingTime(edge)
            except Exception:
                wait = 0
            state.append(min(wait, 500) / 500.0)

            # Density
            try:
                n_vehicles = conn.edge.getLastStepVehicleNumber(edge)
                edge_length = conn.lane.getLength(f"{edge}_0")
                density = n_vehicles / max(edge_length, 1.0)
            except Exception:
                density = 0
            state.append(min(density, 1.0))

            # Halted vehicles
            state.append(halted / 20.0)

        # Global features: phase one-hot
        phase_onehot = [0.0] * 4
        current_phase = self._tl_states[tl_id]["phase"]
        phase_idx = min(current_phase // 2 if current_phase % 2 == 0 else
                        current_phase // 2, 3)
        phase_onehot[phase_idx] = 1.0
        state.extend(phase_onehot)

        # Time in phase (normalized)
        max_phase_steps = self._max_green_steps
        state.append(min(self._tl_states[tl_id]["steps"] / max(max_phase_steps, 1), 1.0))

        # Time of day (0-1)
        try:
            sim_time = conn.simulation.getTime()
            time_of_day = (sim_time % 86400) / 86400.0
        except Exception:
            time_of_day = 0.5
        state.append(time_of_day)

        obs = np.array(state, dtype=np.float32)

        # Domain Randomization: Add sensor noise
        if self.env_config.use_domain_randomization:
            noise = self.np_random.normal(0, self.env_config.noise_level, size=obs.shape)
            obs = np.clip(obs + noise.astype(np.float32), -5.0, 5.0)

        # Adaptive normalization
        if self._normalize_obs:
            self._obs_rms.update(obs)
            obs = self._obs_rms.normalize(obs).astype(np.float32)

        return obs

    def _get_traffic_data(self) -> Dict[str, Any]:
        """Extract comprehensive traffic metrics from SUMO."""
        conn = self._conn
        edges = self.env_config.edges
        queues = {}
        waits = {}
        speeds = {}
        halted = {}

        for direction in self.env_config.directions:
            edge = edges.get(direction, f"{direction.lower()}_in")

            try:
                q = 0
                h = 0
                for i in range(self.env_config.num_lanes_per_direction):
                    lane_id = f"{edge}_{i}"
                    q += conn.lane.getLastStepHaltingNumber(lane_id)
                    h += conn.lane.getLastStepHaltingNumber(lane_id)
                queues[direction] = q
                halted[direction] = h
            except Exception:
                queues[direction] = 0
                halted[direction] = 0

            try:
                waits[direction] = conn.edge.getWaitingTime(edge)
            except Exception:
                waits[direction] = 0

            try:
                speeds[direction] = conn.edge.getLastStepMeanSpeed(edge)
            except Exception:
                speeds[direction] = 0

        # Throughput
        try:
            arrived = conn.simulation.getArrivedNumber()
        except Exception:
            arrived = 0
        throughput = arrived  # Vehicles arrived this step

        # Emergency vehicle detection
        emergency = False
        try:
            for vid in conn.vehicle.getIDList():
                vtype = conn.vehicle.getTypeID(vid)
                if "emergency" in vtype.lower():
                    speed = conn.vehicle.getSpeed(vid)
                    if speed < 0.5:  # Emergency vehicle is stopped
                        emergency = True
                        break
        except Exception:
            pass

        return {
            "queues": queues,
            "waits": waits,
            "speeds": speeds,
            "halted": halted,
            "throughput": throughput,
            "emergency": emergency,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        try:
            n_vehicles = len(self._conn.vehicle.getIDList())
        except Exception:
            n_vehicles = 0

        return {
            "step": self._step_count,
            "phase": "N-S" if self._current_phase == 0 else "E-W",
            "vehicles_in_network": n_vehicles,
            "episode": self._episode_number,
            "flow_scale": self._flow_scale,
        }

    def close(self):
        """Clean up environment."""
        self._disconnect()

    def render(self):
        """Rendering is handled by SUMO GUI when render_mode='sumo_gui'."""
        pass


# =============================================================================
# MOCK ENVIRONMENT (for testing without SUMO)
# =============================================================================

class MockTrafficEnv(gym.Env):
    """
    Mock traffic environment for testing without SUMO.

    Generates realistic synthetic traffic data for unit testing,
    CI/CD pipelines, and agent development.
    """

    metadata = {"render_modes": []}

    def __init__(self, state_dim: int = 26, action_dim: int = 4, max_steps: int = 500):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)

        self._step_count = 0
        self._queues = {"N": 0, "S": 0, "E": 0, "W": 0}
        self._current_phase = 0
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._queues = {d: self._rng.integers(0, 10) for d in ["N", "S", "E", "W"]}
        self._current_phase = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        self._step_count += 1

        # Simulate traffic dynamics
        for d in self._queues:
            # Arrivals
            self._queues[d] += self._rng.poisson(2)
            # Departures (more for green direction)
            green_bonus = 3 if (
                (action in [1, 3] and d in ["N", "S"]) or
                (action == 2 and d in ["E", "W"])
            ) else 1
            departures = min(self._queues[d], self._rng.poisson(green_bonus))
            self._queues[d] -= departures

        # Reward: negative total queue
        total_queue = sum(self._queues.values())
        reward = -total_queue * 0.1

        obs = self._get_obs()
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "step": self._step_count,
            "queues": self._queues.copy(),
            "total_queue": total_queue,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        state = []
        for d in ["N", "S", "E", "W"]:
            q = self._queues[d]
            state.extend([
                q / 25.0,                      # queue_norm
                self._rng.uniform(0, 1),        # speed_norm
                q * 2.0 / 500.0,               # wait_norm
                q / 50.0,                       # density
                max(0, q - 2) / 20.0,          # halted_norm
            ])
        # Global features
        phase_onehot = [0.0] * 4
        phase_onehot[self._current_phase % 4] = 1.0
        state.extend(phase_onehot)
        state.append(self._step_count / self.max_steps)  # time_in_phase
        state.append(0.5)                                 # time_of_day
        return np.array(state, dtype=np.float32)
