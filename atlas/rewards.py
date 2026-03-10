"""
ATLAS Pro — Multi-Objective Reward Functions
==============================================
Advanced reward shaping for traffic signal optimization.

Components:
    - Queue penalty (minimize vehicles waiting)
    - Wait time penalty (minimize accumulated waiting)
    - Throughput reward (maximize vehicles served)
    - Fairness index (balance across directions, Jain's index)
    - Emissions proxy (minimize stop-and-go)
    - Phase change penalty (avoid excessive switching)
    - Emergency vehicle priority
    - Potential-based reward shaping
"""

import numpy as np
from typing import Dict, Optional
from atlas.config import RewardConfig


class MultiObjectiveReward:
    """
    Multi-objective reward function for traffic signal control.
    
    Combines multiple objectives with configurable weights to guide
    the RL agent toward optimal traffic management.
    
    The reward is computed as:
        R = w_q * queue + w_w * wait + w_t * throughput + w_s * speed
          + w_f * fairness + w_e * emissions + w_p * phase_penalty
          + w_emg * emergency
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self._prev_total_wait = 0.0
        self._prev_total_queue = 0.0
        self._prev_throughput = 0
        self._step_count = 0
        
        # Running statistics for normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
    
    def compute(
        self,
        queue_lengths: Dict[str, float],
        wait_times: Dict[str, float],
        speeds: Dict[str, float],
        throughput: int,
        phase_changed: bool,
        emergency_waiting: bool = False,
        halted_vehicles: Dict[str, int] = None,
    ) -> float:
        """
        Compute the multi-objective reward.
        
        Args:
            queue_lengths: Queue length per direction {"N": 5, "S": 3, ...}
            wait_times: Accumulated wait time per direction (seconds)
            speeds: Average speed per direction (m/s)
            throughput: Number of vehicles that arrived at destination this step
            phase_changed: Whether the phase was changed this decision step
            emergency_waiting: Whether an emergency vehicle is waiting
            halted_vehicles: Number of halted vehicles per direction
            
        Returns:
            Scalar reward value
        """
        c = self.config
        reward = 0.0
        
        # --- 1. Queue Length Penalty ---
        total_queue = sum(queue_lengths.values())
        queue_component = c.queue_length_weight * total_queue
        reward += queue_component
        
        # --- 2. Wait Time Penalty (differential) ---
        total_wait = sum(wait_times.values())
        wait_delta = total_wait - self._prev_total_wait
        wait_component = c.wait_time_weight * max(wait_delta, 0)
        reward += wait_component
        self._prev_total_wait = total_wait
        
        # --- 3. Throughput Reward ---
        throughput_component = c.throughput_weight * throughput
        reward += throughput_component
        
        # --- 4. Speed Reward ---
        avg_speed = np.mean(list(speeds.values())) if speeds else 0
        speed_component = c.speed_weight * avg_speed
        reward += speed_component
        
        # --- 5. Fairness (Jain's Fairness Index) ---
        if len(wait_times) > 1:
            waits = np.array(list(wait_times.values()))
            waits = waits + 1e-6  # Avoid division by zero
            jain_index = (waits.sum() ** 2) / (len(waits) * (waits ** 2).sum())
            # Jain index is 1.0 when perfectly fair, lower when unfair
            fairness_penalty = c.fairness_weight * (1.0 - jain_index)
            reward += fairness_penalty
        
        # --- 6. Emissions Proxy ---
        # Halted vehicles produce more emissions due to acceleration cycles
        if halted_vehicles:
            total_halted = sum(halted_vehicles.values())
            emissions_component = c.emissions_weight * total_halted
            reward += emissions_component
        
        # --- 7. Phase Change Penalty ---
        if phase_changed:
            reward += c.phase_change_penalty
        
        # --- 8. Emergency Vehicle Priority ---
        if emergency_waiting:
            reward += c.emergency_weight * (-1.0)  # Large penalty for keeping emergency waiting
        
        # --- Clip reward ---
        reward = np.clip(reward, c.clip_min, c.clip_max)
        
        # --- Normalize if enabled ---
        if c.normalize:
            reward = self._normalize_reward(reward)
        
        self._step_count += 1
        self._prev_throughput = throughput
        self._prev_total_queue = total_queue
        
        return float(reward)
    
    def _normalize_reward(self, reward: float) -> float:
        """Running normalization of rewards."""
        self._reward_count += 1
        old_mean = self._reward_mean
        self._reward_mean += (reward - old_mean) / self._reward_count
        self._reward_var += (reward - old_mean) * (reward - self._reward_mean)
        
        std = np.sqrt(self._reward_var / max(self._reward_count, 1)) + 1e-8
        return (reward - self._reward_mean) / std
    
    def reset(self):
        """Reset per-episode state."""
        self._prev_total_wait = 0.0
        self._prev_total_queue = 0.0
        self._prev_throughput = 0
        self._step_count = 0
    
    def get_component_breakdown(
        self,
        queue_lengths: Dict[str, float],
        wait_times: Dict[str, float],
        speeds: Dict[str, float],
        throughput: int,
        phase_changed: bool,
    ) -> Dict[str, float]:
        """Return individual reward components for logging/debugging."""
        c = self.config
        total_queue = sum(queue_lengths.values())
        total_wait = sum(wait_times.values())
        avg_speed = np.mean(list(speeds.values())) if speeds else 0
        
        breakdown = {
            "queue_penalty": c.queue_length_weight * total_queue,
            "wait_penalty": c.wait_time_weight * total_wait,
            "throughput_reward": c.throughput_weight * throughput,
            "speed_reward": c.speed_weight * avg_speed,
            "phase_penalty": c.phase_change_penalty if phase_changed else 0.0,
        }
        
        if len(wait_times) > 1:
            waits = np.array(list(wait_times.values())) + 1e-6
            jain = (waits.sum() ** 2) / (len(waits) * (waits ** 2).sum())
            breakdown["fairness_penalty"] = c.fairness_weight * (1.0 - jain)
        
        return breakdown


class PotentialBasedRewardShaping:
    """
    Potential-based reward shaping (Ng et al., 1999).
    
    Adds a shaping reward F(s, s') = γ * Φ(s') - Φ(s) that preserves
    the optimal policy while guiding exploration.
    
    The potential function is based on negative total queue length,
    encouraging the agent to reduce queues quickly.
    """
    
    def __init__(self, gamma: float = 0.99, scale: float = 0.1):
        self.gamma = gamma
        self.scale = scale
        self._prev_potential = 0.0
    
    def potential(self, queue_lengths: Dict[str, float]) -> float:
        """
        Compute state potential.
        Lower queue = higher potential = better state.
        """
        total_queue = sum(queue_lengths.values())
        return -total_queue * self.scale
    
    def compute_shaping(self, queue_lengths: Dict[str, float]) -> float:
        """Compute the shaping reward: γ * Φ(s') - Φ(s)."""
        current_potential = self.potential(queue_lengths)
        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential
        return shaping
    
    def reset(self):
        self._prev_potential = 0.0
