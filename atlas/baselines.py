"""
ATLAS Pro — Baseline Policies for Comparison
===============================================
Non-RL traffic signal control strategies for benchmarking.

Baselines:
    - FixedTimePolicy: Standard fixed-cycle timing
    - ActuatedPolicy: Vehicle-detection based extension
    - WebsterPolicy: Optimal cycle via Webster's formula
    - MaxPressurePolicy: Pressure-based greedy control (state-of-art non-RL)
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("ATLAS.Baseline")


class BasePolicy:
    """Base class for traffic control policies."""
    
    name: str = "base"
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        raise NotImplementedError
    
    def reset(self):
        pass


class FixedTimePolicy(BasePolicy):
    """
    Fixed-time traffic signal control.
    
    Alternates between N-S and E-W green phases with fixed durations.
    This is the simplest baseline representing traditional traffic lights.
    """
    
    name = "fixed_time"
    
    def __init__(self, green_ns: int = 30, green_ew: int = 25, decision_interval: int = 5):
        self.green_ns = green_ns // decision_interval  # In decision steps
        self.green_ew = green_ew // decision_interval
        self.step_count = 0
        self.current_phase = 0  # 0=NS, 1=EW
        self.steps_in_phase = 0
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        self.step_count += 1
        self.steps_in_phase += 1
        
        if self.current_phase == 0:  # Currently NS
            if self.steps_in_phase >= self.green_ns:
                self.current_phase = 1
                self.steps_in_phase = 0
                return 2  # Switch to EW
            return 0  # Maintain
        else:  # Currently EW
            if self.steps_in_phase >= self.green_ew:
                self.current_phase = 0
                self.steps_in_phase = 0
                return 1  # Switch to NS
            return 0  # Maintain
    
    def reset(self):
        self.step_count = 0
        self.current_phase = 0
        self.steps_in_phase = 0


class ActuatedPolicy(BasePolicy):
    """
    Actuated traffic signal control.
    
    Extends the current phase if vehicles are still arriving,
    switches when a gap is detected or max green is reached.
    State indices: queue_N(0), speed_N(1), wait_N(2), ...
    """
    
    name = "actuated"
    
    def __init__(self, min_green: int = 10, max_green: int = 50,
                 extension: int = 3, gap_threshold: float = 0.1,
                 decision_interval: int = 5):
        self.min_green = min_green // decision_interval
        self.max_green = max_green // decision_interval
        self.extension = extension // decision_interval
        self.gap_threshold = gap_threshold
        self.steps_in_phase = 0
        self.current_phase = 0
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        self.steps_in_phase += 1
        
        # Get queue for current green direction
        if self.current_phase == 0:  # NS green
            current_queue = state[0] + state[5]   # N + S queues
            other_queue = state[10] + state[15]    # E + W queues
        else:  # EW green
            current_queue = state[10] + state[15]
            other_queue = state[0] + state[5]
        
        # Must serve minimum green
        if self.steps_in_phase < self.min_green:
            return 0  # Maintain
        
        # Max green reached — must switch
        if self.steps_in_phase >= self.max_green:
            self._switch()
            return 2 if self.current_phase == 1 else 1
        
        # Gap detection: switch if little traffic on current green
        if current_queue < self.gap_threshold and other_queue > self.gap_threshold:
            self._switch()
            return 2 if self.current_phase == 1 else 1
        
        return 3  # Extend
    
    def _switch(self):
        self.current_phase = 1 - self.current_phase
        self.steps_in_phase = 0
    
    def reset(self):
        self.steps_in_phase = 0
        self.current_phase = 0


class WebsterPolicy(BasePolicy):
    """
    Webster's optimal cycle timing.
    
    Computes the optimal cycle length and green splits based on
    traffic flow ratios. Classic traffic engineering approach.
    
    C_opt = (1.5L + 5) / (1 - Y)
    where L = total lost time, Y = sum of critical flow ratios
    """
    
    name = "webster"
    
    def __init__(self, saturation_flow: float = 1800,
                 lost_time: float = 8.0, decision_interval: int = 5):
        self.saturation_flow = saturation_flow  # veh/h/lane
        self.lost_time = lost_time
        self.decision_interval = decision_interval
        self.step_count = 0
        self.current_phase = 0
        self.cycle_ns = 6  # Default steps for NS
        self.cycle_ew = 5  # Default steps for EW
        self.steps_in_phase = 0
        self._recalc_counter = 0
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        self.step_count += 1
        self.steps_in_phase += 1
        self._recalc_counter += 1
        
        # Recalculate splits periodically
        if self._recalc_counter >= 20:
            self._recalculate_splits(state)
            self._recalc_counter = 0
        
        if self.current_phase == 0:  # NS
            if self.steps_in_phase >= self.cycle_ns:
                self.current_phase = 1
                self.steps_in_phase = 0
                return 2  # Switch to EW
        else:
            if self.steps_in_phase >= self.cycle_ew:
                self.current_phase = 0
                self.steps_in_phase = 0
                return 1  # Switch to NS
        
        return 0  # Maintain
    
    def _recalculate_splits(self, state: np.ndarray):
        """Recalculate Webster cycle based on current demand."""
        # Estimate flow from queue lengths (proxy)
        ns_demand = (state[0] + state[5]) * 20  # De-normalize
        ew_demand = (state[10] + state[15]) * 20
        
        # Flow ratios
        y_ns = ns_demand / max(self.saturation_flow / 3600, 0.01)
        y_ew = ew_demand / max(self.saturation_flow / 3600, 0.01)
        Y = min(y_ns + y_ew, 0.9)  # Cap to avoid division issues
        
        # Webster formula
        C_opt = (1.5 * self.lost_time + 5) / max(1 - Y, 0.1)
        C_opt = np.clip(C_opt, 30, 120)  # Reasonable range
        
        effective = C_opt - self.lost_time
        if Y > 0.01:
            self.cycle_ns = max(2, int((y_ns / Y * effective) / self.decision_interval))
            self.cycle_ew = max(2, int((y_ew / Y * effective) / self.decision_interval))
        else:
            self.cycle_ns = 6
            self.cycle_ew = 5
    
    def reset(self):
        self.step_count = 0
        self.current_phase = 0
        self.steps_in_phase = 0
        self._recalc_counter = 0


class MaxPressurePolicy(BasePolicy):
    """
    Max Pressure control (Varaiya, 2013).
    
    State-of-the-art non-RL approach. Selects the phase that serves
    the direction with the maximum "pressure" (queue difference
    between incoming and outgoing links).
    
    This is the strongest non-RL baseline for traffic signal control.
    """
    
    name = "max_pressure"
    
    def __init__(self, min_green: int = 10, decision_interval: int = 5):
        self.min_green = min_green // decision_interval
        self.steps_in_phase = 0
        self.current_phase = 0
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        self.steps_in_phase += 1
        
        if self.steps_in_phase < self.min_green:
            return 0
        
        # Compute pressure for each phase
        # State: queue_N(0), speed_N(1), wait_N(2), density_N(3), halted_N(4),
        #        queue_S(5), ..., queue_E(10), ..., queue_W(15), ...
        ns_pressure = state[0] + state[5]   # N + S queue pressure
        ew_pressure = state[10] + state[15]  # E + W queue pressure
        
        # Factor in waiting time for urgency
        ns_urgency = state[2] + state[7]     # N + S wait times
        ew_urgency = state[12] + state[17]   # E + W wait times
        
        ns_total = ns_pressure + 0.5 * ns_urgency
        ew_total = ew_pressure + 0.5 * ew_urgency
        
        # Select phase with maximum pressure
        if self.current_phase == 0:  # Currently NS
            if ew_total > ns_total * 1.1:  # 10% hysteresis
                self.current_phase = 1
                self.steps_in_phase = 0
                return 2  # Switch to EW
        else:  # Currently EW
            if ns_total > ew_total * 1.1:
                self.current_phase = 0
                self.steps_in_phase = 0
                return 1  # Switch to NS
        
        return 0  # Maintain
    
    def reset(self):
        self.steps_in_phase = 0
        self.current_phase = 0


# =============================================================================
# BASELINE REGISTRY
# =============================================================================

BASELINES = {
    "fixed_time": FixedTimePolicy,
    "actuated": ActuatedPolicy,
    "webster": WebsterPolicy,
    "max_pressure": MaxPressurePolicy,
}

def get_baseline(name: str, **kwargs) -> BasePolicy:
    """Get a baseline policy by name."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES.keys())}")
    return BASELINES[name](**kwargs)
