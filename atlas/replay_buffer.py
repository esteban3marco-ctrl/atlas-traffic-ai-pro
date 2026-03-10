"""
ATLAS Pro — Prioritized Experience Replay Buffer
==================================================
Advanced replay buffers for DQN training.

Components:
    - SumTree: Efficient prefix-sum tree for O(log n) sampling
    - PrioritizedReplayBuffer: PER (Schaul et al., 2016)
    - NStepBuffer: Multi-step return accumulation
"""

import numpy as np
from typing import Tuple, List, NamedTuple, Optional
import torch


class Transition(NamedTuple):
    """A single transition tuple."""
    state: np.ndarray
    action: any
    reward: float
    next_state: np.ndarray
    done: bool


# =============================================================================
# SUM TREE (for efficient priority sampling)
# =============================================================================

class SumTree:
    """
    Binary Sum Tree for O(log n) priority-based sampling.
    
    Each leaf stores a priority. Parent nodes store the sum of children.
    Sampling is done by drawing a uniform random value in [0, total_priority]
    and traversing the tree to find the corresponding leaf.
    
    Tree structure:
              [sum=42]
             /        \\
        [24]            [18]
       /    \\          /    \\
     [14]  [10]      [8]   [10]
      ↑     ↑        ↑      ↑
    leaf0 leaf1    leaf2   leaf3
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    @property
    def total_priority(self) -> float:
        return self.tree[0]
    
    @property
    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        return max(self.tree[leaf_start:leaf_start + self.size]) if self.size > 0 else 1.0
    
    @property
    def min_priority(self) -> float:
        leaf_start = self.capacity - 1
        if self.size == 0:
            return 1.0
        priorities = self.tree[leaf_start:leaf_start + self.size]
        nonzero = priorities[priorities > 0]
        return nonzero.min() if len(nonzero) > 0 else 1.0
    
    def add(self, priority: float, data):
        """Add data with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update the priority of a leaf."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, any]:
        """Sample a leaf based on cumulative priority sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY BUFFER
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016).
    
    Transitions with higher TD-error get sampled more frequently,
    leading to more efficient learning.
    
    Features:
        - SumTree-based O(log n) sampling
        - Importance sampling weights for unbiased updates
        - Priority exponent (alpha) controls prioritization
        - IS exponent (beta) annealed from beta_start to 1.0
    
    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0=uniform, 1=full priority)
        beta_start: Initial importance sampling exponent
        beta_frames: Frames to anneal beta to 1.0
    """
    
    def __init__(
        self,
        capacity: int = 200_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self._max_priority = 1.0
    
    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        fraction = min(self.frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)
    
    def __len__(self) -> int:
        return self.tree.size
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition with maximum priority."""
        transition = Transition(state, action, reward, next_state, done)
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int, device: torch.device = None) -> Tuple:
        """
        Sample a batch of transitions with priority-based sampling.
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        if device is None:
            device = torch.device('cpu')
        
        batch_size = min(batch_size, len(self))
        
        indices = []
        priorities = []
        transitions = []
        
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            
            if data is None:
                # Fallback: random sample
                s = np.random.uniform(0, self.tree.total_priority)
                idx, priority, data = self.tree.get(s)
            
            if data is not None:
                indices.append(idx)
                priorities.append(priority)
                transitions.append(data)
        
        if len(transitions) == 0:
            return None
        
        # Compute importance-sampling weights
        total = self.tree.total_priority
        min_prob = self.tree.min_priority / total if total > 0 else 1e-6
        min_prob = max(min_prob, 1e-8)
        
        weights = []
        for p in priorities:
            prob = p / total if total > 0 else 1.0
            prob = max(prob, 1e-8)
            weight = (prob * len(self)) ** (-self.beta)
            weights.append(weight)
        
        max_weight = (min_prob * len(self)) ** (-self.beta)
        max_weight = max(max_weight, 1e-8)
        weights = np.array(weights) / max_weight
        
        # Unpack transitions to tensors
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(device)
        actions = torch.LongTensor([t.action for t in transitions]).to(device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(device)
        dones = torch.FloatTensor([float(t.done) for t in transitions]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on new TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)


# =============================================================================
# N-STEP RETURN BUFFER
# =============================================================================

class NStepBuffer:
    """
    Accumulates N-step transitions for computing multi-step returns.
    
    G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ... + γ^{n-1} * r_{t+n-1} + γ^n * V(s_{t+n})
    
    Multi-step returns reduce bias at the cost of increased variance,
    leading to faster and more stable learning.
    """
    
    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: List[Transition] = []
    
    def add(self, state, action, reward, next_state, done) -> Optional[Transition]:
        """
        Add transition from the latest step.
        
        Returns the accumulated N-step transition when enough steps
        have been collected, or the partial transition on episode end.
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))
        
        if done:
            # Flush all remaining transitions
            result = self._compute_nstep()
            self.buffer.clear()
            return result
        
        if len(self.buffer) >= self.n_step:
            result = self._compute_nstep()
            self.buffer.pop(0)
            return result
        
        return None
    
    def _compute_nstep(self) -> Transition:
        """Compute N-step discounted return."""
        n_step_return = 0.0
        for i in range(len(self.buffer)):
            n_step_return += (self.gamma ** i) * self.buffer[i].reward
        
        return Transition(
            state=self.buffer[0].state,
            action=self.buffer[0].action,
            reward=n_step_return,
            next_state=self.buffer[-1].next_state,
            done=self.buffer[-1].done,
        )
    
    def flush(self) -> List[Transition]:
        """Flush all remaining transitions from buffer."""
        results = []
        while len(self.buffer) > 0:
            results.append(self._compute_nstep())
            self.buffer.pop(0)
        return results
