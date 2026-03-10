"""
ATLAS Pro — Configuration System
==================================
Centralized configuration using dataclasses with YAML serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import yaml
import os


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    
    # Architecture
    algorithm: str = "dueling_ddqn"  # "dueling_ddqn" | "ppo"
    state_dim: int = 26
    action_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    
    # DQN hyperparameters
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    min_buffer_size: int = 1000
    target_update_freq: int = 500
    tau: float = 0.005              # Soft update coefficient
    n_step: int = 3                 # N-step returns
    per_alpha: float = 0.6          # PER priority exponent
    per_beta_start: float = 0.4     # PER importance sampling start
    per_beta_frames: int = 100_000  # Frames to anneal beta to 1.0
    
    # Exploration (Noisy Networks replace epsilon-greedy)
    use_noisy_nets: bool = True
    noisy_sigma: float = 0.5
    
    # Distributional RL (C51)
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    
    # Advanced Architecture V2
    use_transformer: bool = True
    use_world_model: bool = True
    latent_dim: int = 128
    
    # PPO-specific
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_rollout_length: int = 2048
    
    # MARL (QMIX Coordination)
    use_qmix: bool = False
    n_agents: int = 1
    mixer_embed_dim: int = 32
    
    # Gradient clipping
    max_grad_norm: float = 10.0


@dataclass
class RewardConfig:
    """Configuration for multi-objective reward function."""
    
    # Weights for each component
    wait_time_weight: float = -0.6    # Increased from -0.4
    queue_length_weight: float = -0.3
    throughput_weight: float = 0.5    # Increased from 0.3
    speed_weight: float = 0.2         # Increased from 0.1
    emergency_weight: float = 5.0
    
    # Fairness (Jain's index)
    fairness_weight: float = -0.15
    
    # Emissions proxy
    emissions_weight: float = -0.05
    
    # Phase change penalty (discourage excessive switching)
    phase_change_penalty: float = -0.5
    
    # Reward clipping
    clip_min: float = -20.0
    clip_max: float = 20.0
    
    # Normalization
    normalize: bool = True


@dataclass
class EnvironmentConfig:
    """Configuration for the SUMO environment."""
    
    # SUMO settings
    sumo_cfg_file: str = "simulations/simple/simulation.sumocfg"
    gui: bool = False
    step_length: float = 0.1        # SUMO step length in seconds
    delta_time: int = 5             # Seconds between agent decisions
    is_multi_agent: bool = False    # Enable MARL coordination
    
    # Intersection
    traffic_light_ids: List[str] = field(default_factory=lambda: ["center"])
    map_coordinates: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "center": (40.4168, -3.7038)
    })
    
    # State configuration
    num_lanes_per_direction: int = 2
    directions: List[str] = field(default_factory=lambda: ["N", "S", "E", "W"])
    
    # Phase constraints
    min_green_time: float = 10.0    # Minimum green phase duration (seconds)
    max_green_time: float = 60.0    # Maximum green phase duration (seconds)
    yellow_time: float = 4.0        # Yellow phase duration (seconds)
    all_red_time: float = 2.0       # All-red clearance interval (seconds)
    
    # Edge names for each direction
    edges: Dict[str, str] = field(default_factory=lambda: {
        "N": "north_in", "S": "south_in",
        "E": "east_in", "W": "west_in"
    })
    
    # Domain Randomization (Sim-to-Real robustness)
    use_domain_randomization: bool = True
    noise_level: float = 0.05        # 5% observation noise
    latency_range: Tuple[int, int] = (0, 2)  # Delay in decision steps
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"episode_range": [0, 50], "flow_scale": 0.3, "description": "Low traffic"},
        {"episode_range": [50, 150], "flow_scale": 0.6, "description": "Medium traffic"},
        {"episode_range": [150, 300], "flow_scale": 1.0, "description": "Full traffic"},
        {"episode_range": [300, 500], "flow_scale": 1.3, "description": "Rush hour"},
    ])
    
    # Simulation limits
    max_steps: int = 3600           # Max simulation steps per episode
    warmup_steps: int = 100         # Steps before agent starts acting


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    
    # Training
    total_episodes: int = 500
    eval_interval: int = 20         # Evaluate every N episodes
    eval_episodes: int = 5          # Number of evaluation episodes
    save_interval: int = 25         # Save checkpoint every N episodes
    
    # Logging
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "atlas-traffic-ai"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 50              # Episodes without improvement
    min_improvement: float = 0.01   # Minimum reward improvement
    
    # Seeds
    seed: int = 42
    
    # Device
    device: str = "auto"            # "auto", "cuda", "cpu"
    
    # Sub-configurations
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    def save(self, path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        agent = AgentConfig(**data.pop('agent', {}))
        reward = RewardConfig(**data.pop('reward', {}))
        environment = EnvironmentConfig(**data.pop('environment', {}))
        
        return cls(agent=agent, reward=reward, environment=environment, **data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
