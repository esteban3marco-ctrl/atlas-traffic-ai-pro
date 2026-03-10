"""
ATLAS Pro — Autonomous Traffic Light Adaptive System
=====================================================
Commercial-grade AI traffic signal control using Deep Reinforcement Learning.

Features:
    - Dueling Double DQN with Prioritized Experience Replay
    - PPO (Proximal Policy Optimization) agent
    - Gymnasium-compatible SUMO environment
    - Multi-objective reward shaping
    - Real-time dashboard & REST API
    - Multi-intersection coordination (MARL)

Author: Esteban Marco Muñoz
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Esteban Marco Muñoz"
__email__ = "estebanmarcojobs@gmail.com"

from atlas.config import TrainingConfig, EnvironmentConfig, AgentConfig, RewardConfig
