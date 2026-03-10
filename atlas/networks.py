"""
ATLAS Pro — Neural Network Architectures
==========================================
State-of-the-art network architectures for Deep RL traffic signal control.

Architectures:
    - NoisyLinear: Noisy Networks for exploration without epsilon-greedy
    - DuelingNetwork: Dueling DQN architecture (value + advantage streams)
    - ActorCriticNetwork: Shared backbone with separate actor/critic heads for PPO
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


# =============================================================================
# NOISY LINEAR LAYER
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer (Fortunato et al., 2018).
    
    Replaces epsilon-greedy exploration with learned parametric noise.
    The network learns when and how much to explore.
    
    y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise buffers (not parameters)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters using factorized initialization."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorized Gaussian noise: f(x) = sign(x) * sqrt(|x|)."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Sample new noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
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
# DUELING DQN NETWORK
# =============================================================================

# =============================================================================
# ATTENTION & TRANSFORMER MODULES (XAI READY)
# =============================================================================

class AttentionBlock(nn.Module):
    """
    Multi-Head Attention block to weight the 26 state features dynamically.
    Provides intrinsic explainability (XAI) via attention maps.
    """
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, Seq, Dim]
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x, attn_weights


class WorldModel(nn.Module):
    """
    Recurrent State-Space Model (RSSM-lite) for Latent Planning.
    Learns to predict: next_latent_state, reward, and terminal from current state/action.
    """
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 128):
        super().__init__()
        # Encoder: Observe pixels/vectors -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # Dynamics: (latent, action) -> next_latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # Reward Predictor
        self.reward_pred = nn.Linear(latent_dim, 1)
        
    def forward(self, state, action_onehot):
        latent = self.encoder(state)
        combined = torch.cat([latent, action_onehot], dim=-1)
        next_latent = self.dynamics(combined)
        pred_reward = self.reward_pred(next_latent)
        return next_latent, pred_reward


# =============================================================================
# ATLAS PRO V2: TRANSFORMER DUELING NETWORK
# =============================================================================

class DuelingNetwork(nn.Module):
    """
    ATLAS Pro V2 Architecture:
    Transformer Attention + Distributional + Dueling + World Model Integration.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        use_noisy: bool = True,
        sigma_init: float = 0.5,
        use_dueling: bool = True,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_transformer: bool = True,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 512, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy
        self.use_dueling = use_dueling
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_transformer = use_transformer
        
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        
        Linear = NoisyLinear if use_noisy else nn.Linear
        linear_kwargs = {"sigma_init": sigma_init} if use_noisy else {}
        
        # === 1. Transformer/Attention Feature Extractor ===
        if use_transformer:
            self.embedding = nn.Linear(1, 16) # Embed each feature dim
            self.transformer = AttentionBlock(dim=16, heads=4)
            feature_dim = state_dim * 16
        else:
            feature_dim = state_dim

        # === 2. Shared Backbone ===
        shared_layers = []
        in_dim = feature_dim
        for i, h_dim in enumerate(hidden_dims[:-1]):
            shared_layers.append(Linear(in_dim, h_dim, **linear_kwargs))
            shared_layers.append(nn.LayerNorm(h_dim))
            shared_layers.append(nn.GELU())
            in_dim = h_dim
        
        self.shared = nn.Sequential(*shared_layers)
        backbone_dim = hidden_dims[-2]
        stream_hidden = hidden_dims[-1]
        
        if use_dueling:
            # === Value Stream V(s) ===
            self.value_stream = nn.Sequential(
                Linear(backbone_dim, stream_hidden, **linear_kwargs),
                nn.LayerNorm(stream_hidden),
                nn.GELU(),
                Linear(stream_hidden, num_atoms, **linear_kwargs),
            )
            # === Advantage Stream A(s, a) ===
            self.advantage_stream = nn.Sequential(
                Linear(backbone_dim, stream_hidden, **linear_kwargs),
                nn.LayerNorm(stream_hidden),
                nn.GELU(),
                Linear(stream_hidden, action_dim * num_atoms, **linear_kwargs),
            )
        else:
            self.q_head = nn.Sequential(
                Linear(backbone_dim, stream_hidden, **linear_kwargs),
                nn.LayerNorm(stream_hidden),
                nn.GELU(),
                Linear(stream_hidden, action_dim * num_atoms, **linear_kwargs),
            )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional Attention Saliency (XAI).
        """
        attn_weights = None
        if self.use_transformer:
            # state: [B, 26] -> [B, 26, 1]
            x = state.unsqueeze(-1)
            # embedding: [B, 26, 16]
            x = self.embedding(x)
            # transformer: [B, 26, 16]
            x, attn_weights = self.transformer(x)
            # flatten: [B, 26*16]
            features = x.reshape(state.size(0), -1)
        else:
            features = state
            
        features = self.shared(features)
        
        if self.use_dueling:
            value = self.value_stream(features).view(-1, 1, self.num_atoms)
            advantage = self.advantage_stream(features).view(-1, self.action_dim, self.num_atoms)
            dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            dist = self.q_head(features).view(-1, self.action_dim, self.num_atoms)
        
        prob_dist = F.softmax(dist, dim=-1)
        
        if return_attn:
            return prob_dist, attn_weights
        return prob_dist

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Σ (prob * support)"""
        dist = self.forward(state)
        q_values = (dist * self.support).sum(dim=2)
        return q_values

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if hasattr(module, "reset_noise") and module is not self:
                module.reset_noise()
    
# =============================================================================
# QMIX: MONOTONIC VALUE MIXER (MARL COORDINATION)
# =============================================================================

class QMixer(nn.Module):
    """
    QMIX Mixer network (Rashid et al., 2018).
    Mixes individual agent Q-values into a joint Q-tot using hypernetworks
    to ensure the monotonicity constraint: d(Qtot)/d(Qi) >= 0.
    """
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetworks for weights
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(embed_dim, n_agents * embed_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(embed_dim, embed_dim))
        
        # Biases
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # agent_qs: [Batch, n_agents]
        # states: [Batch, state_dim]
        bs = agent_qs.size(0)
        
        # W1: [Batch, n_agents, embed_dim]
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        
        # Hidden layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # W2: [Batch, embed_dim, 1]
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        
        # Output Q-tot
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, -1)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()


# =============================================================================
# ACTOR-CRITIC NETWORK (FOR PPO)
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    
    Shared feature extractor with separate actor (policy) and critic (value) heads.
    
    Architecture:
        Input [state_dim]
            ↓
        Shared: [512] → [256] (with LayerNorm + GELU)
            ↓                    ↓
        Actor Head              Critic Head
        [128] → [action_dim]   [128] → [1]
        (softmax)              (value)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # === Shared Feature Extractor ===
        shared_layers = []
        in_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        
        self.shared = nn.Sequential(*shared_layers)
        feature_dim = hidden_dims[-2] if len(hidden_dims) > 1 else hidden_dims[0]
        head_dim = hidden_dims[-1]
        
        # === Actor Head (Policy) ===
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Linear(head_dim, action_dim),
        )
        
        # === Critic Head (Value) ===
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        
        # Initialize
        self.apply(self._init_weights)
        # Smaller init for policy head to encourage exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [batch_size, state_dim]
            
        Returns:
            action_logits: [batch_size, action_dim]
            value: [batch_size, 1]
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Used during both rollout collection and PPO update.
        """
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
