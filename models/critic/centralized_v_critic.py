from itertools import chain
from torch.nn.utils.parametrizations import spectral_norm

import torch.nn as nn
import torch
from models.utils import orthogonal_init


class CentralizedCriticV(nn.Module):
    """
    Centralized Critic (arch v.2)
    """

    def __init__(self,
                 state_dim,
                 n_agents=2,
                 hidden_dim=64,
                 social_planner=False,
                 ):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(n_agents * state_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, n_agents), std=1.),
        )
        self.social_planner = social_planner

    def __call__(self, state):
        """
        :param state:      [batch_size, n_agents, features]
        :return: q-values: [batch_size, n_agents]
        """
        if len(state) == 2:
            state = state.unsqueeze(0)
        batch_size, n_agents, features = state.shape
        state_flatten = state.reshape(batch_size, n_agents * features)
        out = self.net(state_flatten)
        if self.social_planner:
            out = out.mean(dim=-1, keepdim=True).repeat(1, out.shape[-1])
        return out
