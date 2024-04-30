from itertools import chain
from torch.nn.utils.parametrizations import spectral_norm

import torch.nn as nn
import torch


def layer_init(layer, std=1.4142135623730951, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CentralizedCriticV(nn.Module):
    """
    Centralized Critic (arch v.2)
    """

    def __init__(self,
                 state_dim,
                 n_agents=2,
                 hidden_dim=64):
        super().__init__()
        self.output_layer = nn.Linear(hidden_dim, n_agents)
        self.net = nn.Sequential(
            layer_init(nn.Linear(n_agents * state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            self.output_layer,
        )
        self.init_weights()

    def init_weights(self):
        """
        According to paper: https://arxiv.org/pdf/2006.05990
        :return:
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2 ** 0.5)
                torch.nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.output_layer.weight, 1)

    def __call__(self, state):
        """
        :param state:      [batch_size, n_agents, features]
        :return: q-values: [batch_size, n_agents]
        """
        if len(state) == 2:
            state = state.unsqueeze(0)
        batch_size, n_agents, features = state.shape
        state_flatten = state.reshape(batch_size, n_agents * features)
        return self.net(state_flatten)