from itertools import chain
from torch.nn.utils.parametrizations import spectral_norm

import torch.nn as nn
import torch


class CentralizedCritic(nn.Module):
    """
    Centralized Critic (arch v.1)
    """

    def __init__(self, state_dim, action_dim=None, n_agents=2, hidden_dim=32):
        super().__init__()

        input_dim = n_agents * (state_dim + action_dim)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, n_agents)),
        )

    def __call__(self, state, action=None):
        """
        :param state:      [batch_size, n_agents, features]
        :param action:     [batch_size, n_agents, actions_dim]
        :return: q-values: [batch_size, n_agents]
        """
        if len(state) == 2 and len(action) == 2:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        batch_size, n_agents, features = state.shape
        batch_size, n_agents, actions_dim = action.shape
        state_flatten = state.reshape(batch_size, n_agents * features)
        actions_flatten = action.reshape(batch_size, n_agents * actions_dim)
        concatenated = torch.concat((state_flatten, actions_flatten), dim=-1)
        q_values = self.net(concatenated)
        return q_values


class CentralizedCritic2(nn.Module):
    """
    Centralized Critic (arch v.2)
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 n_agents=2,
                 hidden_dim=32):
        super().__init__()
        self.net_actions = nn.Sequential(
            nn.Linear(n_agents * action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.net_state = nn.Sequential(
            nn.Linear(n_agents * state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_agents, bias=False),
        )

        # for layer in chain(self.net_actions, self.net_state):
        #     if isinstance(layer, nn.Linear):
        #         nn.init.orthogonal_(layer.weight)
        #         layer.bias.data.fill_(0.0)
        #
        # for module in self.final_net:
        #     if isinstance(module, nn.Linear):
        #         nn.init.uniform_(module.weight, a=-0.1, b=0.1)

    def __call__(self, state, action=None):
        """
        :param state:      [batch_size, n_agents, features]
        :param action:     [batch_size, n_agents, actions_dim]
        :return: q-values: [batch_size, n_agents]
        """
        if len(state) == 2 and len(action) == 2:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        batch_size, n_agents, features = state.shape
        batch_size, n_agents, actions_dim = action.shape

        state_flatten = state.reshape(batch_size, n_agents * features)
        actions_flatten = action.reshape(batch_size, n_agents * actions_dim)
        state_output = self.net_state(state_flatten)
        actions_output = self.net_actions(actions_flatten)
        concatenated = torch.concat((state_output, actions_output), dim=-1)
        q_values = self.final_net(concatenated)
        return q_values


if __name__ == "__main__":
    net = CentralizedCritic(10, 20)
    x = torch.empty((32, 2, 10))
    y = torch.empty((32, 2, 20))
    print(net(x, y).shape)
