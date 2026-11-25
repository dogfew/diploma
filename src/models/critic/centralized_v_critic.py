import torch
from torch import nn

from src.models.utils import orthogonal_init


class CentralizedCriticV(nn.Module):
    """
    Centralized Critic (arch v.2)
    """

    def __init__(
        self,
        state_dim,
        n_agents=2,
        hidden_dim=64,
        *,
        common_interest=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(n_agents * state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, n_agents), std=1.0),
        )
        # self.common_interest = common_interest

    def forward(self, state):
        """
        :param state:      [batch_size, n_agents, features]
        :return: q-values: [batch_size, n_agents]
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)
        batch_size, n_agents, features = state.shape
        state_flatten = state.reshape(batch_size, n_agents * features)
        return self.net(state_flatten)
        # if self.common_interest:
        #     out = out.mean(dim=-1, keepdim=True).repeat(1, out.shape[-1])
        # return out


class DecentralizedCriticV(nn.Module):
    agent_id: torch.Tensor

    def __init__(self, state_dim, n_agents: int, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim + n_agents, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.register_buffer("agent_id", torch.eye(n_agents), persistent=False)

    # @torch.compile(dynamic=False)
    def forward(self, state):
        """
        :param state:  [batch_size, n_agents, features]
        :return: values: [batch_size, n_agents]
        """
        # if state.dim() == 2:
        #     state = state.unsqueeze(0)
        # return self.net(state).squeeze(-1)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        agent_id = self.agent_id.expand(state.shape[0], -1, -1)
        return self.net(torch.cat((state, agent_id), dim=-1)).squeeze(-1)

    # @torch.compile(dynamic=False)
    def forward_one(self, state_i, i: int):
        # state_i: [B, F] -> [B]
        agent_id = self.agent_id[i].expand(state_i.shape[0], -1)
        return self.net(torch.cat((state_i, agent_id), dim=-1)).squeeze(-1)
