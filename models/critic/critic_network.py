import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Flatten(0),
        )

    def __call__(self, state, action):
        """
        :param state:      [batch_size, features]
        :param action:     [batch_size, actions_dim]
        :return: q-values: [batch_size]
        """
        if len(state) == 1 and len(action) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        concatenated = torch.concat((state, action), dim=-1)
        q_values = self.net(concatenated)
        # assert len(q_values.shape) == 1 and q_values.shape[0] == state.shape[0]
        return q_values


if __name__ == "__main__":
    net = Critic(10, 20)
    x = torch.empty((32, 10))
    y = torch.empty((32, 20))
    print(net(x, y).shape)
