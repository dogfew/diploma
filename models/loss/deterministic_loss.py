import torch
import torch.nn as nn


class DeterministicCriticLoss(nn.Module):
    def __init__(self, gamma=0.99, *args, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.loss = nn.MSELoss()

    def forward(self, critic_net, obs, obs_next, actions, actions_next, rewards):
        """
        :param critic_net: Network for Critic
        :param obs:      [batch_size, n_agents, n_features]
        :param obs_next: [batch_size, n_agents, n_features]
        :param actions:  [batch_size, n_agents, n_actions]
        :param actions_next: a_j = target_policy_j(o_j)
        :param rewards:  [batch_size, n_agents]
        :return:
        """
        predicted = critic_net(obs, actions)
        with torch.no_grad():
            target = rewards + self.gamma * critic_net(obs_next, actions_next)
        return self.loss(predicted, target)
