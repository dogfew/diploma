from collections import deque
import torch
import random


class ReplayBufferOld:
    def __init__(self, size=100_000):
        self._maxsize = size
        self._storage = deque([], maxlen=size)

    def __len__(self):
        return len(self._storage)

    def add(self, x, x_next, actions, rewards):
        """
        :param x: each agent observations
        :param x_next: each agent observations after they performed their action
        :param actions: actions for each agent
        :param rewards: rewards for each agent
        :return:
        """
        data = x, x_next, actions, rewards
        self._storage.append(data)

    def sample(self, batch_size):
        sampled = random.choices(self._storage, k=batch_size)
        return map(lambda x: torch.stack(x), zip(*sampled))


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, n_firms, size=65536, device="cuda"):
        self._maxsize = size
        self._storage_x = torch.empty((size, n_firms, state_dim), device=device)
        self._storage_x_next = torch.empty((size, n_firms, state_dim), device=device)
        self._storage_actions = torch.empty((size, n_firms, action_dim), device=device)
        self._storage_rewards = torch.empty((size, n_firms, 1), device=device)

        self.n_firms = n_firms
        self.index = 0
        self._length = 0

    def __len__(self):
        return self._length

    def add(self, x, x_next, actions, rewards):
        """
        :param x: each agent observations
        :param x_next: each agent observations after they performed their action
        :param actions: actions for each agent
        :param rewards: rewards for each agent
        :return:
        """
        self._storage_x[self.index] = x
        self._storage_x_next[self.index] = x_next
        self._storage_actions[self.index] = actions
        self._storage_rewards[self.index] = rewards

        self.index = (self.index + 1) % self._maxsize
        self._length = max(self._length, self.index)

    def add_batch(self, x, x_next, actions, rewards):
        batch_size = x.size(0)
        indices = torch.arange(self.index, self.index + batch_size) % self._maxsize
        self._storage_x[indices] = x
        self._storage_x_next[indices] = x_next
        self._storage_actions[indices] = actions
        self._storage_rewards[indices] = rewards

        self.index = (self.index + batch_size) % self._maxsize
        self._length = min(self._maxsize, self._length + batch_size)

    def sample(self, batch_size):
        indices = torch.randint(0, len(self), size=(batch_size,))
        x_batch = self._storage_x[indices]
        x_next_batch = self._storage_x_next[indices]
        actions_batch = self._storage_actions[indices]
        rewards_batch = self._storage_rewards[indices]
        return x_batch, x_next_batch, actions_batch, rewards_batch


class ReplayBufferPPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 prob_dim,
                 n_firms,
                 size=65536, device="cuda"):
        self._maxsize = size
        self._storage_x = torch.empty((size, n_firms, state_dim), device=device)
        self._storage_x_next = torch.empty((size, n_firms, state_dim), device=device)
        self._storage_actions = torch.empty((size, n_firms, action_dim), device=device)
        self._storage_values = torch.empty((size, n_firms, 1), device=device)
        self._storage_advantages = torch.empty((size, n_firms, 1), device=device)
        self._storage_value_targets = torch.empty((size, n_firms, 1), device=device)
        self._storage_probs = torch.empty((size, n_firms, prob_dim), device=device)


        self.n_firms = n_firms
        self.index = 0
        self._length = 0

    def __len__(self):
        return self._length

    def add_batch(self, trajectory):
        x, x_next, actions, values, advantages, value_targets, log_probs = map(
            trajectory.get,
            ['x', 'x_next', 'actions',
             'values', 'advantages',  'value_targets', 'log_probs']
        )
        batch_size = x.size(0)
        indices = torch.arange(self.index, self.index + batch_size) % self._maxsize
        self._storage_x[indices] = x
        self._storage_x_next[indices] = x_next
        self._storage_actions[indices] = actions
        self._storage_values[indices] = values
        self._storage_advantages[indices] = advantages
        self._storage_value_targets[indices] = value_targets
        self._storage_probs[indices] = log_probs

        self.index = (self.index + batch_size) % self._maxsize
        self._length = min(self._maxsize, self._length + batch_size)

    def sample(self, batch_size):
        indices = torch.randint(0, len(self), size=(batch_size,))

        x_batch = self._storage_x[indices]
        x_next_batch = self._storage_x_next[indices]
        actions_batch = self._storage_actions[indices]
        values_batch = self._storage_values[indices]
        advantages_batch = self._storage_advantages[indices]
        values_targets_batch = self._storage_value_targets[indices]
        probs_batch = self._storage_probs[indices]
        return (x_batch,
                x_next_batch,
                actions_batch,
                values_batch,
                advantages_batch,
                values_targets_batch,
                probs_batch
                )
