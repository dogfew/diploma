import torch
import torch.nn as nn


class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_branches, n_firms, limit=False):
        super().__init__()
        self.size = n_branches, n_firms
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Hardtanh(),  # HardTanh to avoid overflow.
            nn.Linear(hidden_dim, hidden_dim),
            nn.Hardtanh(),
        )
        self.buy = nn.Sequential(
            nn.Linear(hidden_dim, n_firms * n_branches + 1),
            nn.Softmax(dim=-1),
        )
        self.sale = nn.Sequential(nn.Linear(hidden_dim, n_branches), nn.Sigmoid())
        self.use = nn.Sequential(
            nn.Linear(hidden_dim, n_branches + n_branches * limit),
            nn.Sigmoid(),
        )
        self.prices = nn.Sequential(nn.Linear(hidden_dim, n_branches), nn.Sigmoid())

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self, state
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param state: [batch_size, features]
        :return:
            actions:   [batch_size, n_actions]
            log_probs: [batch_size, n_actions]
        """
        x = self.net(state)
        percent_to_buy = self.buy(x)
        percent_to_sale = self.sale(x)
        percent_to_use = self.use(x)
        percent_price_change = self.prices(x)
        actions = (
            percent_to_buy,
            percent_to_sale,
            percent_to_use,
            percent_price_change,
        )
        log_probs = (
            torch.zeros_like(percent_to_buy),
            torch.zeros_like(percent_to_sale),
            torch.zeros_like(percent_to_use),
            torch.zeros_like(percent_price_change),
        )
        return actions, log_probs
