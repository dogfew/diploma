import torch
import torch.nn as nn
from torch.distributions import Beta, Dirichlet
from torch.nn.utils import spectral_norm


class BetaPolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        n_branches,
        n_firms,
        hidden_dim=64,
        limit=False,
        eps=1e-8,
        min_log_prob=-10,
    ):
        super().__init__()
        self.eps = eps
        self.min_log_prob = min_log_prob
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.buy = nn.Sequential(
            nn.Linear(hidden_dim, n_firms * n_branches + 1),
            nn.Softplus(),
        )
        self.sale = nn.Sequential(
            nn.Linear(hidden_dim, 2 * n_branches),
            nn.Softplus(),
            nn.Unflatten(-1, (n_branches, 2)),
        )

        self.use = nn.Sequential(
            nn.Linear(hidden_dim, 2 * n_branches if not limit else 3 * n_branches),
            nn.Softplus(),
            nn.Unflatten(-1, (n_branches, 2 + limit)),
        )
        self.prices = nn.Sequential(
            nn.Linear(hidden_dim, 2 * n_branches),
            nn.Softplus(),
            nn.Unflatten(-1, (n_branches, 2)),
        )
        self.init_weights()
        # self.apply_spectral_norm(self)

    def apply_spectral_norm(self, module):
        for child in module.children():
            if isinstance(child, nn.Linear):
                nn.utils.spectral_norm(child)
            elif isinstance(child, nn.Module):
                self.apply_spectral_norm(child)

    def init_weights(self):
        """
        According to paper: https://arxiv.org/pdf/2006.05990
        :return:
        """
        # for module in [self.buy, self.sale, self.use, self.prices]:
        #     for layer in module:
        #         if isinstance(layer, nn.Linear):
        #             nn.init.orthogonal_(layer.weight, 1.41)
        for module in [self.buy, self.sale, self.use, self.prices]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # nn.init.uniform_(layer.weight, 0, 0.01)
                    nn.init.xavier_uniform_(layer.weight)
                    layer.weight.data /= 100
    @property
    def device(self):
        return next(self.parameters()).device

    def main_forward(self, state):
        return self.net(state)

    def forward(self, state):
        """
        :param state: [batch_size, features]
        :return:
            actions:   [batch_size, n_actions]
            log_probs: [batch_size, n_actions]
        """
        x = self.main_forward(state)
        # print("X OLD", x)
        buy_params = self.buy(x)
        sale_params = self.sale(x)
        use_params = self.use(x)
        prices_params = self.prices(x)
        buy_distr = Dirichlet(buy_params)
        sale_distr = Beta(sale_params[..., 0], sale_params[..., 1])
        use_distr = Dirichlet(use_params)
        price_distr = Beta(prices_params[..., 0], prices_params[..., 1])

        percent_to_buy = buy_distr.rsample()
        percent_to_sale = sale_distr.rsample()
        percent_to_use = use_distr.rsample()
        percent_price_change = price_distr.rsample()

        buy_log_prob = buy_distr.log_prob(percent_to_buy)
        use_log_prob = use_distr.log_prob(percent_to_use)
        if buy_log_prob.dim() == 0:
            percent_to_use = percent_to_use[:, :-1].flatten()
        else:
            percent_to_use = percent_to_use[:, :, :-1].flatten(1)
        actions = (
            percent_to_buy,
            percent_to_sale,
            percent_to_use,
            percent_price_change,
        )
        log_probs = (
            buy_log_prob.unsqueeze(-1),
            sale_distr.log_prob(percent_to_sale),
            use_log_prob,
            price_distr.log_prob(percent_price_change),
        )
        return actions, log_probs

    def get_log_probs(self, state, actions):
        (percent_to_buy,
         percent_to_sale,
         percent_to_use,
         prices
         ) = actions
        x = self.main_forward(state)
        buy_params = self.buy(x)
        sale_params = self.sale(x)
        use_params = self.use(x)
        prices_params = self.prices(x)

        log_probs = (
            Dirichlet(buy_params).log_prob(percent_to_buy).unsqueeze(-1),
            Beta(sale_params[..., 0], sale_params[..., 1]).log_prob(percent_to_sale),
            Dirichlet(use_params).log_prob(percent_to_use),
            Beta(prices_params[..., 0], prices_params[..., 1]).log_prob(prices),
        )

        return log_probs

class BetaPolicyNetwork2(BetaPolicyNetwork):
    def __init__(
        self, state_dim, n_branches, n_firms, hidden_dim=32, limit=False, eps=1e-8
    ):
        super().__init__(state_dim, n_branches, n_firms, hidden_dim, limit, eps)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(state_dim, state_dim)),
            nn.Tanh(),  # Limiting max value to avoid overflow.
            spectral_norm(nn.Linear(state_dim, state_dim)),
            nn.Tanh(),  # Limiting max value to avoid overflow.
        )

        self.net2 = nn.Sequential(
            spectral_norm(nn.Linear(state_dim * 2, hidden_dim)),
            nn.Tanh(),  # Limiting max value to avoid overflow.
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),  # Limiting max value to avoid overflow.
        )

    def main_forward(self, state):
        concatenated = torch.concat([self.net(state), state], dim=-1)
        return self.net2(concatenated)
