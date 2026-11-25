from typing import Literal

import torch
from torch import nn
from torch.distributions import Beta, Dirichlet
from torch.nn.functional import softplus
from torch.nn.utils import spectral_norm

from src.models.utils.distributions import AdditiveLogisticNormal, LogitNormal

# from src.models.utils.distributions import FastBeta as Beta
# from src.models.utils.distributions import FastDirichlet as Dirichlet
from src.models.utils.initialization import orthogonal_init

EPS = 1e-6
VALIDATE_ARGS = False


class BasePolicyNetwork(nn.Module):
    distr: str  # задаётся в наследниках

    def __init__(  # noqa: PLR0913
        self,
        state_dim,
        n_branches,
        n_firms,
        hidden_dim=64,
        *,
        limit: bool = False,
        blind_market: bool = False,
        interval_prices: bool = False,
        social_planner_is_present: bool = False,
    ):
        super().__init__()
        self.blind_market = blind_market
        self.interval_prices = interval_prices
        self.use_shape = 2 + limit
        k_buy = (1 if blind_market else n_firms) * n_branches + 1
        self.head_sizes = self.get_head_sizes(k_buy, n_branches)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, sum(self.head_sizes)),
        )
        # self.net = nn.Sequential(
        #     nn.LayerNorm(state_dim),
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, sum(self.head_sizes)),
        # )
        self.social_planner_is_present = social_planner_is_present
        self.i = 0
        self.add = 0.5

    def get_head_sizes(self, k_buy, n_branches):
        raise NotImplementedError

    def get_distributions(self, x):
        raise NotImplementedError

    def init_weights(self):
        """
        According to paper: https://arxiv.org/pdf/2006.05990
        :return:
        """
        *hidden, last = [m for m in self.net if isinstance(m, nn.Linear)]

        for layer in hidden:
            orthogonal_init(layer)

        with torch.no_grad():
            for w in last.weight.split(self.head_sizes, dim=0):
                nn.init.orthogonal_(w, gain=0.01)
            last.bias.zero_()

    @property
    def device(self):
        return next(self.parameters()).device

    def main_forward(self, state):
        return self.net(state)

    # @torch.compile(dynamic=False)
    def forward(self, state: torch.Tensor):
        """
        :param state: [batch_size, features]
        :return:
            actions:   [batch_size, n_actions]
            log_probs: [batch_size, n_actions]
        """
        # x = softplus(self.net(state)) + self.add
        x = self.net(state)
        buy_distr, sale_distr, use_distr, prices_distr = self.get_distributions(x)
        percent_price = prices_distr.rsample()
        if not self.interval_prices:
            percent_price = percent_price + (percent_price.clamp(EPS, 1 - EPS) - percent_price).detach()

        percent_to_buy = buy_distr.rsample()
        percent_to_use = use_distr.rsample()
        percent_to_sale = sale_distr.rsample()
        percent_to_sale = percent_to_sale + (percent_to_sale.clamp(EPS, 1 - EPS) - percent_to_sale).detach()

        buy_log_prob = buy_distr.log_prob(percent_to_buy)
        log_probs = (
            buy_log_prob.unsqueeze(-1),
            sale_distr.log_prob(percent_to_sale),
            use_distr.log_prob(percent_to_use),
            prices_distr.log_prob(percent_price) * (not self.social_planner_is_present),
        )
        actions = (
            percent_to_buy,
            percent_to_sale,
            percent_to_use.flatten(1),
            percent_price,
        )
        return actions, log_probs

    # @torch.compile(dynamic=False)
    def get_log_probs_and_entropy(self, state, actions):
        percent_to_buy, percent_to_sale, percent_to_use, prices = actions
        # x = softplus(self.net(state)) + self.add
        x = self.net(state)
        buy_distr, sale_distr, use_distr, prices_distr = self.get_distributions(x)
        percent_to_use = percent_to_use.unflatten(-1, (-1, self.use_shape))
        log_probs = (
            buy_distr.log_prob(percent_to_buy).unsqueeze(-1),
            sale_distr.log_prob(percent_to_sale),
            use_distr.log_prob(percent_to_use),
            prices_distr.log_prob(prices) * (not self.social_planner_is_present),
        )
        bs = x.shape[0]
        entropy = sum(
            distr.entropy().reshape(bs, -1).sum(dim=-1)
            for distr in [
                buy_distr,
                sale_distr,
                use_distr,
            ]
            + [prices_distr] * (not self.social_planner_is_present)
        )
        return log_probs, entropy


class BetaPolicyNetwork(BasePolicyNetwork):
    def get_head_sizes(self, k_buy, n_branches):
        return [k_buy, 2 * n_branches, self.use_shape * n_branches, (2 + self.interval_prices) * n_branches]

    def get_distributions(self, x):
        buy_params, sale_params, use_params, prices_params = (softplus(x) + self.add).split(
            self.head_sizes,
            dim=-1,
        )
        sale_params = sale_params.unflatten(-1, (-1, 2))
        prices_params = prices_params.unflatten(-1, (-1, 2 + self.interval_prices))

        buy_distr = Dirichlet(buy_params)
        use_distr = Dirichlet(use_params.unflatten(-1, (-1, self.use_shape)))
        sale_distr = Beta(sale_params[..., 0], sale_params[..., 1])
        prices_distr = Beta(prices_params[..., 0], prices_params[..., 1])
        if self.interval_prices:
            prices_distr = Dirichlet(prices_params + self.add)
        return buy_distr, sale_distr, use_distr, prices_distr


class LogisticNormalPolicyNetwork(BasePolicyNetwork):
    def get_head_sizes(self, k_buy, n_branches):  # loc и scale
        return [
            2 * (k_buy - 1),
            2 * n_branches,
            2 * (self.use_shape - 1) * n_branches,
            (2 + self.interval_prices) * n_branches,
        ]

    def get_distributions(self, x):
        buy_params, sale_params, use_params, prices_params = x.split(self.head_sizes, dim=-1)
        sale_params = sale_params.unflatten(-1, (-1, 2))
        prices_params = prices_params.unflatten(-1, (-1, 2 + self.interval_prices))

        buy_loc, buy_scale = buy_params.chunk(2, dim=-1)
        use_loc, use_scale = use_params.unflatten(-1, (-1, 2 * (self.use_shape - 1))).chunk(2, dim=-1)
        buy_distr = AdditiveLogisticNormal(buy_loc, softplus(buy_scale + 1.4) + 1e-2)
        use_distr = AdditiveLogisticNormal(use_loc, softplus(use_scale + 1.4) + 1e-2)
        if False:  # заглушка: Beta вместо LogitNormal
            sale_params = softplus(sale_params) + self.add
            prices_beta = softplus(prices_params) + self.add
            sale_distr = Beta(sale_params[..., 0], sale_params[..., 1])
            prices_distr = Beta(prices_beta[..., 0], prices_beta[..., 1])
        else:
            sale_distr = LogitNormal(sale_params[..., 0], softplus(sale_params[..., 1] + 1.4) + 1e-2)
            prices_distr = LogitNormal(prices_params[..., 0], softplus(prices_params[..., 1] + 1.4) + 1e-2)
        if self.interval_prices:
            prices_distr = Dirichlet(softplus(prices_params) + self.add)
        return buy_distr, sale_distr, use_distr, prices_distr


# class BetaPolicyNetwork(nn.Module):
#     def __init__(  # noqa: PLR0913
#         self,
#         state_dim,
#         n_branches,
#         n_firms,
#         hidden_dim=64,
#         *,
#         limit: bool = False,
#         blind_market: bool = False,
#         interval_prices: bool = False,
#         social_planner_is_present: bool = False,
#         distr: Literal["Beta", "LogitNormal"] = "LogitNormal",
#     ):
#         super().__init__()
#         self.blind_market = blind_market
#         self.interval_prices = interval_prices
#         self.distr = distr
#         k_buy = (1 if blind_market else n_firms) * n_branches + 1
#         self.use_shape = 2 + limit
#         if distr == "Beta":
#             buy_size, use_size = k_buy, self.use_shape * n_branches
#         else:  # loc и scale
#             buy_size, use_size = 2 * (k_buy - 1), 2 * (self.use_shape - 1) * n_branches
#         ln = distr == "LogitNormal"
#         self.head_sizes = [
#             (1 + ln) * (k_buy - ln),
#             2 * n_branches,
#             (1 + ln) * (self.use_shape - ln) * n_branches,
#             (2 + interval_prices) * n_branches,
#         ]
#         self.head_sizes = [buy_size, 2 * n_branches, use_size, (2 + interval_prices) * n_branches]
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, sum(self.head_sizes)),
#         )
#         # self.net = nn.Sequential(
#         #     nn.LayerNorm(state_dim),
#         #     nn.Linear(state_dim, hidden_dim),
#         #     nn.LeakyReLU(),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.LeakyReLU(),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.Linear(hidden_dim, sum(self.head_sizes)),
#         # )
#         self.social_planner_is_present = social_planner_is_present
#         self.i = 0
#         self.add = 0.5

#     def init_weights(self):
#         """
#         According to paper: https://arxiv.org/pdf/2006.05990
#         :return:
#         """
#         *hidden, last = [m for m in self.net if isinstance(m, nn.Linear)]

#         for layer in hidden:
#             orthogonal_init(layer)

#         with torch.no_grad():
#             for w in last.weight.split(self.head_sizes, dim=0):
#                 nn.init.orthogonal_(w, gain=0.01)
#             last.bias.zero_()

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def main_forward(self, state):
#         return self.net(state)

#     def get_distributions(self, x):
#         if self.distr == "Beta":
#             buy_params, sale_params, use_params, prices_params = (softplus(x) + self.add).split(
#                 self.head_sizes,
#                 dim=-1,
#             )
#             sale_params = sale_params.unflatten(-1, (-1, 2))
#             prices_params = prices_params.unflatten(-1, (-1, 2 + self.interval_prices))

#             buy_distr = Dirichlet(buy_params)
#             use_distr = Dirichlet(use_params.unflatten(-1, (-1, self.use_shape)))
#             sale_distr = Beta(sale_params[..., 0], sale_params[..., 1])
#             prices_distr = Beta(prices_params[..., 0], prices_params[..., 1])
#             if self.interval_prices:
#                 prices_distr = Dirichlet(prices_params + self.add)
#         else:
#             buy_params, sale_params, use_params, prices_params = x.split(self.head_sizes, dim=-1)
#             sale_params = sale_params.unflatten(-1, (-1, 2))
#             prices_params = prices_params.unflatten(-1, (-1, 2 + self.interval_prices))

#             buy_loc, buy_scale = buy_params.chunk(2, dim=-1)
#             use_loc, use_scale = use_params.unflatten(-1, (-1, 2 * (self.use_shape - 1))).chunk(2, dim=-1)
#             buy_distr = AdditiveLogisticNormal(buy_loc, softplus(buy_scale + 1.4) + 1e-2)
#             use_distr = AdditiveLogisticNormal(use_loc, softplus(use_scale + 1.4) + 1e-2)
#             # sale_distr = Beta(sale_params[..., 0], sale_params[..., 1])
#             # prices_distr = Beta(prices_params[..., 0], prices_params[..., 1])
#             if True:  # заглушка: Beta вместо LogitNormal
#                 sale_params = softplus(sale_params) + self.add
#                 prices_beta = softplus(prices_params) + self.add
#                 sale_distr = Beta(sale_params[..., 0], sale_params[..., 1])
#                 prices_distr = Beta(prices_beta[..., 0], prices_beta[..., 1])
#             else:
#                 sale_distr = LogitNormal(sale_params[..., 0], softplus(sale_params[..., 1] + 1.4) + 1e-2)
#                 prices_distr = LogitNormal(
#                     prices_params[..., 0], softplus(prices_params[..., 1] + 1.4) + 1e-2
#                 )
#             if self.interval_prices:
#                 prices_distr = Dirichlet(softplus(prices_params) + self.add)
#         return buy_distr, sale_distr, use_distr, prices_distr

#     def forward(self, state: torch.Tensor):
#         """
#         :param state: [batch_size, features]
#         :return:
#             actions:   [batch_size, n_actions]
#             log_probs: [batch_size, n_actions]
#         """
#         # x = softplus(self.net(state)) + self.add
#         x = self.net(state)
#         buy_distr, sale_distr, use_distr, prices_distr = self.get_distributions(x)
#         percent_price = prices_distr.rsample()
#         if not self.interval_prices:
#             percent_price = percent_price + (percent_price.clamp(EPS, 1 - EPS) - percent_price).detach()

#         percent_to_buy = buy_distr.rsample()
#         percent_to_use = use_distr.rsample()
#         percent_to_sale = sale_distr.rsample()
#         percent_to_sale = percent_to_sale + (percent_to_sale.clamp(EPS, 1 - EPS) - percent_to_sale).detach()

#         buy_log_prob = buy_distr.log_prob(percent_to_buy)
#         log_probs = (
#             buy_log_prob.unsqueeze(-1),
#             sale_distr.log_prob(percent_to_sale),
#             use_distr.log_prob(percent_to_use),
#             prices_distr.log_prob(percent_price) * (not self.social_planner_is_present),
#         )
#         actions = (
#             percent_to_buy,
#             percent_to_sale,
#             percent_to_use.flatten(1),
#             percent_price,
#         )
#         return actions, log_probs

#     def get_log_probs_and_entropy(self, state, actions):
#         percent_to_buy, percent_to_sale, percent_to_use, prices = actions
#         # x = softplus(self.net(state)) + self.add
#         x = self.net(state)
#         buy_distr, sale_distr, use_distr, prices_distr = self.get_distributions(x)
#         percent_to_use = percent_to_use.unflatten(-1, (-1, self.use_shape))
#         log_probs = (
#             buy_distr.log_prob(percent_to_buy).unsqueeze(-1),
#             sale_distr.log_prob(percent_to_sale),
#             use_distr.log_prob(percent_to_use),
#             prices_distr.log_prob(prices) * (not self.social_planner_is_present),
#         )
#         bs = x.shape[0]
#         entropy = sum(
#             distr.entropy().reshape(bs, -1).sum(dim=-1)
#             for distr in [
#                 buy_distr,
#                 sale_distr,
#                 use_distr,
#             ]
#             + [prices_distr] * (not self.social_planner_is_present)
#         )
#         return log_probs, entropy


class SocialPlannerBetaPolicyNetwork(BetaPolicyNetwork):
    """Планировщик управляет только ценами. Остальные действия — заглушки для совместимости форм."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_weights()

    def forward(self, state):
        actions, (buy_lp, sale_lp, use_lp, price_lp) = super().forward(state)
        log_probs = (
            torch.zeros_like(buy_lp),
            torch.zeros_like(sale_lp),
            torch.zeros_like(use_lp),
            price_lp,
        )
        return actions, log_probs

    def get_log_probs_and_entropy(self, state, actions):
        percent_to_buy, percent_to_sale, percent_to_use, prices = actions
        *_, prices_distr = self.get_distributions(self.net(state))
        bs = state.shape[0]
        log_probs = (
            state.new_zeros(bs, 1),
            torch.zeros_like(percent_to_sale),
            state.new_zeros(bs, percent_to_use.shape[-1] // self.use_shape),
            prices_distr.log_prob(prices),
        )
        entropy = prices_distr.entropy().reshape(bs, -1).sum(dim=-1)
        return log_probs, entropy


class BetaPolicyNetwork2(BetaPolicyNetwork):
    def __init__(self, state_dim, n_branches, n_firms, hidden_dim=32, *, limit=False):
        super().__init__(
            state_dim,
            n_branches,
            n_firms,
            hidden_dim,
            limit=limit,
        )
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
