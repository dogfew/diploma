import math
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from src.environment.firm import SocialPlanner
from src.models.policy.beta_policy import SocialPlannerBetaPolicyNetwork

from .firm import (
    BatchedFirm,
    BatchedLimitFirm,
    BatchedLimitProductionFirm,
    BatchedProductionFirm,
)
from .market import BatchedMarket, BlindMarket
from .utils import (
    get_action_dim,
    get_log_probs_dim,
    get_state_dim,
    process_actions,
)

if TYPE_CHECKING:
    from src.models.policy.beta_policy import BetaPolicyNetwork

ADD_TO_LOG = 1


class BatchedEnvironment:
    """
    Parallel Environment
    """

    def __init__(
        self,
        market_kwargs,
        policy_class,
        prod_functions,
        batch_size=16,
        hidden_dim=64,
        *,
        production_reg=0.1,
        gamma=0.99,
        target="finance",
        normalize_rewards=False,
        percent_prices=False,
        is_blind=False,
        device="cuda",
        invest_functions=None,
        social_planner=False,
    ):
        self.mode = target
        self.social_planner_is_present = social_planner
        self.normalize_rewards = normalize_rewards
        self.batch_size = batch_size
        self.gamma = gamma
        self.limit = invest_functions is not None
        self.device = device
        # Market
        self.market = (BatchedMarket if not is_blind else BlindMarket)(
            **market_kwargs,
            batch_size=batch_size,
            n_branches=prod_functions[0].n_branches,
            device=device,
        )
        # Firms
        firm_kwargs = {
            "market": self.market,
            "production_reg": production_reg,
            "batch_size": batch_size,
            "social_planner": social_planner,
        }
        if invest_functions is None:
            firm_class = {"production": BatchedProductionFirm, "finance": BatchedFirm}[target]
            self.firms = [firm_class(fun, **firm_kwargs) for fun in prod_functions]
        else:
            firm_class = {"production": BatchedLimitProductionFirm, "finance": BatchedLimitFirm}[target]
            self.firms = [
                firm_class(fun, inv_fun, **firm_kwargs)
                for fun, inv_fun in zip(prod_functions, invest_functions, strict=True)
            ]
        if percent_prices:
            assert not self.market.is_blind, f"Can't use percent prices with {self.market.__class__.__name__}"
            for firm in self.firms:
                firm.define_prices = firm.define_prices_percent  # ty: ignore[invalid-assignment]

        self.state_dim: int = get_state_dim(self.market, limit=self.limit)
        self.action_dim: int = get_action_dim(self.market, limit=self.limit, is_blind=self.market.is_blind)
        self.probs_dim: int = get_log_probs_dim(self.market)
        self.policies: list[BetaPolicyNetwork | SocialPlannerBetaPolicyNetwork] = [
            policy_class(
                hidden_dim=hidden_dim,
                state_dim=self.state_dim,
                n_branches=self.market.n_branches,
                n_firms=self.market.n_firms,
                limit=self.limit,
                blind_market=self.market.is_blind,
                social_planner_is_present=social_planner,
            ).to(device)
            for _ in self.firms
        ]
        self.social_planner_id = len(self.firms)
        if social_planner:
            self.firms.append(SocialPlanner(market=self.market))  # ty: ignore[invalid-argument-type]
            self.policies.append(
                SocialPlannerBetaPolicyNetwork(
                    hidden_dim=hidden_dim,
                    state_dim=self.state_dim,
                    n_branches=self.market.n_branches,
                    n_firms=self.market.n_firms,
                    limit=self.limit,
                    blind_market=self.market.is_blind,
                ).to(device),
            )
            self.firms[-1].id = self.social_planner_id
        self.target_policies = [deepcopy(policy) for policy in self.policies]
        self.limit = invest_functions is not None

        # Record Dynamics of Environment
        self.state_history = []
        self.actions_history = {firm.id: [] for firm in self.firms}
        n_firms = self.market.n_firms
        n_branches = self.market.n_branches
        interval_prices = False
        self.actions_split_sizes = [
            n_branches * n_firms + 1 if not self.market.is_blind else n_branches + 1,  # percent to buy
            n_branches,  # percent to sale
            2 * self.market.price_matrix.shape[2]
            + self.limit * self.market.price_matrix.shape[2],  # percent to use
            self.market.price_matrix.shape[2] * (1 + 2 * interval_prices),  # price change
            # self.market.price_matrix.shape[2] * 1,  # price change
        ]

        # Running vars
        # self.running_vars = self.init_running_vars()

    # def init_running_vars(self):
    #     return {
    #         firm_id: {
    #             "mean_volume": torch.zeros(self.market.n_branches * self.market.n_firms, device=self.device),
    #             "mean_reserve": torch.zeros(self.market.n_branches, device=self.device),
    #             "var_volume": torch.ones(self.market.n_branches * self.market.n_firms, device=self.device),
    #             "var_reserve": torch.ones(self.market.n_branches, device=self.device),
    #             "count_volume": 1e-4,
    #             "count_reserve": 1e-4,
    #             "mean_reward": torch.zeros(self.batch_size, device=self.device),
    #             "var_reward": torch.ones(self.batch_size, device=self.device),
    #             "count_reward": 1e-4,
    #             "cum_reward": torch.zeros(self.batch_size, device=self.device),
    #         }
    #         for firm_id in range(self.n_agents)
    #     }

    @property
    def n_agents(self):
        return len(self.firms)  # self.market.max_id + self.social_planner_id

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.market.change_batch_size(batch_size)
        for firm in self.firms:
            firm.change_batch_size(batch_size)
        self.reset()

    def reset(self) -> None:
        """
        Reset Environment (Market + all firms) state
        """
        self.market.reset()
        self.total_money = sum(self.market.gains[0])
        for firm in self.firms:
            firm.reset()
            self.total_money += firm.financial_resources[0].item()
        if len(self.state_history):
            self.state_history = []
            self.actions_history = {firm.id: [] for firm in self.firms}
        # self.running_vars = self.init_running_vars()

    @torch.no_grad()
    def get_social_planner_state(self):
        market = self.market
        parts = []
        for xfirm in self.firms[:-1]:
            parts.extend(
                [
                    xfirm.reserves.flatten(start_dim=1),
                    xfirm.financial_resources + market.gains[:, xfirm.id, None],
                ],
            )

        parts.extend(
            [
                torch.cat([xfirm.limit, xfirm.capital[:, xfirm.current_step, None]], dim=1)
                for xfirm in self.firms[:-1]
                if isinstance(xfirm, (BatchedLimitFirm, BatchedLimitProductionFirm))
            ],
        )

        combined = torch.cat(parts, dim=1)
        padding = torch.zeros((market.batch_size, self.state_dim - combined.shape[1]), device=self.device)
        return (torch.cat([combined, padding], dim=1).float() + ADD_TO_LOG).log() / math.log(10)

    @torch.no_grad()
    # @torch.compile(dynamic=False)
    def get_state(
        self,
        firm_id,
    ):
        """
        input_prices  = log(p)
        input_volumes = log(v + 1)
        """
        add = ADD_TO_LOG
        firm = self.firms[firm_id]
        if firm_id == self.social_planner_id:
            return self.get_social_planner_state()
        market = self.market
        # Get price state. Normalize prices to [-1, 1] range
        price_range = market.max_price - market.min_price
        price_state = (2 * market.price_matrix.flatten(start_dim=1) - price_range) / price_range
        # Get Volume State.
        volume_state = market.observable_volume_matrix.flatten(start_dim=1).log1p()

        # Get Reserve State.
        reserves_state = firm.reserves.flatten(start_dim=1).log1p()
        # Normalize finance to [-1, 1]
        finance_state = (firm.financial_resources + market.gains[:, firm.id, None]) / self.total_money - 1.0
        # finance_state = (finance_state - self.total_money) / self.total_money
        state_lst = [
            price_state,  # [-1, 1]
        ] * (not self.market.is_blind) + [
            volume_state,  # [0, log1p)
            reserves_state,  # [0, log1p)
            finance_state,  # [-1, 1]
        ]
        if isinstance(firm, (BatchedLimitFirm, BatchedLimitProductionFirm)):
            limit_state = torch.cat([firm.limit, firm.capital[:, firm.current_step, None]], dim=1).log1p()
            state_lst.append(limit_state)
        return torch.concatenate(state_lst, dim=1)

    @torch.no_grad()
    # @torch.compile(dynamic=False)
    def step(self, firm_id) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param firm_id:
        :return: state, actions_concatenated, log_probs_concatenated, revenue, costs
        """
        firm = self.firms[firm_id]
        policy = self.policies[firm_id]
        state = self.get_state(firm_id)
        actions, log_probs = policy(state)
        if self.market.interval_prices:
            a, b, c = actions[-1].shape
            actions = (*actions[:-1], actions[-1].reshape(a, b * c))
        actions_concatenated = torch.concatenate(actions, dim=-1)
        log_probs_concatenated = torch.concatenate(log_probs, dim=-1)
        processed_actions = process_actions(actions, self.market.price_matrix.shape)
        revenue, costs = firm.step(*processed_actions)
        return state, actions_concatenated, log_probs_concatenated, revenue, costs

    @torch.no_grad()
    def restore_actions(self, actions):
        assert actions.dim() != 1, "dim"
        percent_to_buy, percent_to_sale, percent_to_use, prices = torch.split(
            actions,
            self.actions_split_sizes,
            dim=1,
        )
        return (
            percent_to_buy,
            percent_to_sale,
            percent_to_use,
            prices,
        )

    @torch.no_grad()
    def step_and_record_batch(self, firm_id):
        _state, actions_concatenated, _log_probs_concatenated, _revenue, _costs = self.step(firm_id)
        state_info = {
            "price_matrix": self.market.price_matrix.clone().cpu().numpy(),
            "volume_matrix": self.market.actual_volume_matrix.clone().cpu().numpy(),
            "finance": self.market.gains.clone().cpu().numpy()
            + torch.stack(
                [firm.financial_resources for firm in self.firms if not isinstance(firm, SocialPlanner)],
            )
            .permute(1, 0, 2)
            .squeeze(-1)
            .cpu()
            .numpy(),
            "reserves": torch.stack(
                [firm.reserves for firm in self.firms if not isinstance(firm, SocialPlanner)],
            )
            .cpu()
            .numpy(),
        }
        if self.limit:
            state_info["limits"] = (
                torch.stack(
                    [firm.limit for firm in self.firms if not isinstance(firm, SocialPlanner)],
                )
                .cpu()
                .numpy()
            )
        percent_to_buy, percent_to_sale, percent_to_use, prices = process_actions(
            torch.split(
                actions_concatenated.cpu(),
                self.actions_split_sizes,
                dim=1,
            ),
            size=self.market.price_matrix.shape,
            is_blind=self.market.is_blind,
        )
        policy_info = {
            "percent_to_buy": percent_to_buy.numpy(),
            "percent_to_sale": percent_to_sale.numpy(),
            "percent_to_use": percent_to_use.numpy(),
            "prices": prices.numpy(),
        }

        self.state_history.append(state_info)
        self.actions_history[firm_id].append(policy_info)
