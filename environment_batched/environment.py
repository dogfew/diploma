import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from tensordict import TensorDictBase, TensorDict

from .market import BatchedMarket
from .firm import BatchedFirm, BatchedLimitFirm, BatchedLimitProductionFirm, BatchedProductionFirm
from .utils import get_state_log, get_state_dim, get_action_dim, process_actions, get_log_probs_dim


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
            production_reg=0.1,
            gamma=0.99,
            target='finance',
            normalize_rewards=False,
            percent_prices=False,
            device="cuda",
            invest_functions=None,
    ):
        self.mode = target
        self.normalize_rewards = normalize_rewards
        self.batch_size = batch_size
        self.gamma = gamma
        self.limit = invest_functions is not None
        self.device = device
        # Market
        self.market = BatchedMarket(
            **market_kwargs,
            batch_size=batch_size,
            n_branches=prod_functions[0].n_branches,
            device=device
        )

        # Firms
        firm_kwargs = {
            "market": self.market,
            "production_reg": production_reg,
            "batch_size": batch_size}
        if invest_functions is None:
            firm_class = {'production': BatchedProductionFirm,
                          'finance': BatchedFirm}[target]
            self.firms = [
                firm_class(fun, **firm_kwargs)
                for fun in prod_functions
            ]
        else:
            firm_class = {'production': BatchedLimitProductionFirm,
                          'finance': BatchedLimitFirm}[target]
            self.firms = [
                firm_class(fun, inv_fun, **firm_kwargs)
                for fun, inv_fun in zip(prod_functions, invest_functions)
            ]
        if percent_prices:
            for firm in self.firms:
                firm.define_prices = firm.define_prices_percent
        self.state_dim = get_state_dim(self.market, self.limit)
        self.action_dim = get_action_dim(self.market, self.limit)
        self.probs_dim = get_log_probs_dim(self.market)
        self.policies = [
            policy_class(
                hidden_dim=hidden_dim,
                state_dim=self.state_dim,
                n_branches=self.market.n_branches,
                n_firms=self.market.n_firms,
                limit=self.limit,
            ).to(device)
            for _ in self.firms
        ]
        self.target_policies = [deepcopy(policy) for policy in self.policies]
        self.limit = invest_functions is not None

        # Record Dynamics of Environment
        self.state_history = []
        self.actions_history = {firm.id: [] for firm in self.firms}

        self.actions_split_sizes = [
            math.prod(self.market.price_matrix.shape[1:]) + 1,  # percent to buy
            self.market.price_matrix.shape[2],  # percent to sale
            self.market.price_matrix.shape[2]
            + self.limit * self.market.price_matrix.shape[2],  # percent to use
            self.market.price_matrix.shape[2],  # price change
        ]

        # Running vars
        self.running_vars = self.init_running_vars()

    def init_running_vars(self):
        return {
            firm_id: {
                'mean_volume': torch.zeros(self.market.n_branches * self.market.n_firms, device=self.device),
                'mean_reserve': torch.zeros(self.market.n_branches, device=self.device),
                'var_volume': torch.ones(self.market.n_branches * self.market.n_firms, device=self.device),
                'var_reserve': torch.ones(self.market.n_branches, device=self.device),
                'count_volume': 1e-4,
                'count_reserve': 1e-4,
                'mean_reward': torch.zeros(self.batch_size, device=self.device),
                'var_reward': torch.ones(self.batch_size, device=self.device),
                'count_reward': 1e-4,
                'cum_reward': torch.zeros(self.batch_size, device=self.device),
            }
            for firm_id in range(self.n_agents)
        }

    @property
    def n_agents(self):
        return self.market.max_id

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
        for firm in self.firms:
            firm.reset()
        if len(self.state_history):
            self.state_history = []
            self.actions_history = {firm.id: [] for firm in self.firms}
        self.running_vars = self.init_running_vars()

    def update_moments(self, vector, name, firm_id):
        batch_mean = vector.mean(dim=0)
        batch_var = torch.nan_to_num(vector.var(dim=0, correction=vector.dim() > 1), 1)
        batch_count = vector.shape[0] if vector.dim() > 1 else 1
        mean, var, count = map(lambda x: self.running_vars[firm_id].get(f'{x}_{name}'), ['mean', 'var', 'count'])
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        self.running_vars[firm_id][f'mean_{name}'] = new_mean
        self.running_vars[firm_id][f'var_{name}'] = new_var
        self.running_vars[firm_id][f'count_{name}'] = new_count
        return new_mean, new_var.clamp_min(1e-6), new_count

    @torch.no_grad()
    def get_state(self, firm_id, normalize=False):
        """
        input_prices  = log10(p)
        input_volumes = log(v + 1)
        """
        market = self.market
        firm = self.firms[firm_id]

        # Get price state. Normalize prices to [-1, 1] range
        price_scaling = (market.max_price + market.min_price) / 2
        price_state = (market.price_matrix.flatten(start_dim=1) - price_scaling) / price_scaling

        # Get Volume State.
        volume_state = (market.volume_matrix.flatten(start_dim=1).float() + 0.5).log10()

        # Get Reserve State.
        reserves_state = (firm.reserves.flatten(start_dim=1).float() + 0.5).log10()

        # Optional Normalize
        if normalize:
            mean, var, count = self.update_moments(volume_state, 'volume', firm_id)
            volume_state = (volume_state - mean) / var
            mean, var, count = self.update_moments(reserves_state, 'reserve', firm_id)
            reserves_state = (reserves_state - mean) / var
        # Normalize finance to [-1, 1]
        finance_state = firm.financial_resources + market.gains[:, firm.id, None]
        finance_state = (finance_state - market.max_price / market.n_firms) / market.max_price
        state_lst = [
            price_state,  # [-1, 1]
            volume_state,  # [0, log10)
            reserves_state,  # [0, log10)
            finance_state,  # [-1, 1]
        ]
        if hasattr(firm, "capital"):
            first_part = firm.limit
            second_part = firm.capital[:, firm.current_step, None]
            limit_state = (
                (torch.cat([first_part, second_part], dim=1) + 0.5).log10()
            )
            state_lst.append(limit_state)
        concatenated_state = torch.concatenate(state_lst, dim=1)
        return concatenated_state.type(torch.float32)

    @torch.no_grad()
    def step(self, firm_id) -> tuple[[torch.Tensor]]:
        """
        :param firm_id:
        :return: state, actions_concatenated, log_probs_concatenated, revenue, costs
        """
        firm = self.firms[firm_id]
        policy = self.policies[firm_id]

        state = self.get_state(firm_id)
        actions, log_probs = policy(state)
        actions_concatenated = torch.concatenate(actions, dim=-1)
        log_probs_concatenated = torch.concatenate(log_probs, dim=-1)
        processed_actions = process_actions(actions, self.market.price_matrix.shape)
        revenue, costs = firm.step(*processed_actions)
        return state, actions_concatenated, log_probs_concatenated, revenue, costs




    @torch.no_grad()
    def restore_actions(self, actions):
        if actions.dim() == 1:
            actions = actions.unsqueeze(dim=0)
        percent_to_buy, percent_to_sale, percent_to_use, prices = torch.split(
            actions, self.actions_split_sizes, dim=1
        )
        if not self.limit:
            percent_to_use = torch.stack([percent_to_use, 1 - percent_to_use], dim=-1)
        else:
            percent_to_use = percent_to_use.reshape(
                self.batch_size, self.market.n_branches, 2)
            percent_to_use = torch.cat([percent_to_use, 1 - percent_to_use.sum(dim=-1, keepdim=True)], dim=-1).clamp(
                1e-6, 1 - 1e-6)
        return (percent_to_buy,
                percent_to_sale,
                percent_to_use,
                prices
                )

    @torch.no_grad()
    def step_and_record(self, firm_id):
        state, actions_concatenated, log_probs_concatenated, revenue, costs = map(
            lambda x: x[0], self.step(firm_id)
        )
        state_info = {
            "price_matrix": self.market.price_matrix[0].cpu().numpy(),
            "volume_matrix": self.market.volume_matrix[0].cpu().numpy(),
            "finance": self.market.gains[0].cpu().numpy()
                       + np.array([firm.financial_resources[0].item() for firm in self.firms]),
            "reserves": torch.stack([firm.reserves[0] for firm in self.firms])
            .cpu()
            .numpy(),
        }
        if self.limit:
            state_info["limits"] = (
                torch.tensor([firm.limit[0] for firm in self.firms]).cpu().numpy()
            )
        percent_to_buy, percent_to_sale, percent_to_use, prices = torch.split(
            actions_concatenated.cpu(), self.actions_split_sizes
        )
        policy_info = {
            "percent_to_buy": percent_to_buy[:-1]
            .unflatten(-1, self.market.price_matrix.shape[1:])
            .numpy(),
            "percent_to_sale": percent_to_sale.numpy(),
            "percent_to_use": percent_to_use.numpy(),
            "prices": prices.numpy(),
        }

        self.state_history.append(state_info)
        self.actions_history[firm_id].append(policy_info)

    def process_rewards(self, firm_id, revenue, costs, cliprange=5):

        self.running_vars[firm_id]['cum_reward'] *= self.gamma
        self.running_vars[firm_id]['cum_reward'] += (revenue - costs).squeeze()
        var_rewards = self.update_moments(self.running_vars[firm_id]['cum_reward'], 'reward', firm_id)[1]
        sqrt_reward = var_rewards.sqrt().unsqueeze(dim=-1)
        revenue = torch.clip(revenue / sqrt_reward, -5, 5)
        costs = torch.clip(costs / sqrt_reward, -5, 5)
        return revenue, costs