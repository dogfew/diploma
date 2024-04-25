import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

from enviroment import BaseFirm, LimitFirm
from models.utils import process_actions
from models.utils.preprocessing import get_state_log, get_state_dim


class Environment:
    def __init__(
            self,
            market,
            policy_class,
            prod_functions,
            invest_functions=None,
    ):
        self.market = market
        self.limit = invest_functions is not None
        if invest_functions is None:
            self.firms = [BaseFirm(fun, self.market) for fun in prod_functions]
        else:
            self.firms = [
                LimitFirm(fun, inv_fun, self.market)
                for fun, inv_fun in zip(prod_functions, invest_functions)
            ]
        self.state_dim = get_state_dim(market, self.limit)
        init_kwargs = dict(hidden_dim=32,
                           state_dim=self.state_dim ,
                           n_branches=market.n_branches,
                           n_firms=market.n_firms,
                           limit=self.limit)
        self.policies = [policy_class(**init_kwargs) for _ in self.firms]
        self.target_policies = [deepcopy(policy) for policy in self.policies]
        self.limit = invest_functions is not None

        # Record Dynamics of Environment
        self.state_history = []
        self.actions_history = {firm.id: [] for firm in self.firms}

        self.actions_split_sizes = [
            math.prod(self.market.price_matrix.shape) + 1,  # percent to buy
            self.market.price_matrix.shape[1],  # percent to sale
            self.market.price_matrix.shape[1] + self.limit * self.market.price_matrix.shape[1],  # percent to use
            self.market.price_matrix.shape[1]  # price change
        ]

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

    def set(self, market, firms):
        self.firms = firms
        self.market = market

    @torch.no_grad()
    def step(self, firm_id) -> tuple[[torch.Tensor]]:
        """
        :param firm_id:
        :return: state, actions_concatenated, log_probs_concatenated, revenue, costs
        """
        firm = self.firms[firm_id]
        policy = self.policies[firm_id]

        state = get_state_log(self.market, firm).to(policy.device)
        actions, log_probs = policy(state)

        actions_concatenated = torch.concatenate(actions, dim=-1)
        log_probs_concatenated = torch.concatenate(log_probs, dim=-1)
        processed_actions = map(
            lambda x: x.to(self.market.device), process_actions(actions, self.market.price_matrix.shape)
        )
        revenue, costs = firm.step(*processed_actions)
        return state, actions_concatenated, log_probs_concatenated, revenue, costs

    @torch.no_grad()
    def step_and_record(self, firm_id):
        state, actions_concatenated, log_probs_concatenated, revenue, costs = self.step(firm_id)
        state_info = {'price_matrix': self.market.price_matrix.cpu().numpy(),
                      'volume_matrix': self.market.volume_matrix.cpu().numpy(),
                      'finance': self.market.gains.cpu().numpy() + np.array(
                          [firm.financial_resources.item() for firm in self.firms]),
                      'reserves': torch.stack([firm.reserves for firm in self.firms]).cpu().numpy()}
        if self.limit:
            state_info['limits'] = torch.tensor([firm.limit for firm in self.firms]).cpu().numpy()
        percent_to_buy, percent_to_sale, percent_to_use, prices = torch.split(actions_concatenated.cpu(),
                                                                              self.actions_split_sizes)
        if percent_to_use.shape[0] == 2 * self.market.n_branches:
            percent_to_use = percent_to_use.numpy()
            percent_to_use[self.market.n_branches:] *= (1 - percent_to_use[:self.market.n_branches])
        policy_info = {'percent_to_buy': percent_to_buy[:-1].unflatten(-1, self.market.price_matrix.shape).numpy(),
                       'percent_to_sale': percent_to_sale.numpy(),
                       'percent_to_use': percent_to_use,
                       'prices': prices.numpy()}

        self.state_history.append(state_info)
        self.actions_history[firm_id].append(policy_info)
