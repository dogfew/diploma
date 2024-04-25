import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from tensordict import TensorDictBase, TensorDict

from .market import BatchedMarket
from .firm import BatchedFirm, BatchedLimitFirm, BatchedLimitProductionFirm
from .utils import get_state_log, get_state_dim, get_action_dim, process_actions


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
        hidden_dim=32,
        target='finance',
        device="cuda",
        invest_functions=None,
    ):
        self.batch_size = batch_size
        self.limit = invest_functions is not None

        # Market
        self.market = BatchedMarket(
            **market_kwargs, batch_size=batch_size, device=device
        )

        # Firms
        if invest_functions is None:
            self.firms = [
                BatchedFirm(fun, self.market, batch_size=batch_size)
                for fun in prod_functions
            ]
        else:
            firm_class = {'production': BatchedLimitProductionFirm,
                          'finance': BatchedLimitFirm}[target]
            self.firms = [
                firm_class(fun, inv_fun, self.market, batch_size=batch_size)
                for fun, inv_fun in zip(prod_functions, invest_functions)
            ]

        self.state_dim = get_state_dim(self.market, self.limit)
        self.action_dim = get_action_dim(self.market, self.limit)
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

    @torch.no_grad()
    def step(self, firm_id) -> tuple[[torch.Tensor]]:
        """
        :param firm_id:
        :return: state, actions_concatenated, log_probs_concatenated, revenue, costs
        """
        firm = self.firms[firm_id]
        policy = self.policies[firm_id]

        state = get_state_log(self.market, firm)
        actions, log_probs = policy(state)
        actions_concatenated = torch.concatenate(actions, dim=-1)
        log_probs_concatenated = torch.concatenate(log_probs, dim=-1)
        processed_actions = process_actions(actions, self.market.price_matrix.shape)
        revenue, costs = firm.step(*processed_actions)
        return state, actions_concatenated, log_probs_concatenated, revenue, costs

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
