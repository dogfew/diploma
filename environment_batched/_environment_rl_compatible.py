import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from tensordict import TensorDictBase, TensorDict

from .market import BatchedMarket
from .firm import BatchedFirm
from .utils import get_state_log, get_state_dim, get_action_dim, process_actions
from torchrl.envs import EnvBase


class EnvironmentTorchRL(EnvBase):
    def __init__(self, market_kwargs, prod_functions, device="cuda"):
        super().__init__(
            device=device,
            dtype=...,
            batch_size=...,
        )

        self.market = BatchedMarket(**market_kwargs, device=device)
        self.state_size = get_state_dim(self.market)
        self.action_size = get_action_dim(self.market, limit=False)
        self.firms = [
            BatchedFirm(fun, self.market, device=device) for fun in prod_functions
        ]
        self.episode = 0

    @property
    def state(self):
        return torch.stack([get_state_log(self.market, firm) for firm in self.firms])

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"]  # информация о действиях есть для всех фирм
        firm_id = self.episode % len(self.firms)  # ходит по факту только одна фирма
        firm = self.firms[firm_id]
        processed_actions = map(
            lambda x: x.to(self.market.device),
            process_actions(action, self.market.price_matrix.shape),
        )

        revenue, costs = firm.step(*processed_actions)

        out_tensordict = TensorDict(
            dict(
                state=self.state,
                reward=revenue - costs,
                firm_id=torch.full(tensordict.batch_size, firm_id),
                done=False,
            ),
            batch_size=tensordict.batch_size,
        )

        return out_tensordict

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.market.reset()
        self.episode = 0
        for firm in self.firms:
            firm.reset()
        out_tensordict = TensorDict({}, batch_size=torch.Size())
        out_tensordict.set("observation", self.state)
        return out_tensordict

    def _set_seed(self, seed: Optional[int]):
        pass

    def set(self, market, firms):
        self.firms = firms
        self.market = market

    @torch.no_grad()
    def step(self, firm_id) -> tuple[torch.Tensor]:
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
            lambda x: x.to(self.market.device),
            process_actions(actions, self.market.price_matrix.shape),
        )
        revenue, costs = firm.step(*processed_actions)
        self.revenues[firm_id].append(revenue)
        self.costs[firm_id].append(costs.item())
        return state, actions_concatenated, log_probs_concatenated, revenue, costs
