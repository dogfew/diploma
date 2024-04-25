from copy import deepcopy

import torch
import torch.nn as nn
from typing import Callable


class BatchedMarket:
    def __init__(
            self,
            n_branches: int = 2,
            min_price: float = 1,
            base_price: float = 100,
            start_gains: int = 250,
            start_volumes: int = 1,
            deprecation_steps: int = 2,
            batch_size=1,
            max_price: int = None,
            dtype=torch.int64,
            device='cuda'
    ):
        """
        :param n_branches: Число отраслей
        """
        self.batch_size = batch_size
        self.n_branches: int = n_branches
        self.dtype = dtype
        self.volume_matrix: torch.Tensor = torch.zeros(
            (batch_size, 0, n_branches), dtype=self.dtype, device=device
        )
        self.price_matrix: torch.Tensor = torch.empty((batch_size, 0, n_branches), dtype=self.dtype, device=device)
        self.gains: torch.Tensor = torch.zeros((batch_size, 0), dtype=self.dtype, device=device)

        # Prices
        self.min_price = min_price
        self.base_price = base_price
        self._max_price = max_price

        self.start_gains = start_gains
        self.start_volumes = start_volumes
        self.max_id = 0
        self.deprecation_steps = deprecation_steps
        self.device = device

    def change_batch_size(self, batch_size):
        self.volume_matrix = self.volume_matrix[:1].repeat(batch_size, 1, 1)
        self.price_matrix = self.price_matrix[:1].repeat(batch_size, 1, 1)
        self.gains = self.gains[:1].repeat(batch_size, 1)
        self.batch_size = batch_size
        return self

    @property
    def max_price(self):
        if self._max_price is not None:
            return self._max_price
        return self.start_gains * self.n_firms

    @property
    def n_firms(self):
        return self.max_id

    def reset(self):
        self.price_matrix.fill_(self.base_price)
        self.gains.fill_(0)  # self.start_gains
        self.volume_matrix.fill_(0)  # self.start_volumes

    def __add_row(self) -> None:
        """
        Добавить строки в v_matrix, p_matrix и gains
        """
        shape = (self.batch_size, 1, self.n_branches)
        self.volume_matrix = torch.cat(
            (
                self.volume_matrix,
                torch.zeros(shape, dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        self.price_matrix = torch.cat(
            (
                self.price_matrix,
                torch.full(shape, self.base_price, dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        self.gains = torch.cat(
            (self.gains, torch.zeros((self.batch_size, 1),
                                     dtype=self.dtype, device=self.device)),
            dim=1,
        )

    def generate_id(self):
        self.__add_row()
        self.max_id += 1
        return self.max_id

    def process_purchases(
            self, purchase_matrix: torch.Tensor, sellers_gains: torch.Tensor
    ):
        self.volume_matrix -= purchase_matrix
        self.gains += sellers_gains

    def process_sales(self, firm_id: int, volumes: torch.Tensor):
        self.volume_matrix[:, firm_id] += volumes

    def process_gains(self, firm_id: int):
        firms_gain = self.gains[:, firm_id].clone()
        self.gains[:, firm_id] -= firms_gain
        return firms_gain

    def process_prices(self, firm_id: int, new_prices: torch.Tensor):
        self.price_matrix[:, firm_id] = torch.clip(
            new_prices, self.min_price, self.max_price
        )

    def __repr__(self):
        representation = (
            f"BatchedMarket(n_firms: {self.n_firms}, n_commodities: {self.n_branches})"
            f"\n\tVolume Matrix:\n{self.volume_matrix}"
            f"\n\tPrice Matrix:\n{self.price_matrix}"
            f"\n\tGain Matrix:\n{self.gains}"
        )
        return representation


if __name__ == '__main__':
    market = BatchedMarket(64)
    market.generate_id()
    market.generate_id()
    market.generate_id()

    print(market.change_batch_size(16))
    # Всё ок
