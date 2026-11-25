from typing import Literal

import torch


class BatchedMarket:
    def __init__(
        self,
        n_branches: int = 2,
        min_price: int = 1,
        base_price: int = 100,
        start_gains: int = 250,
        start_volumes: int = 1,
        deprecation_steps: int = 2,
        batch_size=1,
        max_price: int | None = None,
        *,
        interval_prices=False,
        dtype: Literal["int32", "int64"] = "int64",
        device="cuda",
    ):
        """
        :param n_branches: Число отраслей
        """
        self.dtype = torch.int32 if dtype == "int32" else torch.int64
        self.interval_prices = interval_prices
        self.batch_size = batch_size
        self.n_branches: int = n_branches
        self.volume_matrix: torch.Tensor = torch.zeros(
            (batch_size, 0, n_branches),
            dtype=self.dtype,
            device=device,
        )
        self.price_matrix: torch.Tensor = torch.empty(
            (batch_size, 0, n_branches),
            dtype=self.dtype,
            device=device,
        )
        self.gains: torch.Tensor = torch.zeros((batch_size, 0), dtype=self.dtype, device=device)

        self.is_blind = False

        # Prices
        self.min_price = min_price
        self.base_price = base_price
        self._max_price = max_price

        self.start_gains: int = start_gains
        self.start_volumes: int = start_volumes
        self.max_id = 0
        self.deprecation_steps = deprecation_steps
        self.device = device

        # Calculated rewards for social planner
        self.rewards: torch.Tensor = torch.zeros((batch_size, 0), dtype=self.dtype, device=device)
        self.prices_from_social_planner = torch.full(
            (batch_size, n_branches),
            fill_value=base_price,
            dtype=self.dtype,
            device=device,
        )

    def change_batch_size(self, batch_size):
        self.volume_matrix = self.volume_matrix[:1].repeat(batch_size, 1, 1)
        self.price_matrix = self.price_matrix[:1].repeat(batch_size, 1, 1)
        self.gains = self.gains[:1].repeat(batch_size, 1)
        self.rewards = self.rewards[:1].repeat(batch_size, 1)
        self.prices_from_social_planner = self.prices_from_social_planner[:1].repeat(batch_size, 1)
        self.batch_size = batch_size
        return self

    @property
    def actual_volume_matrix(self):
        return self.volume_matrix

    @property
    def observable_volume_matrix(self):
        return self.volume_matrix

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
        self.prices_from_social_planner.fill_(min(self.base_price, self.max_price - 1))
        self.rewards.fill_(0)

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
            (
                self.gains,
                torch.zeros((self.batch_size, 1), dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        self.rewards = torch.cat(
            (
                self.rewards,
                torch.zeros((self.batch_size, 1), dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )

    def generate_id(self):
        self.__add_row()
        self.max_id += 1
        return self.max_id

    def process_purchases(self, money_distribution: torch.Tensor):
        purchase_matrix = torch.min(
            money_distribution // self.price_matrix,
            self.volume_matrix,
        ).type(self.dtype)
        sellers_gains = (purchase_matrix * self.price_matrix).sum(dim=2)
        total_cost = sellers_gains.sum(dim=-1, keepdim=True)
        delta_reserves = purchase_matrix.sum(dim=1)
        self.volume_matrix -= purchase_matrix
        self.gains += sellers_gains
        return total_cost, delta_reserves

    def process_sales(self, firm_id: int, volumes: torch.Tensor, prices: torch.Tensor | None = None):
        if prices is None:
            prices = self.prices_from_social_planner
        self.volume_matrix[:, firm_id] += volumes
        self.price_matrix[:, firm_id] = torch.clip(prices, self.min_price, self.max_price)

    def process_gains(self, firm_id: int):
        firms_gain = self.gains[:, firm_id].clone()
        self.gains[:, firm_id] -= firms_gain
        return firms_gain

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_firms: {self.n_firms}, n_commodities: {self.n_branches})"
            f"\n\tVolume Matrix ({self.volume_matrix.shape}):\n{self.volume_matrix}"
            f"\n\tPrice Matrix ({self.price_matrix.shape}):\n{self.price_matrix}"
            f"\n\tGain Matrix:\n{self.gains}"
        )


class BlindMarket(BatchedMarket):
    def __init__(
        self,
        n_branches: int = 2,
        start_gains: int = 250,
        start_volumes: int = 1,
        deprecation_steps: int = 2,
        batch_size=1,
        *,
        min_price: int = 1,
        max_price: int | None = None,
        base_price: int = 100,
        interval_prices=False,
        dtype: Literal["int32", "int64"] = "int64",
        device="cuda",
    ):
        super().__init__(
            n_branches=n_branches,
            min_price=min_price,
            base_price=base_price,
            start_gains=start_gains,
            start_volumes=start_volumes,
            deprecation_steps=deprecation_steps,
            batch_size=batch_size,
            max_price=max_price,
            interval_prices=interval_prices,
            dtype=dtype,
            device=device,
        )
        self.gains: torch.Tensor = torch.zeros((batch_size, 0), dtype=self.dtype, device=device)
        self._max_price: int = max_price or max(start_gains, int(base_price))
        self.price_matrix.fill_(min_price)
        self.volume_matrix: torch.Tensor = torch.zeros(
            (batch_size, 0, n_branches, self._max_price),
            dtype=self.dtype,
            device=device,
        )
        self.prices_vec = torch.arange(
            start=1,
            end=self.max_price + 1,
            dtype=self.dtype,
            device=self.device,
        ).reshape(1, 1, self.max_price)
        self.is_blind = True
        self.prices_from_social_planner = torch.empty(
            (batch_size, n_branches),
            dtype=self.dtype,
            device=device,
        )

    def change_batch_size(self, batch_size):
        self.volume_matrix = self.volume_matrix[:1].repeat(batch_size, 1, 1, 1)
        self.price_matrix = self.price_matrix[:1].repeat(batch_size, 1, 1)
        self.gains = self.gains[:1].repeat(batch_size, 1)
        self.rewards = self.rewards[:1].repeat(batch_size, 1)
        self.prices_from_social_planner = self.prices_from_social_planner[:1].repeat(batch_size, 1)
        self.batch_size = batch_size
        return self

    @property
    def max_price(self):
        return self._max_price

    def reset(self):
        super().reset()
        self.price_matrix.fill_(self.min_price)

    @property
    def observable_volume_matrix(self):
        # [B, C, log2 P]
        return self.volume_matrix.sum(dim=1)
        # seg = torch.log2(torch.arange(self.max_price, device=self.device).add(1)).type(self.dtype)
        # n_buckets = int(seg.max()) + 1
        # v = self.volume_matrix.sum(dim=1)  # [BS, N, P]
        # out = torch.zeros(*v.shape[:-1], n_buckets, dtype=v.dtype, device=self.device)
        # return out.index_add_(-1, seg, v).float()

    @property
    def actual_volume_matrix(self):
        return self.volume_matrix.sum(dim=-1)

    def __add_row(self) -> None:
        """
        Добавить строки в v_matrix, p_matrix и gains
        """
        shape = (self.batch_size, 1, self.n_branches)
        shape_v = (self.batch_size, 1, self.n_branches, self.max_price)
        self.volume_matrix = torch.cat(
            (
                self.volume_matrix,
                torch.zeros(shape_v, dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        self.price_matrix = torch.cat(
            (
                self.price_matrix,
                torch.zeros(shape, dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        self.gains = torch.cat(
            (
                self.gains,
                torch.zeros((self.batch_size, 1), dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        self.rewards = torch.cat(
            (
                self.rewards,
                torch.zeros((self.batch_size, 1), dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        # self.prices_from_social_planner = torch.cat(
        #     (
        #         self.prices_from_social_planner,
        #         torch.zeros((self.batch_size, 1), dtype=self.dtype, device=self.device),
        #     ),
        #     dim=1,
        # )

    def generate_id(self):
        self.__add_row()
        self.max_id += 1
        return self.max_id

    # @torch.compile(dynamic=False)
    @torch.no_grad()
    def process_sales(self, firm_id: int, volumes: torch.Tensor, prices: torch.Tensor | None = None):
        # Volume Matrix: [BS, n_firms, n_branches, n_prices]
        # Volumes: [BS, n_branches]
        # Prices [BS, n_branches, 2]
        # print(f"{volumes.shape=}, {prices.shape=}, {self.volume_matrix.shape=}")
        if prices is None:
            prices = self.prices_from_social_planner
        if self.interval_prices:
            prices = prices.reshape(self.batch_size, self.n_branches, 3)
            prices_min, prices_max = prices[..., 0:1], prices[..., 0:1] + prices[..., 1:2]
            idx = torch.arange(self.max_price, dtype=self.dtype, device=self.device)
            prices_delta = prices_max - prices_min + 1
            volumes = volumes.unsqueeze(-1)
            rank = idx - prices_min
            mask = (idx >= prices_min) & (idx <= prices_max)
            contrib = mask * ((rank * volumes + volumes) // prices_delta - (rank * volumes) // prices_delta)
            self.volume_matrix[:, firm_id] += contrib
        else:
            price_idx = prices
            self.volume_matrix[:, firm_id].scatter_add_(
                -1,
                price_idx.long().unsqueeze(-1),
                volumes.to(self.volume_matrix.dtype).unsqueeze(-1),
            )
        firm_volume = self.volume_matrix[:, firm_id]
        # self.price_matrix[:, firm_id] = (firm_volume * self.prices_vec).sum(dim=-1) // firm_volume.sum(
        #     -1,
        # ).clamp(min=1)
        self.price_matrix[:, firm_id] = torch.where(firm_volume > 0, self.prices_vec, self.max_price).amin(
            dim=2,
        )

    # @torch.compile(dynamic=False)
    @torch.no_grad()
    def process_purchases(self, money_distribution: torch.Tensor):
        # self.volume_matrix: [BS, N, C, P]
        # money_distribution: [BS, C]
        blind_volumes = self.volume_matrix.sum(dim=1)  # [BS, B, P]
        price = self.prices_vec
        cost = blind_volumes * price
        bought = torch.minimum(
            blind_volumes,
            (money_distribution.unsqueeze(-1) - (cost.cumsum(dim=-1) - cost)).clamp(min=0) // price,
        )
        partial = (self.volume_matrix[:, :-1].cumsum(dim=1) * bought.unsqueeze(1)) // blind_volumes.clamp_min(
            1,
        ).unsqueeze(1)
        # alloc = (self.volume_matrix.cumsum(dim=1) * bought.unsqueeze(1)) // blind_volumes.clamp(
        #     min=1,
        # ).unsqueeze(1)
        # purchase_matrix = alloc.diff(dim=1, prepend=torch.zeros_like(alloc[:, :1]))
        alloc = torch.cat((partial, bought.unsqueeze(1)), dim=1)
        # alloc_old = (self.volume_matrix.cumsum(dim=1) * bought.unsqueeze(1)) // blind_volumes.clamp(
        #     min=1,
        # ).unsqueeze(1)
        # assert torch.equal(alloc, alloc_old), "alloc mismatch"
        delta_reserves = bought.sum(dim=-1)
        total_cost = (bought * price).sum(dim=(1, 2)).unsqueeze(-1)  # [BS, 1]
        purchase_matrix = alloc.diff(dim=1, prepend=torch.zeros_like(alloc[:, :1]))
        sellers_gains = (purchase_matrix * price).sum(dim=(2, 3))  # [BS, F]
        self.volume_matrix -= purchase_matrix
        self.gains += sellers_gains
        # assert (self.volume_matrix >= 0).all() and (self.gains >= 0).all(), "overflow"  # noqa: PT018
        return total_cost, delta_reserves
