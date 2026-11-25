import math
from typing import TYPE_CHECKING

import torch

from src.environment.prod_functions import CobbDouglas, MoneyPrintingLeontief
from src.environment.utils import price_change_function

if TYPE_CHECKING:
    from src.environment.market import BatchedMarket
    from src.environment.prod_functions import BatchedLeontief

REVENUE_DTYPE = torch.float32


class BatchedFirm:
    """
    Firm without Fixed Capital
    """

    def __init__(
        self,
        prod_function,
        market,
        batch_size=1,
        production_reg=0.1,
        *,
        social_planner=False,
    ):
        """
        :param prod_function: Производственная функция
        :param market: Собственно, рынок
        :param batch_size:
        :param production_reg: Коэффициент регуляризации для производства
        :attribute reserves: Объём запасов фирмы
        """
        self.batch_size = batch_size
        self.prod_function: BatchedLeontief | CobbDouglas | MoneyPrintingLeontief = prod_function
        self.market: BatchedMarket = market
        self.n_branches = self.market.n_branches
        self.reserves = torch.zeros(
            (self.batch_size, self.market.n_branches),
            dtype=self.market.dtype,
            device=self.device,
        )
        self.financial_resources = torch.zeros(
            (self.batch_size, 1),
            dtype=self.market.dtype,
            device=self.device,
        )
        self.id = market.generate_id() - 1
        self.production_reg = production_reg
        self.social_planner = social_planner

    @property
    def limit(self):
        return None

    @property
    def device(self):
        return self.market.device

    def change_batch_size(self, batch_size):
        self.reserves = self.reserves[:1].repeat(batch_size, 1)
        self.financial_resources = self.financial_resources[:1].repeat(batch_size, 1)
        self.batch_size = batch_size

    def reset(self):
        self.reserves.fill_(self.market.start_volumes)
        self.financial_resources.fill_(self.market.start_gains)

    def sell(self, percent_to_sale: torch.Tensor, prices: torch.Tensor | None):
        """
        :param percent_to_sale: Процент резервов на продажу
        """
        assert torch.all((percent_to_sale >= 0) & (percent_to_sale <= 1))
        amounts = (self.reserves * percent_to_sale).type(self.market.dtype)
        new_prices = self.define_prices(prices)
        self.market.process_sales(self.id, amounts, new_prices)
        self.reserves -= amounts

    def buy(self, percent_to_buy: torch.Tensor):
        """
        :param percent_to_buy: (n_firms, n_branches)
        Какой процент от финансовых ресурсов потратить на товар i для каждой j фирмы.
        """
        if not self.market.is_blind:
            want_purchase_matrix = (
                percent_to_buy.reshape(
                    self.market.volume_matrix.shape,
                )
                .mul(self.financial_resources.unsqueeze(-1))
                .to(self.market.dtype)
            )
        else:
            want_purchase_matrix = percent_to_buy.mul(self.financial_resources).to(self.market.dtype)
        total_cost, delta_reserves = self.market.process_purchases(money_distribution=want_purchase_matrix)
        self.financial_resources -= total_cost
        self.reserves += delta_reserves
        return total_cost

    def produce(self, input_reserves: torch.Tensor):
        """
        :param input_reserves: (n_branches)
        Какую долю резервов от каждого товара потратить для производство
        """
        used_reserves, new_reserves = self.prod_function(
            torch.min(input_reserves, self.reserves),
            limit=self.limit,
        )
        self.reserves -= used_reserves
        self.reserves += new_reserves
        return used_reserves, new_reserves

    def define_prices(self, prices):
        """
        :param prices: (n_branches)
        """
        if prices is None or self.social_planner:
            return None
        return (prices * self.market.max_price).type(
            self.market.dtype,
        )

    def define_prices_percent(self, percent):
        return price_change_function(self.market.price_matrix[:, self.id], percent).type(
            self.market.dtype,
        )

    def receive_revenue(self):
        revenue = self.market.process_gains(self.id)[:, None]
        if isinstance(self.prod_function, MoneyPrintingLeontief):
            self.financial_resources += self.prod_function.money_delta
        self.financial_resources += revenue
        return revenue

    def step(
        self,
        percent_to_buy: torch.Tensor,
        percent_to_sale: torch.Tensor,
        percent_to_use: torch.Tensor,
        prices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param percent_to_buy: [B, n_firms, n_branches]
        :param percent_to_sale: [B, n_branches]
        :param percent_to_use: [B, n_branches]
        :param prices: [B, n_branches]
        :return: revenue, costs
        """

        revenue = self.receive_revenue()
        costs = self.buy(percent_to_buy)
        input_prod = (self.reserves * percent_to_use).round().type(self.market.dtype)
        _, produced = self.produce(input_prod)
        self.sell(percent_to_sale, prices)
        if self.production_reg > 0:
            revenue = revenue.type(REVENUE_DTYPE)
            costs = costs.type(REVENUE_DTYPE)
            costs -= (produced.sum(dim=1, keepdims=True) + 0.5).log() * self.production_reg
        return revenue, costs

    def __repr__(self):
        return f"Firm id: {self.id}\nReserves: {self.reserves.tolist()}\nFinance: {self.financial_resources}"


class BatchedLimitFirm(BatchedFirm):
    """
    Firm with fixed Capital
    """

    def __init__(
        self,
        prod_function,
        invest_function,
        market,
        *,
        batch_size=1,
        start_capital=2,
        production_reg=0.1,
        is_deprecating=True,
        social_planner=False,
    ):
        super().__init__(
            prod_function=prod_function,
            market=market,
            batch_size=batch_size,
            production_reg=production_reg,
            social_planner=social_planner,
        )
        deprecation_steps = market.deprecation_steps
        self.invest_function = invest_function
        self.deprecation_steps = deprecation_steps  # How much time Fixed Capital exists
        self.capital = torch.zeros(
            (batch_size, deprecation_steps + 1),
            device=self.device,
            dtype=self.market.dtype,
        )
        self.start_capital = start_capital
        self.current_step = 1
        self.capital[:, 0] = start_capital
        self.is_deprecating = is_deprecating

    def change_batch_size(self, batch_size):
        super().change_batch_size(batch_size)
        self.capital = self.capital[:1].repeat(batch_size, 1)

    def reset(self):
        super().reset()
        self.capital.fill_(0)
        self.capital[:, 0] = self.start_capital
        self.current_step = 1

    @property
    def limit(self):
        return self.capital.sum(dim=1, keepdim=True)

    def invest(self, input_reserves: torch.Tensor):
        """
        :param input_reserves: (n_branches)
        Какую долю резервов от каждого товара потратить на инвестициии
        """
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_limits = self.invest_function(input_reserves)
        self.reserves -= used_reserves
        self.capital[:, self.current_step] = new_limits.flatten()
        self.current_step += 1
        self.current_step %= self.deprecation_steps + 1
        return used_reserves, new_limits

    def step(
        self,
        percent_to_buy: torch.Tensor,
        percent_to_sale: torch.Tensor,
        percent_to_use: torch.Tensor,
        prices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param percent_to_buy: [n_firms, n_branches]
        :param percent_to_sale: [n_branches]
        :param percent_to_use: [2 * n_branches]
        :param prices: [n_branches]
        :return: revenue, costs
        """
        percent_to_use_prod = percent_to_use[:, :, 0]
        percent_to_use_invest = percent_to_use[:, :, 1]
        input_prod = (self.reserves * percent_to_use_prod).round().type(self.market.dtype)
        input_invest = (self.reserves * percent_to_use_invest).round().type(self.market.dtype)
        revenue = self.receive_revenue()
        costs = self.buy(percent_to_buy)
        _, _new_limits = self.invest(input_invest)
        _, produced = self.produce(input_prod)
        self.sell(percent_to_sale, prices)

        if self.production_reg > 0:
            revenue = revenue.type(REVENUE_DTYPE)
            costs = costs.type(REVENUE_DTYPE)
            costs -= (produced.sum(dim=1, keepdims=True) + 0.5).log() * self.production_reg
        self.market.rewards[:, self.id] = revenue
        return revenue, costs

    def __repr__(self):
        return (
            f"Firm id: {self.id}"
            f"\nReserves: {self.reserves.tolist()}"
            f"\nFinance: {self.financial_resources}"
            f"\nLimit: {self.limit}"
            f"\nCapital: {self.capital}"
        )


class BatchedLimitProductionFirm(BatchedLimitFirm):
    """
    It's the same firm, however only cares about production volumes.
    """

    def step(
        self,
        percent_to_buy: torch.Tensor,
        percent_to_sale: torch.Tensor,
        percent_to_use: torch.Tensor,
        prices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param percent_to_buy: [B, n_firms, n_branches]
        :param percent_to_sale: [B, n_branches]
        :param percent_to_use: [B, 2 * n_branches]
        :param prices: [B, n_branches]
        :return: revenue, costs
        """
        percent_to_use_prod = percent_to_use[:, :, 0]
        percent_to_use_invest = percent_to_use[:, :, 1]
        input_prod = (self.reserves * percent_to_use_prod).round().type(self.market.dtype)
        input_invest = (self.reserves * percent_to_use_invest).round().type(self.market.dtype)
        self.receive_revenue()
        self.buy(percent_to_buy)
        _used_reserves_invest, _new_limits = self.invest(input_invest)
        _used_reserves_produce, new_reserves = self.produce(input_prod)
        self.sell(percent_to_sale, prices)
        revenue = new_reserves.sum(dim=1, keepdims=True)
        costs = torch.zeros_like(revenue, dtype=torch.float32)
        self.market.rewards[:, self.id] = revenue.flatten()
        return costs, -(revenue).log1p() / math.log(20)
        # return costs, -revenue / self.market.n_branches


class BatchedProductionFirm(BatchedFirm):
    """
    It's the same firm, however only cares about production volumes.
    """

    def step(
        self,
        percent_to_buy: torch.Tensor,
        percent_to_sale: torch.Tensor,
        percent_to_use: torch.Tensor,
        prices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param percent_to_buy: [B, n_firms, n_branches]
        :param percent_to_sale: [B, n_branches]
        :param percent_to_use: [B, 2 * n_branches]
        :param prices: [B, n_branches]
        :return: revenue, costs
        """
        input_prod = (self.reserves * percent_to_use).round().type(self.market.dtype)
        self.receive_revenue()
        self.buy(percent_to_buy)
        _used_reserves_produce, new_reserves = self.produce(input_prod)
        self.sell(percent_to_sale, prices)
        revenue = new_reserves.sum(dim=1, keepdims=True)
        costs = torch.zeros_like(revenue, dtype=torch.float32)
        self.market.rewards[:, self.id] = revenue.flatten()
        return costs, -revenue.log1p() / math.log(10)
        # return costs, -revenue / self.market.n_branches


class SocialPlanner:
    def __init__(self, market):
        self.market: BatchedMarket = market
        self.n_branches = self.market.n_branches
        self.id = None

    @property
    def financial_resources(self):
        return torch.zeros((self.market.batch_size, 1), device=self.market.device, dtype=self.market.dtype)

    def reset(self): ...

    def change_batch_size(self, batch_size: int): ...

    def step(
        self,
        percent_to_buy: torch.Tensor,  # noqa: ARG002
        percent_to_sale: torch.Tensor,  # noqa: ARG002
        percent_to_use: torch.Tensor,  # noqa: ARG002
        prices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param percent_to_buy: [B, n_firms, n_branches]
        :param percent_to_sale: [B, n_branches]
        :param percent_to_use: [B, 2 * n_branches]
        :param prices: [B, n_branches]
        :return: revenue, costs
        """
        self.market.prices_from_social_planner = (
            (prices * self.market.max_price)
            .type(
                self.market.dtype,
            )
            .clamp(max=self.market.max_price - 1)
        )
        revenue = self.market.rewards.sum(dim=1, keepdim=True)
        costs = torch.zeros_like(revenue).float()
        return (
            costs,
            # -revenue / self.market.n_branches / self.market.n_firms,
            -(revenue / self.market.n_branches / self.market.n_firms + 1).log10(),
        )  # / self.market.n_branches / self.market.n_firms  # -(revenue + 0.5).log()
