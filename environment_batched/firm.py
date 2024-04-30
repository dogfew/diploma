import torch
from copy import deepcopy

from environment_batched.utils import price_change_function


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
    ):
        """
        :param prod_function: Производственная функция
        :param market: Собственно, рынок
        :param batch_size:
        :param production_reg: Коэффициент регуляризации для производства
        :attribute reserves: Объём запасов фирмы
        """
        self.batch_size = batch_size
        self.prod_function = prod_function
        self.market = market
        self.n_branches = self.market.n_branches
        self.reserves = torch.zeros(
            (self.batch_size, self.market.n_branches),
            dtype=self.market.dtype,
            device=self.device,
        )
        self.financial_resources = torch.zeros(
            (self.batch_size, 1), dtype=self.market.dtype, device=self.device
        )
        self.id = market.generate_id() - 1
        self.production_reg = production_reg

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

    def sell(self, percent_to_sale: torch.Tensor):
        """
        :param percent_to_sale: Процент резервов на продажу
        """
        assert torch.all((0 <= percent_to_sale) & (percent_to_sale <= 1))
        goods = (self.reserves * percent_to_sale).type(self.market.dtype)
        self.market.process_sales(self.id, goods)
        self.reserves -= goods

    def buy(self, percent_to_buy: torch.Tensor):
        """
        :param percent_to_buy: (n_firms, n_branches)
        Какой процент от финансовых ресурсов потратить на товар i для каждой j фирмы.
        """
        assert percent_to_buy.shape == self.market.volume_matrix.shape
        purchase_matrix = torch.min(
            percent_to_buy
            * self.financial_resources.unsqueeze(-1)
            // self.market.price_matrix,
            self.market.volume_matrix,
        ).type(self.market.dtype)
        sellers_gains = (purchase_matrix * self.market.price_matrix).sum(dim=2)
        total_cost = sellers_gains.sum(dim=-1, keepdim=True)
        new_reserves = purchase_matrix.sum(dim=1)
        self.financial_resources -= total_cost
        self.reserves += new_reserves
        self.market.process_purchases(purchase_matrix, sellers_gains)
        return total_cost

    def produce(self, percent_to_use: torch.Tensor):
        """
        :param percent_to_use: (n_branches)
        Какую долю резервов от каждого товара потратить для производство
        """
        input_reserves = (self.reserves * percent_to_use).round().type(torch.int64)
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_reserves = self.prod_function(input_reserves)
        self.reserves -= used_reserves
        self.reserves += new_reserves
        return used_reserves, new_reserves

    def define_prices(self, prices):
        """
        :param prices: (n_branches)
        """
        if prices is None:
            return
        new_prices = (prices * self.market.max_price).type(self.market.dtype)
        self.market.process_prices(self.id, new_prices)
        return
        # new_prices = price_change_function(
        #     self.market.price_matrix[:, self.id], prices
        # ).type(self.market.dtype)
        # self.market.process_prices(self.id, new_prices=new_prices)
        # return new_prices

    def receive_revenue(self):
        revenue = self.market.process_gains(self.id)[:, None]
        self.financial_resources += revenue
        return revenue

    def step(
            self,
            percent_to_buy: torch.Tensor,
            percent_to_sale: torch.Tensor,
            percent_to_use: torch.Tensor,
            prices: torch.Tensor = None,
    ) -> tuple[float, float]:
        """
        :param percent_to_buy: [B, n_firms, n_branches]
        :param percent_to_sale: [B, n_branches]
        :param percent_to_use: [B, n_branches]
        :param prices: [B, n_branches]
        :return: revenue, costs
        """

        revenue = self.receive_revenue()
        costs = self.buy(percent_to_buy)
        _, produced = self.produce(percent_to_use)
        self.define_prices(prices)
        self.sell(percent_to_sale)
        if self.production_reg > 0:
            revenue = revenue.type(torch.float64)
            costs = costs.type(torch.float64)
            costs -= (produced.sum(dim=1, keepdims=True) + 0.5).log() * self.production_reg
        return revenue, costs

    def __repr__(self):
        representation = (
            f"Firm id: {self.id}"
            f"\nReserves: {self.reserves.tolist()}"
            f"\nFinance: {self.financial_resources}"
        )
        return representation

    def copy(self):
        return deepcopy(self.reserves), deepcopy(self.financial_resources)

    def set(self, reserves, financial_resources):
        self.reserves = reserves
        self.financial_resources = financial_resources


class BatchedLimitFirm(BatchedFirm):
    """
    Firm with fixed Capital
    """

    def __init__(
            self,
            prod_function,
            invest_function,
            market,
            batch_size=1,
            start_capital=2,
            production_reg=0.1,
            is_deprecating=True,
    ):
        super().__init__(
            prod_function=prod_function,
            market=market,
            batch_size=batch_size,
            production_reg=production_reg
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
        self.production_regularization = production_reg
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

    def produce(self, input_reserves: torch.Tensor):
        """
        :param input_reserves: (n_branches)
        Какую долю резервов от каждого товара потратить для производство
        """
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_reserves = self.prod_function(
            input_reserves, limit=self.limit
        )
        self.reserves -= used_reserves
        self.reserves += new_reserves
        return used_reserves, new_reserves

    def step(
            self,
            percent_to_buy: torch.Tensor,
            percent_to_sale: torch.Tensor,
            percent_to_use: torch.Tensor,
            prices: torch.Tensor = None,
    ) -> tuple[float, float]:
        """
        :param percent_to_buy: [n_firms, n_branches]
        :param percent_to_sale: [n_branches]
        :param percent_to_use: [2 * n_branches]
        :param prices: [n_branches]
        :return: revenue, costs
        """
        percent_to_use_prod, percent_to_use_invest = torch.split(
            percent_to_use, [self.n_branches, self.n_branches], dim=1
        )
        input_prod = (self.reserves * percent_to_use_prod).round().type(torch.int64)
        input_invest = (self.reserves * percent_to_use_invest).round().type(torch.int64)
        revenue = self.receive_revenue()
        costs = self.buy(percent_to_buy)
        _, new_limits = self.invest(input_invest)
        _, produced = self.produce(input_prod)
        self.define_prices(prices)
        self.sell(percent_to_sale)

        if self.production_reg > 0:
            revenue = revenue.type(torch.float64)
            costs = costs.type(torch.float64)
            costs -= (produced.sum(dim=1, keepdims=True) + 0.5).log() * self.production_reg
        return revenue, costs

    def __repr__(self):
        representation = (
            f"Firm id: {self.id}"
            f"\nReserves: {self.reserves.tolist()}"
            f"\nFinance: {self.financial_resources}"
            f"\nLimit: {self.limit}"
            f"\nCapital: {self.capital}"
        )
        return representation


class BatchedLimitProductionFirm(BatchedLimitFirm):
    """
    It's the same firm, however only cares about production volumes.
    """

    def step(
            self,
            percent_to_buy: torch.Tensor,
            percent_to_sale: torch.Tensor,
            percent_to_use: torch.Tensor,
            prices: torch.Tensor = None,
    ) -> tuple[float, float]:
        """
        :param percent_to_buy: [n_firms, n_branches]
        :param percent_to_sale: [n_branches]
        :param percent_to_use: [2 * n_branches]
        :param prices: [n_branches]
        :return: revenue, costs
        """
        percent_to_use_prod, percent_to_use_invest = torch.split(
            percent_to_use, [self.n_branches, self.n_branches], dim=1
        )
        # print(percent_to_use_invest.shape)
        # exit()
        input_prod = (self.reserves * percent_to_use_prod).round().type(torch.int64)
        input_invest = (self.reserves * percent_to_use_invest).round().type(torch.int64)
        self.receive_revenue()
        self.buy(percent_to_buy)
        used_reserves_invest, new_limits = self.invest(input_invest)
        used_reserves_produce, new_reserves = self.produce(input_prod)
        self.define_prices(prices)
        self.sell(percent_to_sale)
        revenue = new_reserves.sum(dim=1, keepdims=True)
        revenue -= used_reserves_produce.sum(dim=1, keepdims=True)
        costs = torch.zeros_like(revenue)
        return costs, -(revenue + 0.5).log()



class BatchedProductionFirm(BatchedFirm):
    """
    It's the same firm, however only cares about production volumes.
    """

    def step(
            self,
            percent_to_buy: torch.Tensor,
            percent_to_sale: torch.Tensor,
            percent_to_use: torch.Tensor,
            prices: torch.Tensor = None,
    ) -> tuple[float, float]:
        """
        :param percent_to_buy: [n_firms, n_branches]
        :param percent_to_sale: [n_branches]
        :param percent_to_use: [2 * n_branches]
        :param prices: [n_branches]
        :return: revenue, costs
        """
        input_prod = (self.reserves * percent_to_use).round().type(torch.int64)
        self.receive_revenue()
        self.buy(percent_to_buy)
        used_reserves_produce, new_reserves = self.produce(input_prod)
        self.define_prices(prices)
        self.sell(percent_to_sale)
        revenue = new_reserves.sum(dim=1, keepdims=True)
        revenue -= used_reserves_produce.sum(dim=1, keepdims=True)
        costs = torch.zeros_like(revenue)
        return costs, -(revenue + 0.5).log()
