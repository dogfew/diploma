import torch
from copy import deepcopy
from .market import Market
from .prod_functions import Leontief


class BaseFirm:
    def __init__(
        self,
        prod_function,
        market: Market,
        *args,
        **kwargs,
    ):
        """
        :param prod_function: Производственная функция
        :param invest_function: Инвестиционная функция
        :attribute reserves: Объём запасов фирмы
        """
        self.prod_function = prod_function
        self.market = market
        self.n_branches = self.market.n_branches
        self.reserves = torch.zeros(self.market.n_branches, dtype=self.market.dtype, device=market.device)
        self.financial_resources = torch.tensor(0, dtype=self.market.dtype, device=market.device)
        self.id = market.generate_id() - 1

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
        assert 0 - 1e-5 <= percent_to_buy.sum() <= 1 + 1e-5, percent_to_buy.sum()
        purchase_matrix = torch.min(
            percent_to_buy * self.financial_resources // self.market.price_matrix,
            self.market.volume_matrix,
        ).type(self.market.dtype)
        sellers_gains = (purchase_matrix * self.market.price_matrix).sum(dim=1)
        total_cost = sellers_gains.sum()
        new_reserves = purchase_matrix.sum(dim=0)
        self.financial_resources -= total_cost
        self.reserves += new_reserves
        self.market.process_purchases(purchase_matrix, sellers_gains)
        return total_cost

    def produce(self, percent_to_use: torch.Tensor):
        """
        :param percent_to_use: (n_branches)
        Какую долю резервов от каждого товара потратить для производство
        """
        assert torch.all((0 <= percent_to_use) & (percent_to_use <= 1))
        input_reserves = (self.reserves * percent_to_use).round().type(torch.int64)
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_reserves = self.prod_function(input_reserves)
        self.reserves -= used_reserves
        self.reserves += new_reserves

    def define_prices(self, prices):
        """
        :param prices: (n_branches)
        """
        if prices is None:
            return
        new_prices = (prices * self.market.max_price).type(self.market.dtype)
        self.market.process_prices(self.id, new_prices)  # если на вход цены

        # а это если на вход проценты
        # self.market.process_prices(
        #     self.id,
        #     new_prices=(self.market.price_matrix[self.id] * (prices + 0.5)).type(
        #         self.market.dtype
        #     ),
        # )

    def receive_revenue(self):
        revenue = self.market.process_gains(self.id)
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
        :param percent_to_buy: [n_firms, n_branches]
        :param percent_to_sale: [n_branches]
        :param percent_to_use: [n_branches]
        :param prices: [n_branches]
        :return: revenue, costs
        """
        revenue = self.receive_revenue()
        costs = self.buy(percent_to_buy)
        self.produce(percent_to_use)
        self.define_prices(prices)
        self.sell(percent_to_sale)
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


class LimitFirm(BaseFirm):
    def __init__(
        self,
        prod_function,
        invest_function,
        market: Market,
        financial_resources=0,
        is_deprecating=True,
    ):
        super().__init__(
            prod_function=prod_function,
            market=market,
            financial_resources=financial_resources,
        )
        deprecation_steps = market.deprecation_steps
        self.invest_function = invest_function
        self.deprecation_steps = deprecation_steps
        self.capital = torch.tensor([deprecation_steps])
        self.is_deprecating = is_deprecating

    def reset(self):
        super().reset()
        self.capital = torch.tensor([self.deprecation_steps])

    @property
    def limit(self):
        return len(self.capital)

    def deprecation(self):
        if self.is_deprecating:
            self.capital = self.capital[self.capital >= 1] - 1

    def invest(self, percent_to_use: torch.Tensor):
        """
        :param percent_to_use: (n_branches)
        Какую долю резервов от каждого товара потратить на инвестициии
        """
        assert torch.all((0 <= percent_to_use) & (percent_to_use <= 1))
        input_reserves = (self.reserves * percent_to_use).round().type(torch.int64)
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_limits = self.invest_function(input_reserves)
        self.reserves -= used_reserves
        self.capital = torch.hstack(
            [self.capital, torch.full((new_limits,), self.deprecation_steps)]
        )

    def produce(self, percent_to_use: torch.Tensor):
        """
        :param percent_to_use: (n_branches)
        Какую долю резервов от каждого товара потратить для производство
        """
        assert torch.all((0 <= percent_to_use) & (percent_to_use <= 1))
        input_reserves = (self.reserves * percent_to_use).round().type(torch.int64)
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_reserves = self.prod_function(
            input_reserves, limit=self.limit
        )
        self.reserves -= used_reserves
        self.reserves += new_reserves

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
            percent_to_use, [self.n_branches, self.n_branches]
        )
        revenue = self.receive_revenue()
        costs = self.buy(percent_to_buy)
        self.invest(percent_to_use_invest)
        self.produce(percent_to_use_prod)
        self.define_prices(prices)
        self.sell(percent_to_sale)
        self.deprecation()
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
