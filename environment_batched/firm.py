import torch
from copy import deepcopy


class BatchedFirm:
    """
    Firm without Fixed Capital
    """

    def __init__(
            self,
            prod_function,
            market,
            batch_size=1,
    ):
        """
        :param prod_function: Производственная функция
        :attribute reserves: Объём запасов фирмы
        """
        self.batch_size = batch_size
        self.prod_function = prod_function
        self.market = market
        self.n_branches = self.market.n_branches
        self.reserves = torch.zeros(
            (self.batch_size, self.market.n_branches),
            dtype=self.market.dtype, device=self.device)
        self.financial_resources = torch.zeros((self.batch_size, 1), dtype=self.market.dtype, device=self.device)
        self.id = market.generate_id() - 1

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
            percent_to_buy * self.financial_resources.unsqueeze(-1) // self.market.price_matrix,
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

    def define_prices(self, prices):
        """
        :param prices: (n_branches)
        """
        if prices is None:
            return
        new_prices = (prices * self.market.max_price).type(self.market.dtype)
        self.market.process_prices(self.id, new_prices)

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
            is_deprecating=True,
    ):
        super().__init__(
            prod_function=prod_function,
            market=market,
            batch_size=batch_size
        )
        deprecation_steps = market.deprecation_steps
        self.invest_function = invest_function
        self.deprecation_steps = deprecation_steps  # How much time Fixed Capital exists
        self.capital = [torch.tensor([deprecation_steps], device=self.device) for _ in range(self.batch_size)]
        self.is_deprecating = is_deprecating

    def reset(self):
        super().reset()
        self.capital = [torch.tensor([self.deprecation_steps]) for _ in range(self.batch_size)]

    @property
    def limit(self):
        return torch.tensor([x.size(0) for x in self.capital], device=self.device).unsqueeze(1)

    def deprecation(self):
        if self.is_deprecating:
            self.capital = [x[x >= 1] - 1 for x in self.capital]

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
        for i, new_limit in enumerate(new_limits):
            if new_limit > 0:
                self.capital[i] = torch.hstack(
                    [self.capital[i], torch.full((new_limit, ), self.deprecation_steps)]
                )

    def produce(self, percent_to_use: torch.Tensor):
        """
        :param percent_to_use: (n_branches)
        Какую долю резервов от каждого товара потратить для производство
        """
        assert torch.all((0 <= percent_to_use) & (percent_to_use <= 1))
        input_reserves = (self.reserves * percent_to_use).round().type(torch.int64)
        input_reserves = torch.min(input_reserves, self.reserves)
        used_reserves, new_reserves = self.prod_function(input_reserves, limit=self.limit)
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
            percent_to_use, [self.n_branches, self.n_branches], dim=1
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


if __name__ == '__main__':
    from environment_batched.market import BatchedMarket
    from environment_batched.prod_functions import BatchedLeontief as Leontief

    batch_size = 64
    market = BatchedMarket(batch_size=batch_size)
    firm = BatchedLimitFirm(market=market,
                            batch_size=batch_size,
                            prod_function=Leontief(torch.tensor([1, 0]), torch.tensor([0, 2])),
                            invest_function=Leontief(torch.tensor([1, 0]), torch.tensor([0, 2]))
                            )
    firm2 = BatchedFirm(market=market,
                        batch_size=batch_size,
                        prod_function=Leontief(torch.tensor([1, 0]), torch.tensor([0, 2])))
    firm3 = BatchedFirm(market=market,
                        batch_size=batch_size,
                        prod_function=Leontief(torch.tensor([1, 0]), torch.tensor([0, 2])))
    print(firm.capital)
