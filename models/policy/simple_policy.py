import torch


class SimplePolicy:
    def __init__(self, firm, *args, **kwargs):
        self.firm = firm
        self.prod_function_input = firm.prod_function.input_tensor
        self.prod_function_output = firm.prod_function.output_tensor
        self.n_branches = firm.n_branches
        self.n_firms = firm.market.n_firms
        self.shape = (self.n_firms, self.n_branches)

    def __call__(
        self, market
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param market:
        :return: tuple[
            percent_to_buy: [n_firms, n_branches],
            percent_to_sale: [n_branches],
            percent_to_use: [n_branches],
            prices: [n_branches]
        ]
        """
        matrix_to_consider = market.volume_matrix.clone()
        matrix_to_consider[self.firm.id] = 0
        percent_to_buy: torch.Tensor = torch.where(
            matrix_to_consider != 0, 1 / matrix_to_consider.nonzero().sum(), 0
        )

        percent_to_sale: torch.Tensor = torch.ones(self.n_branches)
        percent_to_use: torch.Tensor = torch.ones(self.n_branches)
        prices: torch.Tensor = None
        return percent_to_buy, percent_to_sale, percent_to_use, prices
