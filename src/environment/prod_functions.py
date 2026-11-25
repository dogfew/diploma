import torch

EPS = 1e-4


class BatchedLeontief:
    """
    Leontief Production Function
    """

    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, device="cpu"):
        """
        :param input_tensor: [n_branches]
        :param output_tensor: [n_branches]
        """
        self.device = device
        self.n_branches = input_tensor.shape[0]
        self.input_tensor = input_tensor.to(device)
        self.output_tensor = output_tensor.to(device)
        self.dtype = input_tensor.dtype
        self.max_limit = torch.iinfo(self.dtype).max
        self.input_idx = (self.input_tensor != 0).nonzero(as_tuple=True)[0]
        self.input_nonzero = self.input_tensor[self.input_tensor != 0]
        # assert self.input_tensor.shape == self.output_tensor.shape
        assert len(self.input_tensor.shape) == 1

    def __call__(self, input_reserves, limit=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param input_reserves: [bs, n_branches]
        :param limit: [bs, 1] или None — максимальный выпуск
        :return: used_reserves, new_reserves
        """
        produced = torch.amin(
            input_reserves[:, self.input_idx] // self.input_nonzero,
            dim=1,
            keepdim=True,
        )
        if limit is not None:
            produced.clamp_max_(limit)
        used_reserves = produced * self.input_tensor
        new_reserves = produced * self.output_tensor
        return used_reserves, new_reserves

    # def __call__(self, input_reserves, limit=None) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     :param: input_reserves
    #     :return: used_reserves, new_reserves
    #     """
    #     bs = input_reserves.shape[0]
    #     if limit is None:
    #         limit = torch.full((bs, 1), self.max_limit, device=self.device)
    #     minimum = torch.full((bs, 1), 0, device=self.device)
    #     produced = torch.min(
    #         input_reserves[:, self.input_tensor != 0] // self.input_tensor[self.input_tensor != 0],
    #         dim=1,
    #         keepdim=True,
    #     ).values
    #     produced = torch.clamp(
    #         produced,
    #         min=minimum,
    #         max=limit,
    #     )
    #     used_reserves = produced * self.input_tensor
    #     new_reserves = produced * self.output_tensor
    #     return (
    #         used_reserves,
    #         new_reserves,
    #     )


class MoneyPrintingLeontief(BatchedLeontief):
    """
    Leontief Production Function
    """

    def __init__(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        money_delta: int = 0,
        device="cpu",
    ):
        """
        :param input_tensor: [n_branches]
        :param output_tensor: [n_branches]
        """
        super().__init__(input_tensor, output_tensor, device)
        self.money_delta = money_delta  # processing is on the side of the firm


class CobbDouglas:
    """
    Cobb-Douglas Production Function: Q = prod_i x_i ** alpha_i
    """

    def __init__(self, alpha: torch.Tensor, output_tensor: torch.Tensor, device="cpu"):
        """
        :param alpha: [n_branches], показатели степени (0 — ресурс не используется)
        :param output_tensor: [n_branches], выпуск на единицу продукции
        """
        assert alpha.dim() == 1
        assert (alpha >= 0).all()
        self.device = device
        self.n_branches = alpha.shape[0]
        self.alpha = alpha.to(device, torch.float64)
        self.output_tensor = output_tensor.to(device)
        self.input_idx = (self.alpha > 0).nonzero(as_tuple=True)[0]
        self.alpha_nonzero = self.alpha[self.input_idx]
        self.inv_alpha_sum = 1 / self.alpha_nonzero.sum()
        assert self.inv_alpha_sum > 0

    def __call__(self, input_reserves, limit=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param input_reserves: [bs, n_branches]
        :param limit: [bs, 1] или None — максимальный выпуск
        :return: used_reserves, new_reserves
        """
        x = input_reserves[:, self.input_idx].double()
        full_output = (self.alpha_nonzero * x.log()).sum(dim=1, keepdim=True).exp()
        produced = torch.floor(full_output + EPS)
        if limit is not None:
            produced = torch.minimum(produced, limit.double())
        share = (produced / full_output.clamp_min(EPS)).clamp_max(1) ** self.inv_alpha_sum
        used_inputs = torch.ceil(share * x - EPS)
        used_reserves = torch.zeros_like(input_reserves)
        used_reserves[:, self.input_idx] = used_inputs.to(input_reserves.dtype)
        new_reserves = produced.to(input_reserves.dtype) * self.output_tensor
        return used_reserves, new_reserves
