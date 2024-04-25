import torch


class BatchedLeontief:
    """
    Leontief Production Function
    """

    def __init__(
        self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, device="cpu"
    ):
        """
        :param input_tensor: [n_branches]
        :param output_tensor: [n_branches]
        """
        self.device = device
        self.input_tensor = input_tensor.to(device)
        self.output_tensor = output_tensor.to(device)
        self.dtype = input_tensor.dtype
        self.max_limit = (
            torch.iinfo(self.dtype).max
            if self.dtype in [torch.int64, torch.int32]
            else torch.inf
        )
        # assert self.input_tensor.shape == self.output_tensor.shape
        assert len(self.input_tensor.shape) == 1

    def __call__(self, input_reserves, limit=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param: input_reserves
        :return: used_reserves, new_reserves
        """
        bs = input_reserves.shape[0]
        if limit is None:
            limit = torch.full((bs, 1), self.max_limit, device=self.device)
        minimum = torch.full((bs, 1), 0, device=self.device)
        produced = torch.min(
            input_reserves[:, self.input_tensor != 0]
            // self.input_tensor[self.input_tensor != 0],
            dim=1,
            keepdim=True,
        ).values
        produced = torch.clamp(
            produced,
            min=minimum,
            max=limit,
        )
        used_reserves = produced * self.input_tensor
        new_reserves = produced * self.output_tensor
        return used_reserves, new_reserves


class CobbDouglas:
    def __init__(self, alpha: torch.Tensor, output_tensor: torch.Tensor):
        self.alpha = alpha
        self.output_tensor = output_tensor
        self.device = "cpu"
        assert self.alpha.shape == self.output_tensor.shape
        assert len(self.alpha.shape) == 1

    def __call__(
        self, input_reserves, limit=torch.inf
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Callable[[torch.Tensor, ], tuple[torch.Tensor, torch.Tensor]]
        :param: input_reserves
        :return: used_reserves, new_reserves
        """
        used_reserves = (input_reserves**self.alpha).type(torch.int64)
        produced = torch.clamp(torch.prod(used_reserves), min=0, max=limit)
        new_reserves = produced * self.output_tensor
        return used_reserves, new_reserves


class BatchedProdFunction:
    Leontief = BatchedLeontief
    CobbDouglas = CobbDouglas
