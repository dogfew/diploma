import torch
import math


def process_actions(actions, size):
    percent_to_buy, percent_to_sale, percent_to_use, percent_price_change = actions
    return (
        percent_to_buy[:-1].unflatten(-1, size),
        percent_to_sale,
        percent_to_use,
        percent_price_change,
    )


def get_state_dim(market, limit) -> int:
    return (
        2 * math.prod(market.price_matrix.shape)
        + market.price_matrix.shape[1]
        + 1
        + limit * 2
    )


def get_state(market, firm):
    return torch.concatenate(
        [
            market.price_matrix.flatten(),
            market.volume_matrix.flatten(),
            firm.reserves.flatten(),
            (firm.financial_resources + market.gains[firm.id]).unsqueeze(0),
        ]
    ).type(torch.float32)


@torch.no_grad()
def get_state_log(market, firm):
    """
    input_prices  = log10(p)
    input_volumes = log(v + 1)
    """
    price_scaling = (market.max_price + market.min_price) / 2

    price_state = (market.price_matrix.flatten() - price_scaling) / price_scaling
    volume_state = (market.volume_matrix.flatten() + 1).log10()
    reserves_state = (firm.reserves.flatten() + 1).log10()
    finance_state = firm.financial_resources.unsqueeze(0) + market.gains[firm.id]
    finance_state = (
        finance_state - market.max_price / market.n_firms
    ) / market.max_price

    state_lst = [
        price_state,  # [-1, 1]
        volume_state,  # [0, log10)
        reserves_state,  # [0, log10)
        finance_state,  # [-1, 1]
    ]
    if hasattr(firm, "limit"):
        limit_state = (
            torch.tensor(
                [
                    firm.limit,
                    firm.capital.type(torch.float32).mean(dim=0, keepdim=True),
                ],
                device=market.device,
            )
            .log1p()
            .nan_to_num(0, 0, 0)
        )
        state_lst.append(limit_state)
    concatenated_state = torch.concatenate(state_lst)
    return concatenated_state.type(torch.float32)


def get_action_dim(market, limit=False):
    return (
        market.n_firms * market.n_branches
        + 1
        + market.n_branches
        + market.n_branches
        + market.n_branches
        + limit * market.n_branches
    )
