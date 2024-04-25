import torch
import math


def process_actions(actions, size):
    percent_to_buy, percent_to_sale, percent_to_use, percent_price_change = actions
    return (
        percent_to_buy[:, :-1].unflatten(1, size[1:]),
        percent_to_sale,
        percent_to_use,
        percent_price_change,
    )


def get_state_dim(market, limit=False) -> int:
    return 2 * math.prod(market.price_matrix.shape[1:]) + market.price_matrix.shape[2] * (1 + limit) + 1


def get_state(market, firm):
    return torch.concatenate(
        [
            market.price_matrix.flatten(start_dim=1),
            market.volume_matrix.flatten(start_dim=1),
            firm.reserves.flatten(start_dim=1),
            (firm.financial_resources + market.gains[:, firm.id]).unsqueeze(0),
        ]
    ).type(torch.float32)


@torch.no_grad()
def get_state_log(market, firm):
    """
    input_prices  = log10(p)
    input_volumes = log(v + 1)
    """
    price_scaling = (market.max_price + market.min_price) / 2

    price_state = (market.price_matrix.flatten(start_dim=1) - price_scaling) / price_scaling
    volume_state = (market.volume_matrix.flatten(start_dim=1) + 1).log10()
    reserves_state = (firm.reserves.flatten(start_dim=1) + 1).log10()
    finance_state = (firm.financial_resources + market.gains[:, firm.id, None])
    finance_state = (finance_state - market.max_price / market.n_firms) / market.max_price
    state_lst = [
            price_state,  # [-1, 1]
            volume_state,  # [0, log10)
            reserves_state,  # [0, log10)
            finance_state,  # [-1, 1]
        ]
    if hasattr(firm, 'capital'):
        first_part = firm.limit
        second_part = torch.tensor(
            [x.type(torch.float32).mean(dim=0, keepdim=True) for x in firm.capital],
            device=market.device
        ).unsqueeze(1)
        limit_state = (torch.cat([first_part, second_part], dim=1)
                       .log1p().nan_to_num(0, 0, 0))
        state_lst.append(limit_state)
    concatenated_state = torch.concatenate(
        state_lst,
        dim=1
    )
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
