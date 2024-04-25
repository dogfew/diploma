import torch
from torch import tensor
from environment_batched import BatchedMarket, BatchedLeontief, BatchedFirm, BatchedEnvironment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.critic import CentralizedCritic, CentralizedCritic2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerAC, TrainerSAC, Trainer3

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = 'cuda'
market_kwargs = dict(start_volumes=10, base_price=100, start_gains=100, deprecation_steps=1)
prod_functions = [
    BatchedLeontief(torch.tensor([1, 0]), torch.tensor([0, 2]), device=device),
    BatchedLeontief(torch.tensor([0, 1]), torch.tensor([2, 0]), device=device),
]
invest_functions = [
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(1), device=device),
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(1), device=device),
]
env = BatchedEnvironment(market_kwargs,
                         BetaPolicyNetwork2,
                         prod_functions,
                         invest_functions=invest_functions,
                         batch_size=3)
env.reset()
invest_tensor = env.firms[0].reserves * torch.tensor([[1, 1], [0.5, 1], [1, 0.3]], device='cuda')
env.firms[0].invest(invest_tensor.round().type(torch.int64))
env.firms[0].invest(invest_tensor.round().type(torch.int64))
env.firms[0].invest(invest_tensor.round().type(torch.int64))
env.firms[0].invest(invest_tensor.round().type(torch.int64))
env.firms[0].invest(invest_tensor.round().type(torch.int64))
env.firms[0].invest(invest_tensor.round().type(torch.int64))

# env.firms[0].invest(invest_tensor.round().type(torch.int64))
# env.firms[0].deprecation() # 0, 5, 3
print(env.firms[0].capital, env.firms[0].limit)
# right_answer = [tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                 tensor([0, 0, 0, 0, 0, 0, 1, 1]),
#                 tensor([0, 0, 0, 0, 1, 1])]
# print([torch.all(x == y).item() for x, y in zip(env.firms[0].capital, right_answer)])
