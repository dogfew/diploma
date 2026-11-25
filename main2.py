import torch
from enviroment import BaseFirm, Environment, Leontief, LimitFirm, Market

from models.critic import CentralizedCritic, CentralizedCritic2
from models.policy import BetaPolicyNetwork, BetaPolicyNetwork2, DeterministicPolicyNetwork
from models.utils import get_action_dim, get_state, get_state_dim, process_actions
from trainer import Trainer3, TrainerAC, TrainerSAC

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = "cuda"
market = Market(start_volumes=10, base_price=100, start_gains=500, deprecation_steps=3, device=device)
prod_functions = [
    Leontief(torch.tensor([1, 0]), torch.tensor([0, 2]), device=device),
    Leontief(torch.tensor([0, 1]), torch.tensor([2, 0]), device=device),
]
invest_functions = [
    Leontief(torch.tensor([1, 1]), torch.tensor(1), device=device),
    Leontief(torch.tensor([1, 1]), torch.tensor(1), device=device),
]
env = Environment(market, BetaPolicyNetwork2, prod_functions, invest_functions)
critic = CentralizedCritic2
trainer = TrainerSAC(env, q_critic=critic, batch_size=128)
trainer.train(1000, episode_length=50)
