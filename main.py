import torch
from enviroment import Market, Leontief, BaseFirm, Environment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.critic import CentralizedCritic, CentralizedCritic2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerAC, TrainerSAC, Trainer3

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = 'cuda'
market = Market(start_volumes=1, base_price=100, start_gains=100, device=device)
prod_functions = [
    Leontief(torch.tensor([1, 0]), torch.tensor([0, 2]), device=device),
    Leontief(torch.tensor([0, 1]), torch.tensor([2, 0]), device=device),
]
env = Environment(market,
                  BetaPolicyNetwork2,
                  prod_functions)
critic = CentralizedCritic2
trainer = TrainerSAC(env, q_critic=critic, batch_size=128)
trainer.train(1000, episode_length=30)
