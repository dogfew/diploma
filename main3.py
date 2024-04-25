import torch
from environment_batched import BatchedMarket, BatchedLeontief, BatchedFirm, BatchedEnvironment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.critic import CentralizedCritic, CentralizedCritic2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerAC, TrainerSAC, Trainer3
from utils.plotting import plot_actions, plot_environment

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = 'cuda'
market_kwargs = dict(start_volumes=10, base_price=100, start_gains=100, deprecation_steps=2)
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
                         target='production',
                         batch_size=128)
critic = CentralizedCritic2
trainer = TrainerSAC(env, q_critic=critic, batch_size=512)
# trainer.train_epoch()
trainer.train(1000, episode_length=30)
#
# env.change_batch_size(1)
# env.reset()
# n_periods = 100
# for i in range(n_periods):
#     env.step_and_record(i % env.market.n_firms)
# plot_environment(env.state_history)
# plot_actions(env.actions_history[0], 'Политика Фирма 1 (1)')
# plot_actions(env.actions_history[1], 'Политика Фирма 2 (2)')