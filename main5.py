import torch
from environment_batched import BatchedMarket, BatchedLeontief, BatchedFirm, BatchedEnvironment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.critic import CentralizedCritic, CentralizedCriticV
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerPPO, TrainerSAC
from utils.plotting import plot_actions, plot_environment, plot_volumes

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = 'cuda'
market_kwargs = dict(start_volumes=10,
                     base_price=50,
                     start_gains=500,
                     deprecation_steps=5,
                     max_price=100,
                     )
prod_functions = [
    BatchedLeontief(torch.tensor([1, 0]), torch.tensor([0, 2]), device=device),
    BatchedLeontief(torch.tensor([0, 1]), torch.tensor([2, 0]), device=device),
]
invest_functions = [
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(2), device=device),
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(2), device=device),
]
env = BatchedEnvironment(market_kwargs,
                         BetaPolicyNetwork,
                         prod_functions,
                         invest_functions=invest_functions,
                         target='production',
                         production_reg=1,  # 10 is good
                         device=device,
                         batch_size=512)
critic = CentralizedCriticV
trainer = TrainerPPO(env,
                     critic=critic,
                     learning_rates=(3e-3, 3e-4),
                     batch_size=512,
                     entropy_reg=0.1,
                     buffer_size=8192 * 64,
                     device=device,
                     entropy_gamma=0.999,
                     lr_gamma=0.991,
                     common_optimizer=True
                     )
# trainer.train_epoch(1)
trainer.train(300, episode_length=32, debug_period=10)
env.change_batch_size(1)
env.reset()
n_periods = 64
for i in range(n_periods):
    env.step_and_record(i % env.market.n_firms)
plot_environment(env.state_history)
plot_volumes(env.state_history)
plot_actions(env.actions_history[0], 'Политика Фирма 1 (1)')
plot_actions(env.actions_history[1], 'Политика Фирма 2 (2)')
#