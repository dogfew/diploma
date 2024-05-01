import torch
from environment_batched import BatchedMarket, BatchedLeontief, BatchedFirm, BatchedEnvironment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerAC, TrainerSAC, TrainerMASAC
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
                         invest_functions=None,
                         target='production',
                         production_reg=0,
                         batch_size=512)
trainer = TrainerSAC(env,
                    batch_size=512,
                    learning_rates=(3e-4, 3e-4),
                    buffer_size=8192 * 128,
                    entropy_reg=0.025,
                    device=device,
                    entropy_gamma=0.995,
                    max_grad_norm=0.5,
                    tau=0.995,
                    lr_gamma=0.991,
                    )
# trainer.train_epoch()
trainer.train(250, episode_length=32, shuffle_order=False)
#
env.change_batch_size(1)
env.reset()
n_periods = 64
for i in range(n_periods):
    env.step_and_record(i % env.market.n_firms)
plot_environment(env.state_history)
plot_volumes(env.state_history)
plot_actions(env.actions_history[0], 'Политика Фирма 1 (1)')
plot_actions(env.actions_history[1], 'Политика Фирма 2 (2)')
