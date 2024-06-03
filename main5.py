import torch
from environment_batched import BatchedMarket, BatchedLeontief, BatchedFirm, BatchedEnvironment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerPPO, TrainerSAC
from utils.plotting_mean import plot_environment_batch, plot_volumes_batch, plot_actions_batch
# from utils.plotting_tikz import plot_environment_batch, plot_volumes_batch, plot_actions_batch, plot_loss_batch

torch.manual_seed(123)
torch.backends.cudnn.deterministic = False
device = 'cuda'
market_kwargs = dict(start_volumes=10,
                     base_price=50,
                     start_gains=500,
                     deprecation_steps=2,
                     min_price=1,
                     max_price=100)
prod_functions = [
    BatchedLeontief(torch.tensor([0, 1]), torch.tensor([2, 0]), device=device),  # 0 товара А + 1 товар  Б => 2 товара А.
    BatchedLeontief(torch.tensor([1, 0]), torch.tensor([0, 2]), device=device),  # 1 товара А + 0 товара Б => 2 товара Б
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
                         percent_prices=False,
                         production_reg=0,  # 10 is good
                         device=device,
                         batch_size=1024,
                         )
trainer = TrainerPPO(env,
                     learning_rates=(3e-3, 3e-4),
                     batch_size=1024,
                     entropy_reg=1e-5,
                     device=device,
                     max_grad_norm=0.5,
                     entropy_gamma=0.9999,
                     social_planner=False,
                     lr_gamma=0.999,
                     common_optimizer=True)
trainer.train(320,
              episode_length=32,
              debug_period=20,
              shuffle_order=False)
env.reset()
env.change_batch_size(1)
n_periods = 64
for i in range(n_periods):
    env.step_and_record_batch(i % env.market.n_firms)
plot_environment_batch(env.state_history)
plot_volumes_batch(env.state_history)
plot_actions_batch(env.actions_history[0], 'policy1')
plot_actions_batch(env.actions_history[1], 'policy2')
env.reset()
env.change_batch_size(512)
n_periods = 64
for i in range(n_periods):
    env.step_and_record_batch(i % env.market.n_firms)
plot_environment_batch(env.state_history)
plot_volumes_batch(env.state_history)
plot_actions_batch(env.actions_history[0], 'policy1')
plot_actions_batch(env.actions_history[1], 'policy2')
