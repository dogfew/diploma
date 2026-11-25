import torch

from src.environment import BatchedEnvironment, BatchedLeontief
from src.models.policy import BetaPolicyNetwork
from src.trainer.trainer_masac import TrainerMASAC
from src.trainer.trainer_ppo import TrainerPPO
from src.utils.plotting_mean import plot_actions_batch, plot_environment_batch, plot_volumes_batch

# from utils.plotting_tikz import plot_environment_batch, plot_volumes_batch, plot_actions_batch, plot_loss_batch
torch.set_num_threads(4)
torch.manual_seed(888)
device = "cpu"
market_kwargs = {
    "start_volumes": 10,
    "base_price": 50,
    "start_gains": 500,
    "deprecation_steps": 2,
    "min_price": 1,
    "max_price": 100,
}
prod_functions = [
    BatchedLeontief(
        torch.tensor([0, 4]),
        torch.tensor([5, 0]),
        device=device,
    ),  # 0 товара A + 1 товар  B => 2 товара A.
    BatchedLeontief(
        torch.tensor([4, 0]),
        torch.tensor([0, 5]),
        device=device,
    ),  # 1 товара A + 0 товара B => 2 товара B
]

invest_functions = [
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(2), device=device),
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(2), device=device),
]
env = BatchedEnvironment(
    market_kwargs,
    BetaPolicyNetwork,
    prod_functions,
    invest_functions=invest_functions,
    target="production",
    percent_prices=False,
    production_reg=0,  # 10 is good
    device=device,
    batch_size=512,
    is_blind=True,
    # normalize_rewards=True,
)
trainer = TrainerMASAC(
    env,
    learning_rates=(3e-3, 3e-4),
    entropy_reg=-8,
    device=device,
    max_grad_norm=1.0,
    buffer_size=8192 * 8,
    lr_gamma=0.998,
)
# trainer = TrainerPPO(
#     env,
#     learning_rates=(3e-3, 3e-4),
#     entropy_reg=1e-5,
#     device=device,
#     max_grad_norm=0.5,
#     entropy_gamma=0.98,
#     social_planner=False,
#     use_buffer=False,
#     lr_gamma=0.998,
#     common_optimizer=True,
#     normalize_advantages=True,
# )

trainer.train(n_epochs=1024, episode_length=32, debug_period=20, shuffle_order=False)
# env.reset()
# env.change_batch_size(1)
# n_periods = 64
# for i in range(n_periods):
#     env.step_and_record_batch(i % env.market.n_firms)
# plot_environment_batch(env.state_history)
# plot_volumes_batch(env.state_history)
# plot_actions_batch(env.actions_history[0], "policy1")
# plot_actions_batch(env.actions_history[1], "policy2")
env.reset()
env.change_batch_size(512)
n_periods = 128
for i in range(n_periods):
    env.step_and_record_batch(i % env.market.n_firms)
plot_environment_batch(env.state_history)
plot_volumes_batch(env.state_history)
plot_actions_batch(env.actions_history[0], "policy1")
plot_actions_batch(env.actions_history[1], "policy2")
