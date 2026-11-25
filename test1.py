import torch

from src.environment import BatchedEnvironment, BatchedLeontief
from src.models.policy import BetaPolicyNetwork
from src.trainer import TrainerPPO
from src.utils import plot_actions_batch, plot_environment_batch, plot_volumes_batch

torch.manual_seed(123)
device = "cpu"
# Гиперпараметры рынка
market_kwargs = {
    "start_volumes": 20,  # У всех фирм в резервах изначально 4 товара А и 10 товара Б
    "base_price": 20,  # Изначальные цены на рынке - 50
    "start_gains": 128,  # Изначальные финансовые ресурсы у каждой фирмы - 500
    "deprecation_steps": 3,  # За сколько ходов износится основной капитал
    "min_price": 1,  # Минимальная возможная цена на рынке
    "max_price": 256,  # Максимальная возможная цена на рынке
}
# Производственные функции
prod_functions = [
    BatchedLeontief(
        torch.tensor([0, 1, 1]),
        torch.tensor([3, 0, 0]),
        device=device,
    ),  # 0 товара А + 1 товар  Б => 2 товара А.
    BatchedLeontief(
        torch.tensor([1, 0, 1]),
        torch.tensor([0, 3, 0]),
        device=device,
    ),  # 0 товара А + 1 товар  Б => 2 товара А.
    BatchedLeontief(
        torch.tensor([0, 1, 1]),
        torch.tensor([0, 0, 2]),
        device=device,
    ),  # 0 товара А + 1 товар  Б => 2 товара А.
]

# Инвестиционные функции
invest_functions = [
    BatchedLeontief(torch.tensor([0, 0, 2]), torch.tensor(2), device=device),
    BatchedLeontief(torch.tensor([0, 0, 2]), torch.tensor(2), device=device),
    BatchedLeontief(torch.tensor([0, 0, 2]), torch.tensor(2), device=device),
]
env_kwargs = {"device": device, "batch_size": 1024}
env = BatchedEnvironment(
    market_kwargs,
    BetaPolicyNetwork,
    prod_functions,
    invest_functions=None,
    target="production",
    production_reg=0,
    is_blind=True,
    **env_kwargs,
)
trainer = TrainerPPO(
    env,
    learning_rates=(3e-3, 3e-4),
    batch_size=env_kwargs["batch_size"],
    entropy_reg=1e-5,
    device=device,
    max_grad_norm=0.5,
    entropy_gamma=0.9999,
    social_planner=False,
    lr_gamma=0.999,
    common_optimizer=True,
)
# print(env.market)
trainer.train(32, episode_length=32)
env.reset()
n_periods = 64
for i in range(n_periods):
    env.step_and_record_batch(i % env.market.n_firms)
plot_environment_batch(env.state_history)
plot_volumes_batch(env.state_history)
plot_actions_batch(env.actions_history[0], "policy1")
plot_actions_batch(env.actions_history[1], "policy2")
