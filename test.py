import torch
from environment_batched import BatchedMarket, BatchedLeontief, BatchedFirm, BatchedEnvironment
from models.policy import BetaPolicyNetwork, DeterministicPolicyNetwork, BetaPolicyNetwork2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerPPO, TrainerSAC
from utils import plot_environment_batch, plot_volumes_batch
from utils.plotting import plot_actions, plot_environment, plot_volumes
import torch.nn.functional as F

device = 'cuda'
env_kwargs = dict(
     device=device,
     batch_size=512
)

trainer_kwargs = dict(
     learning_rates=(3e-3, 3e-4),
     batch_size=512,
     entropy_reg=0.01,
     buffer_size=8192 * 64,
     device=device,
     entropy_gamma=0.999,
     lr_gamma=0.991,
     common_optimizer=True
)
market_kwargs = dict(
    start_volumes=10,      # У всех фирм в резервах изначально 4 товара А и 10 товара Б
    base_price=50,         # Изначальные цены на рынке - 50
    start_gains=500,       # Изначальные финансовые ресурсы у каждой фирмы - 500
    deprecation_steps=2,   # За сколько ходов износится основной капитал
    min_price=1,           # Минимальная возможная цена на рынке
    max_price=100          # Максимальная возможная цена на рынке
)
# Производственные функции
prod_functions = [
    BatchedLeontief(torch.tensor([0, 1, 1]), torch.tensor([3, 0, 0]), device=device),  # 0 товара А + 1 товар  Б => 2 товара А.
    BatchedLeontief(torch.tensor([1, 0, 1]), torch.tensor([0, 3, 0]), device=device),  # 0 товара А + 1 товар  Б => 2 товара А.
    BatchedLeontief(torch.tensor([1, 1, 0]), torch.tensor([0, 0, 3]), device=device),  # 0 товара А + 1 товар  Б => 2 товара А.
]

# Инвестиционные функции
invest_functions = [
    BatchedLeontief(torch.tensor([0, 0, 2]), torch.tensor(2), device=device),
    BatchedLeontief(torch.tensor([0, 0, 2]), torch.tensor(2), device=device),
    BatchedLeontief(torch.tensor([0, 0, 2]), torch.tensor(2), device=device),
]

torch.manual_seed(123)
env = BatchedEnvironment(market_kwargs,
                         BetaPolicyNetwork,
                         prod_functions,
                         invest_functions=invest_functions,
                         percent_prices=True,
                         target='production',
                         # production_reg=10,
                         **env_kwargs
                        )
trainer = TrainerPPO(env, **trainer_kwargs)
# trainer.train(500, 32, shuffle_order=True)
env.change_batch_size(512)
env.reset()
n_periods = 256
for i in range(n_periods):
    env.step_and_record_batch(i % env.market.n_firms)

plot_environment_batch(env_history=env.state_history)
plot_volumes_batch(env_history=env.state_history)