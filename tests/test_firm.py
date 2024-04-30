import torch
from torch import tensor
from environment_batched import (
    BatchedMarket,
    BatchedLeontief,
    BatchedFirm,
    BatchedEnvironment,
)
from models.policy import (
    BetaPolicyNetwork,
    DeterministicPolicyNetwork,
    BetaPolicyNetwork2,
)
from models.critic import CentralizedCritic, CentralizedCritic2
from models.utils import get_state, get_state_dim, process_actions, get_action_dim
from trainer import TrainerAC, TrainerSAC, Trainer3

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = "cuda"
market_kwargs = dict(
    start_volumes=10, base_price=100, start_gains=100, deprecation_steps=1
)
prod_functions = [
    BatchedLeontief(torch.tensor([1, 0]), torch.tensor([0, 2]), device=device),
    BatchedLeontief(torch.tensor([0, 1]), torch.tensor([2, 0]), device=device),
]
invest_functions = [
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(1), device=device),
    BatchedLeontief(torch.tensor([1, 1]), torch.tensor(1), device=device),
]
env = BatchedEnvironment(
    market_kwargs,
    BetaPolicyNetwork2,
    prod_functions,
    invest_functions=invest_functions,
    batch_size=3,
)
env.reset()
invest_tensor = env.firms[0].reserves * torch.tensor(
    [[1, 1], [0.5, 1], [1, 0.3]], device="cuda"
)
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
@torch.no_grad()
def get_trajectory(self, order=None, num_steps=50):
    if order is None:
        order = range(len(self.environment.firms))

    batch_size = self.environment.batch_size
    state_dim = self.environment.state_dim
    probs_dim = self.environment.probs_dim
    n_agents = self.environment.n_agents

    total_steps = num_steps + 1
    kwargs = dict(device=self.device, fill_value=float('nan'))

    all_advantages = torch.full((total_steps, batch_size, n_agents, 1), **kwargs)
    all_states = torch.full((total_steps + 1, batch_size, n_agents, state_dim), **kwargs)
    all_rewards = torch.full((total_steps, batch_size, n_agents, 1), **kwargs)
    all_log_probs = torch.full((total_steps, batch_size, n_agents, probs_dim), **kwargs)
    all_values = torch.full((total_steps + 1, batch_size, n_agents), **kwargs)

    # First Step
    for firm_id in order:
        state, _, log_probs, revenue, costs = self.environment.step(firm_id)
        all_states[0, :, firm_id, :] = state
        all_log_probs[0, :, firm_id, :] = log_probs
        all_rewards[0, :, firm_id, :] = -costs
    all_values[0] = self.critic(all_states[0])
    # Other steps
    for step in range(1, total_steps):
        for firm_id in order:
            state, _, log_probs, revenue, costs = self.environment.step(
                firm_id
            )
            all_states[step, :, firm_id, :] = state
            all_log_probs[step, :, firm_id, :] = log_probs
            all_rewards[step, :, firm_id, :] = -costs
            all_rewards[step - 1, :, firm_id, :] += revenue
        all_values[step] = self.critic(all_states[step])

    for firm_id in order:
        # First Idea
        # state, _, _, _, _ = self.environment.step(
        #     firm_id
        # )
        # state = get_state_log(self.environment.market,
        #                       self.environment.firms[firm_id])
        all_states[-1, :, firm_id, :] = state
    all_values[-1] = self.critic(all_states[-1])
    all_values = all_values.unsqueeze(-1)
    all_rewards /= self.environment.market.start_gains
    # Compute GAE
    gae = 0
    gamma, lambda_ = 0.99, 0.95
    for t in reversed(range(total_steps)):
        next_values = all_values[t + 1]
        current_values = all_values[t]

        delta = current_values + gamma * next_values - current_values
        all_advantages[t] = gae = delta + gamma * lambda_ * gae

    all_value_targets = all_advantages + all_values[:-1]
    trajectory = dict(
        x=all_states[:-1],
        x_next=all_states[1:],
        values=all_values[:-1],
        advantages=all_advantages,
        value_targets=all_value_targets[:],
        log_probs=all_log_probs[:]
    )
    # Permute Batch
    for key in trajectory:
        trajectory[key] = trajectory[key].flatten(0, 1)
    # total_indices = (total_steps - 1) * batch_size
    # indices = torch.randperm(total_indices)
    # for key in trajectory:
    #     reshaped_data = trajectory[key][indices]
    #     trajectory[key] = reshaped_data.view(total_steps - 1, batch_size, *trajectory[key].shape[1:])
    self.buffer.add_batch(trajectory)
    return trajectory, all_rewards.permute(0, 2, 1, 3) * self.environment.market.start_gains
