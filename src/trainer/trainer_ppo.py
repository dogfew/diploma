from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.nn.functional import smooth_l1_loss

from src.models.critic.centralized_v_critic import CentralizedCriticV, DecentralizedCriticV
from src.models.policy.beta_policy import BasePolicyNetwork
from src.trainer.base_trainer import BaseTrainer
from src.trainer.optimizers import KLAdaptiveLR
from src.trainer.replay_buffer import ReplayBufferPPO

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler


USE_VALUE_NORM = True


class TrainerPPO(BaseTrainer[ReplayBufferPPO]):
    """
    Multi-Agent Proximal Policy Optimization Algorithm
    """

    def __init__(
        self,
        environment,
        entropy_reg=0.01,
        learning_rates=(3e-4, 3e-4),
        actor_lr_range=(1e-5, 1e-3),
        critic_hidden_dim=64,
        gamma=0.99,
        entropy_gamma=0.999,
        max_grad_norm=0.5,
        *,
        shared_weights=True,
        # social_planner=False,
        normalize_advantages=True,
        critic_loss=smooth_l1_loss,
        batch_size: int = 1024,
        cliprange: float = 0.2,
        # use_buffer=False,
        device="cuda",
    ):
        super().__init__(
            environment,
            entropy_gamma=entropy_gamma,
            gamma=gamma,
            max_grad_norm=max_grad_norm,
            entropy_reg=entropy_reg,
            critic_loss=critic_loss,
            batch_size=batch_size,
            device=device,
        )
        critic_lr, actor_lr = learning_rates

        market = self.environment.market
        # state_dim, action_dim = (
        #     self.environment.state_dim,
        #     self.environment.action_dim,
        # )
        n_firms = market.n_firms
        self.critic_loss = critic_loss
        self.shared_weights = shared_weights
        self.critic_scheduler: LRScheduler | None = None
        self.critics = [
            DecentralizedCriticV(
                state_dim=self.environment.state_dim,
                n_agents=self.environment.n_agents,
                hidden_dim=critic_hidden_dim,
                # social_planner=social_planner,
            ).to(device)
            for _ in range(self.environment.n_agents if not self.shared_weights else 1)
        ]
        self.critic_params = [p for c in self.critics for p in c.parameters()]
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=critic_lr, eps=1e-5, fused=True)
        # self.critic_scheduler = LinearLR(
        #     self.critic_optimizer,
        #     start_factor=1,
        #     end_factor=0.1,
        #     total_iters=total_iters,
        # )

        self.policies: list[BasePolicyNetwork] = self.environment.policies
        for policy in self.environment.policies:
            policy.to(device)

        # Actors
        self.actor_parameters = [list(policy.parameters()) for policy in self.policies]
        self.actor_optimizer = torch.optim.Adam(
            [{"params": ps} for ps in self.actor_parameters],
            lr=actor_lr,
            eps=1e-5,
            fused=True,
        )
        assert actor_lr_range[0] <= actor_lr <= actor_lr_range[1]
        self.actor_scheduler = KLAdaptiveLR(
            self.actor_optimizer,
            min_lr=actor_lr_range[0],
            max_lr=actor_lr_range[1],
            target_kl=0.01,
        )
        # Hyperparams
        self.normalize_advantages = normalize_advantages
        self.cliprange = cliprange
        self.lambda_ = 0.95
        self.value_norm = ValueNorm(n_firms=self.environment.n_agents, beta=0.99, device=device)

    @property
    def last_lr(self):
        return self.actor_scheduler.get_last_lr()[0]

    def train_epoch(self, max_episode_length=32, order: list[int] | None = None):
        history = []
        trajectory, rewards_lst = self.get_trajectory(num_steps=max_episode_length, order=order)
        last_kl = [None] * self.n_agents
        n_samples = trajectory["x"].shape[0]
        n_batches = n_samples // self.batch_size
        n_batch_repetitions = 1
        for i in range(n_batch_repetitions * n_batches):
            idx = i % n_batches
            sampled = (
                trajectory.get(y)[idx * self.batch_size : (idx + 1) * self.batch_size]
                for y in ["x", "actions", "values", "advantages", "value_targets", "log_probs"]
            )
            x, actions, values, all_advantages, value_targets, log_probs_old = sampled
            total_critic_loss = 0
            total_actor_loss = 0
            v_all = self.get_critic_output(x)
            for firm_id in range(self.n_agents):
                # Compute Critic Loss
                v = v_all[:, firm_id].unsqueeze(-1)
                v_old = values[:, firm_id]
                v_hat = value_targets[:, firm_id]
                critic_loss = torch.maximum(
                    self.critic_loss(v, v_hat, reduction="none"),
                    self.critic_loss(
                        v.clamp(v_old - self.cliprange, v_old + self.cliprange),
                        v_hat,
                        reduction="none",
                    ),
                ).mean()

                # Compute Actor Loss
                advantages = all_advantages[:, firm_id]
                assert v.shape == v_old.shape == v_hat.shape == advantages.shape, (
                    v.shape,
                    v_old.shape,
                    v_hat.shape,
                    advantages.shape,
                )

                actions_restored = self.environment.restore_actions(
                    actions[:, firm_id].squeeze(),
                )
                log_probs_tuple, entropy = self.policies[firm_id].get_log_probs_and_entropy(
                    state=x[:, firm_id],
                    actions=actions_restored,
                )
                log_probs_new = torch.concatenate(
                    log_probs_tuple,
                    dim=-1,
                )
                log_ratios = (log_probs_new - log_probs_old[:, firm_id, :]).sum(dim=-1, keepdim=True)
                ratios = log_ratios.exp()
                actor_loss = -torch.minimum(
                    advantages * ratios,
                    advantages * ratios.clamp(1 - self.cliprange, 1 + self.cliprange),
                ).mean()
                entropy_loss = -entropy.mean()
                with torch.no_grad():
                    approx_kl = ((ratios - 1) - log_ratios).mean()
                last_kl[firm_id] = approx_kl
                total_actor_loss += actor_loss + entropy_loss * self.entropy_reg
                total_critic_loss += critic_loss
                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": critic_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "approx_kl": approx_kl.item(),
                            "firm_id": firm_id,
                            "lr": self.actor_optimizer.param_groups[firm_id]["lr"],
                        },
                    )
            self.optimize(
                loss=total_critic_loss,
                optimizer=self.critic_optimizer,
            )
            if max(kl.item() for kl in last_kl) > self.actor_scheduler.max_kl:
                break
            self.optimize(
                loss=total_actor_loss,
                optimizer=self.actor_optimizer,
            )

        self.actor_scheduler.step(last_kl)
        rewards = rewards_lst.mean(dim=(0, 2)).flatten().cpu().numpy()

        # assert self.critic_scheduler is not None
        # self.critic_scheduler.step()
        # self.entropy_reg *= self.entropy_gamma

        df_out = pd.DataFrame(history).groupby("firm_id").mean()
        df_out["reward"] = rewards
        return df_out

    def get_critic_output(self, state):
        if self.shared_weights:
            return self.critics[0](state)
        return torch.stack([c.forward_one(state[:, i], i) for i, c in enumerate(self.critics)], dim=1)
        return torch.stack([c(state)[:, i] for i, c in enumerate(self.critics)], dim=1)

    @torch.no_grad()
    def get_trajectory(self, order=None, num_steps=50):
        self.environment.reset()
        if order is None:
            order = range(len(self.environment.firms))
            if self.environment.social_planner_is_present:
                order = [order[-1], *list(order[:-1])]

        batch_size = self.environment.batch_size
        state_dim = self.environment.state_dim
        probs_dim = self.environment.probs_dim
        action_dim = self.environment.action_dim
        n_agents = self.environment.n_agents

        total_steps = num_steps + 1

        # all_advantages = torch.empty((total_steps, batch_size, n_agents, 1), device=self.device)
        all_actions = torch.empty((total_steps, batch_size, n_agents, action_dim), device=self.device)
        all_states = torch.empty((total_steps + 1, batch_size, n_agents, state_dim), device=self.device)
        all_rewards = torch.empty((total_steps, batch_size, n_agents, 1), device=self.device)
        all_log_probs = torch.empty((total_steps, batch_size, n_agents, probs_dim), device=self.device)

        # First Step
        for firm_id in order:
            state, actions, log_probs, revenue, costs = self.environment.step(firm_id)
            all_states[0, :, firm_id, :] = state
            all_log_probs[0, :, firm_id, :] = log_probs
            all_rewards[0, :, firm_id, :] = revenue - costs
            all_actions[0, :, firm_id] = actions
        # Other steps
        for step in range(1, total_steps):
            for firm_id in order:
                state, actions, log_probs, revenue, costs = self.environment.step(
                    firm_id,
                )
                all_states[step, :, firm_id, :] = state
                all_log_probs[step, :, firm_id, :] = log_probs
                all_rewards[step, :, firm_id, :] = -costs
                all_actions[step, :, firm_id] = actions
                all_rewards[step - 1, :, firm_id, :] += revenue

        for firm_id in order:
            state = self.environment.get_state(firm_id)
            if self.environment.mode == "finance":
                all_rewards[-1, :, firm_id] += self.environment.market.gains[:, firm_id].unsqueeze(-1)
            all_states[-1, :, firm_id, :] = state
        if self.environment.mode == "finance":
            all_rewards /= self.environment.market.start_gains
        all_values = self.get_critic_output(all_states.flatten(0, 1)).view(
            total_steps + 1,
            batch_size,
            n_agents,
        )

        if USE_VALUE_NORM:
            all_values_norm = all_values.unsqueeze(-1)
            all_values = self.value_norm.denormalize(all_values).unsqueeze(-1)
        else:
            all_values = all_values.unsqueeze(-1)

        rewards_to_show = all_rewards.clone()
        # Compute GAE
        # gae = 0
        # lambda_ = self.lambda_
        # for t in reversed(range(total_steps)):
        #     next_values = all_values[t + 1]
        #     current_rewards = all_rewards[t]
        #     current_values = all_values[t]
        #     delta = current_rewards + self.gamma * next_values - current_values
        #     all_advantages[t] = gae = delta + self.gamma * lambda_ * gae
        # GAE: advantages[t] = sum_{k >= t} (gamma * lambda)^(k - t) * delta[k]
        deltas = all_rewards + self.gamma * all_values[1:] - all_values[:-1]
        assert deltas.shape == all_rewards.shape, deltas.shape
        steps = torch.arange(total_steps, device=self.device)
        lag = steps[None, :] - steps[:, None]
        discount = ((self.gamma * self.lambda_) ** lag.clamp_min(0)) * (lag >= 0)
        all_advantages = torch.einsum("tk,k...->t...", discount.to(deltas.dtype), deltas)

        if USE_VALUE_NORM:
            returns = (all_advantages + all_values[:-1]).squeeze(-1)
            self.value_norm.update(returns.flatten(0, 1))  # сначала обновить статистики
            all_value_targets = self.value_norm.normalize(returns).unsqueeze(-1)
        else:
            all_value_targets = all_advantages + all_values[:-1]
        if self.normalize_advantages:
            adv_mean = all_advantages.mean(dim=(0, 1), keepdim=True)
            adv_std = all_advantages.std(dim=(0, 1), keepdim=True).clamp_min(0.1)
            all_advantages = (all_advantages - adv_mean) / adv_std
        trajectory = {
            "x": all_states[:-1],
            "actions": all_actions,
            "values": (all_values_norm if USE_VALUE_NORM else all_values)[:-1],
            "advantages": all_advantages,
            "value_targets": all_value_targets,
            "log_probs": all_log_probs,
        }
        # Permute Batch
        permutation = torch.randperm(total_steps * batch_size, device=self.device)
        for key in trajectory:
            trajectory[key] = trajectory[key].flatten(0, 1)[permutation]
        return trajectory, rewards_to_show.permute(0, 2, 1, 3)


class ValueNorm:
    def __init__(self, n_firms, beta=0.99, device="cpu"):
        self.beta = beta
        self.mean = torch.zeros(n_firms, device=device)
        self.mean_squared = torch.zeros(n_firms, device=device)
        self.debias = torch.zeros((), device=device)

    @torch.no_grad()
    def update(self, x):
        b = self.beta
        self.mean.mul_(b).add_((1 - b) * x.mean(0))
        self.mean_squared.mul_(b).add_((1 - b) * (x**2).mean(0))
        self.debias.mul_(b).add_(1 - b)

    def stats(self):
        d = self.debias.clamp_min(1e-5)
        mu = self.mean / d
        std = (self.mean_squared / d - mu**2).clamp_min(1e-2).sqrt()
        return mu, std

    def normalize(self, x):
        mu, s = self.stats()
        return (x - mu) / s

    def denormalize(self, x):
        mu, s = self.stats()
        return x * s + mu
