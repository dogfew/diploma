import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from trainer.base_trainer import BaseTrainer
from trainer.replay_buffer import ReplayBuffer
from models.utils.preprocessing import get_state_dim, get_action_dim


class TrainerAC(BaseTrainer):
    def __init__(
        self,
        environment,
        q_critic,
        gamma=0.99,
        tau=0.95,
        learning_rates=(3e-4, 3e-4),
        batch_size=32,
        max_grad_norm=1,
        device="cuda",
    ):
        super().__init__(environment)

        market = self.environment.market
        limit = self.environment.limit
        critic_lr, actor_lr = learning_rates
        state_dim, action_dim = get_state_dim(market, limit), get_action_dim(
            market, limit
        )
        n_firms = market.n_firms
        # Replay Buffer
        self.buffer = ReplayBuffer(
            state_dim=state_dim, action_dim=action_dim, n_firms=n_firms
        )

        # Critic that returns Q-value
        self.critic_loss = nn.MSELoss()
        market = self.environment.market
        self.q_critic_network = q_critic(
            state_dim=state_dim, action_dim=action_dim, n_agents=n_firms
        ).to(device)
        self.q_critic_target_network = deepcopy(self.q_critic_network)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic_network.parameters(), lr=critic_lr, weight_decay=1e-6
        )
        self.q_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.q_critic_optimizer, 0.995
        )

        self.policies = self.environment.policies
        self.target_policies = self.environment.target_policies

        for policy in self.environment.policies:
            policy.to(device)
        for policy in self.environment.target_policies:
            policy.to(device)

        # Actors
        self.actor_optimizers = [
            torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-6)
            for policy in self.policies
        ]
        self.actor_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.995)
            for actor_optimizer in self.actor_optimizers
        ]

        # Plotting
        self.episode = 0
        self.df_list = []
        self.window_size = 10

        # Hyperparams
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.entropy_reg = 0.1
        self.batch_size = batch_size
        self.device = device

    def train(self, n_episodes, episode_length=100, debug_period=5):
        pbar = tqdm(range(n_episodes))
        for _ in pbar:
            df = self.train_epoch(max_episode_length=episode_length, order=None)
            df["episode"] = self.episode
            self.df_list.append(df)
            if self.episode % debug_period == 0:
                self.plot_loss(self.df_list)
            self.episode += 1

            pbar.set_postfix({"LR": f"{self.q_critic_scheduler.get_last_lr()[0]}"})

    def train_epoch(self, max_episode_length=10, order: list[int] = None):
        self.environment.reset()
        history = []
        for _ in range(max_episode_length):
            rewards_debug = self.collect_experience(order)
            for firm_id in range(self.n_agents):
                # Extract Batch and move it to device
                x, x_next, actions, rewards = map(
                    lambda i: i.to(self.device), self.buffer.sample(self.batch_size)
                )
                target_policies = self.target_policies
                policies = self.policies

                # Q-values
                with torch.no_grad():
                    actions_next, log_probs_next = self.get_actions(
                        x, target_policies, firm_id=firm_id
                    )
                    next_q_values = self.q_critic_target_network(x_next, actions_next)
                    q_values_target = (
                        rewards.squeeze(-1)[:, firm_id]
                        + self.gamma * next_q_values[:, firm_id]
                    )
                q_values = self.q_critic_network(x, actions)[:, firm_id]
                critic_loss = self.critic_loss(q_values, q_values_target)

                # Optimize for Centralized Critic
                self.q_critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.q_critic_network.parameters(), self.max_grad_norm
                )
                self.q_critic_optimizer.step()

                actions, log_probs = self.get_actions(x, policies, firm_id=firm_id)
                actions_firm, log_probs_firm = (
                    actions[:, firm_id, :],
                    log_probs[:, firm_id, :],
                )
                q_values = self.q_critic_network(x, actions)[:, firm_id]
                actor_loss = (
                    log_probs_firm * self.entropy_reg - q_values.unsqueeze(1)
                ).mean()
                # Optimize for Actor[firm_id]
                self.actor_optimizers[firm_id].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    policies[firm_id].parameters(), self.max_grad_norm
                )
                self.actor_optimizers[firm_id].step()

                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": critic_loss.item(),
                            "reward": rewards_debug[firm_id].mean().item(),
                            "firm_id": firm_id,
                        }
                    )
            self._soft_update_target_network()

        for actor_scheduler in self.actor_schedulers:
            actor_scheduler.step()
        self.q_critic_scheduler.step()
        return pd.DataFrame(history).groupby("firm_id").mean()

    @torch.no_grad()
    def _soft_update_target_network(self):
        for target_policy_network, policy_network in zip(
            self.target_policies, self.policies
        ):
            for target_param, param in zip(
                target_policy_network.parameters(), policy_network.parameters()
            ):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

        for target_param, param in zip(
            self.q_critic_target_network.parameters(),
            self.q_critic_network.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )
