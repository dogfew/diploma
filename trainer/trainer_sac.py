import random
from collections import deque

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from environment_batched.utils import get_state_log
from trainer.base_trainer import BaseTrainer
from trainer.replay_buffer import ReplayBuffer
from models.utils.preprocessing import get_state_dim, get_action_dim


class TrainerSAC(BaseTrainer):
    """
    Soft Actor Critic Algorithm
    """

    def __init__(
        self,
        environment,
        q_critic,
        gamma=0.99,
        tau=0.95,
        entropy_reg=0.05,
        learning_rates=(3e-4, 3e-4),
        critic_hidden_dim=64,
        buffer_size=8192,
        batch_size=32,
        lr_gamma=0.98,
        entropy_gamma=0.999,
        max_grad_norm=1,
        device="cuda",
    ):
        super().__init__(environment)
        critic_lr, actor_lr = learning_rates

        market = self.environment.market
        state_dim, action_dim = self.environment.state_dim, get_action_dim(
            market, limit=environment.limit
        )
        n_firms = market.n_firms
        # Replay Buffer
        self.buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            n_firms=n_firms,
            size=buffer_size,
        )

        # Critic that returns Q-value (1)
        self.critic_loss = nn.MSELoss()
        self.q_critic = q_critic(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_firms,
            hidden_dim=critic_hidden_dim
        ).to(device)
        self.q_critic_target = deepcopy(self.q_critic)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=critic_lr, weight_decay=0
        )
        self.q_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.q_critic_optimizer, 0.995
        )
        # Critic that returns Q-value (2)
        self.q_critic2 = q_critic(
            state_dim=state_dim, action_dim=action_dim, n_agents=n_firms
        ).to(device)
        self.q_critic2_target = deepcopy(self.q_critic2)
        self.q_critic2_optimizer = torch.optim.Adam(
            self.q_critic2.parameters(), lr=critic_lr, weight_decay=0
        )
        self.q_critic2_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.q_critic2_optimizer, gamma=lr_gamma
        )

        self.policies = self.environment.policies
        for policy in self.environment.policies:
            policy.to(device)

        # Actors
        self.actor_optimizers = [
            torch.optim.Adam(policy.parameters(), lr=actor_lr, weight_decay=0)
            for policy in self.policies
        ]
        self.actor_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=lr_gamma)
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
        self.entropy_reg = entropy_reg
        self.entropy_gamma = entropy_gamma
        self.batch_size = batch_size
        self.device = device

    def train(self, n_epochs, episode_length=100, debug_period=5, shuffle_order=False):
        """
        :param n_epochs: Number of Epochs to train model
        :param episode_length: Number of environment full steps per epoch
        :param debug_period: how often to update plot
        :param shuffle_order: whether to shuffle agent's order
        :return:
        """
        pbar = tqdm(range(n_epochs))
        order = list(range(self.n_agents))
        for _ in pbar:
            df = self.train_epoch(max_episode_length=episode_length, order=order)
            df["episode"] = self.episode
            self.df_list.append(df)
            if self.episode % debug_period == 0:
                self.plot_loss(self.df_list)
            self.episode += 1
            if shuffle_order:
                random.shuffle(order)
            pbar.set_postfix(
                {
                    "LR": self.q_critic_scheduler.get_last_lr()[0],
                    "Buffer Index": self.buffer.index,
                    "Order": str(order),
                }
            )

    def train_epoch(self, max_episode_length=10, order: list[int] = None):
        self.environment.reset()
        history = []
        rewards_lst = self.collect_experience(num_steps=max_episode_length, order=order)
        for idx in range(max_episode_length):
            rewards_debug = rewards_lst[idx]
            for firm_id in range(self.n_agents):
                # Extract Batch and move it to device
                x, x_next, actions, rewards = self.buffer.sample(self.batch_size)
                policies = self.policies

                # # Normalize rewards
                # rewards = (rewards - rewards.mean(dim=0, keepdims=True)) / (rewards.std(dim=0, keepdims=True) + 1e-6)

                # Compute Critic Target
                with torch.no_grad():
                    actions_next, log_probs_next = self.get_actions(
                        x_next, policies, firm_id=firm_id
                    )
                    next_q_values1 = self.q_critic_target(x_next, actions_next)
                    next_q_values2 = self.q_critic2_target(x_next, actions_next)
                    next_v = torch.minimum(
                        next_q_values1, next_q_values2
                    ) - self.entropy_reg * log_probs_next.mean(dim=-1)
                    q_values_target = (
                        rewards.squeeze(-1)[:, firm_id]
                        + self.gamma * next_v[:, firm_id]
                    )
                # Compute Critic Loss (1)
                q_values1 = self.q_critic(x, actions)[:, firm_id]
                critic_loss = self.critic_loss(q_values1, q_values_target)

                # Optimize for Centralized Critic (1)
                self.optimize(
                    critic_loss,
                    self.q_critic,
                    self.q_critic_optimizer,
                )

                # Compute Critic Loss (2)
                q_values2 = self.q_critic2(x, actions)[:, firm_id]
                critic_loss = self.critic_loss(q_values2, q_values_target)

                # Optimize for Centralized Critic (2)
                self.optimize(
                    critic_loss,
                    self.q_critic2,
                    self.q_critic2_optimizer,
                )

                # Compute Actor Loss
                actions, log_probs = self.get_actions(x, policies, firm_id=firm_id)
                actions_firm, log_probs_firm = (
                    actions[:, firm_id, :],
                    log_probs[:, firm_id, :],
                )
                q_values1 = self.q_critic(x, actions)[:, firm_id]
                q_values2 = self.q_critic2(x, actions)[:, firm_id]
                q_values = torch.minimum(q_values1, q_values2)
                actor_loss = (
                    log_probs_firm.mean(dim=1) * self.entropy_reg - q_values
                ).mean()

                # Optimize for Actor[firm_id]
                self.optimize(
                    actor_loss,
                    policies[firm_id],
                    self.actor_optimizers[firm_id],
                )

                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": critic_loss.item(),
                            "entropy_loss":  log_probs_firm.mean().item(),
                            "reward": rewards_debug[firm_id].mean().item(),
                            "firm_id": firm_id,
                        }
                    )
            self._soft_update_target_network()

        for actor_scheduler in self.actor_schedulers:
            actor_scheduler.step()
        self.q_critic_scheduler.step()
        self.entropy_reg *= self.entropy_gamma
        return pd.DataFrame(history).groupby("firm_id").mean()

    @torch.no_grad()
    def _clip_grad_norm(self, model, norm_type=2):
        try:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=True,
            )
        except RuntimeError:
            return False
        return True

    def optimize(self, loss, model, optimizer):
        optimizer.zero_grad()
        loss.backward()
        if self._clip_grad_norm(model):
            optimizer.step()
        else:
            print("Got NaN gradients", model.__class__.__name__)

    @torch.no_grad()
    def _soft_update_target_network(self):
        for target_param, param in zip(
            self.q_critic_target.parameters(),
            self.q_critic.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )

        for target_param, param in zip(
            self.q_critic2_target.parameters(),
            self.q_critic2.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )

    @torch.no_grad()
    def collect_experience(self, order, num_steps=50):
        if order is None:
            order = range(len(self.environment.firms))

        batch_size = self.environment.batch_size
        action_dim = self.environment.action_dim
        state_dim = self.environment.state_dim
        n_agents = self.environment.n_agents

        total_steps = num_steps + 1
        kwargs = dict(device=self.device)

        all_states = torch.empty((total_steps, batch_size, n_agents, state_dim), **kwargs)
        all_actions = torch.empty((total_steps, batch_size, n_agents, action_dim), **kwargs)
        all_rewards = torch.empty((total_steps, batch_size, n_agents, 1), **kwargs)

        # First Step
        for firm_id in order:
            state, actions, log_probs, revenue, costs = self.environment.step(firm_id)
            all_states[0, :, firm_id, :] = state
            all_actions[0, :, firm_id, :] = actions
            all_rewards[0, :, firm_id, :] = revenue - costs

        # Other steps
        for step in range(1, total_steps):
            for firm_id in order:
                state, actions, log_probs, revenue, costs = self.environment.step(
                    firm_id
                )
                all_states[step, :, firm_id, :] = state
                all_actions[step, :, firm_id, :] = actions
                all_rewards[step, :, firm_id, :] = -costs

                all_rewards[step - 1, :, firm_id, :] += revenue

        self.buffer.add_batch(
            x=all_states[:-1].flatten(0, 1),
            x_next=all_states[1:].flatten(0, 1),
            actions=all_actions[:-1].flatten(0, 1),
            rewards=all_rewards[:-1].flatten(0, 1) / self.environment.market.start_gains,
        )
        return all_rewards.permute(0, 2, 1, 3)
