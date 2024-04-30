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
from trainer.replay_buffer import ReplayBuffer, ReplayBufferPPO
from models.utils.preprocessing import get_state_dim, get_action_dim


class TrainerPPO(BaseTrainer):
    """
    Multi-Agent Proximal Policy Optimization Algorithm
    """

    def __init__(
            self,
            environment,
            critic,
            gamma=0.99,
            entropy_reg=0.05,
            learning_rates=(3e-4, 3e-4),
            critic_hidden_dim=64,
            buffer_size=8192 * 16,
            batch_size=3,
            lr_gamma=0.99,
            entropy_gamma=0.999,
            max_grad_norm=0.5,
            normalize_advantages=True,
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
        self.buffer = ReplayBufferPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            prob_dim=self.environment.probs_dim,
            n_firms=n_firms,
            size=buffer_size,
            device=device
        )

        # Critic that returns Q-value (1)
        # self.critic_loss = nn.MSELoss()
        self.critic = critic(
            state_dim=state_dim,
            n_agents=n_firms,
            hidden_dim=critic_hidden_dim
        ).to(device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=0
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=lr_gamma
        )

        self.policies = self.environment.policies
        for policy in self.environment.policies:
            policy.to(device)

        # Actors
        self.actor_optimizers = [
            torch.optim.Adam(list(policy.parameters()), lr=actor_lr, weight_decay=0)
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
        self.normalize_advantages = normalize_advantages
        self.cliprange = 0.2
        self.gamma = gamma
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
                    "LR": self.critic_scheduler.get_last_lr()[0],
                    "Buffer Index": self.buffer.index,
                    "Order": str(order),
                }
            )

    def train_epoch(self, max_episode_length=32, order: list[int] = None):
        self.environment.reset()
        history = []
        _, rewards_lst = self.get_trajectory(num_steps=max_episode_length, order=order)
        policies = self.policies

        for idx in range(max_episode_length):
            rewards_debug = rewards_lst[idx]
            for firm_id in range(self.n_agents):
                x, x_next, actions, values, all_advantages, value_targets, log_probs_old = self.buffer.sample(self.batch_size)
                # Compute Critic Loss
                v = self.critic(x).unsqueeze(-1)[:, firm_id]
                v_old = values[:, firm_id]
                v_hat = value_targets[:, firm_id]
                critic_loss = torch.maximum(
                    (v - v_hat).pow(2),
                    (v.clamp(v_old-self.cliprange, v_old+self.cliprange) - v_hat).pow(2)
                ).mean()
                # Optimize for Centralized Critic
                self.optimize(
                    critic_loss,
                    list(self.critic.parameters()),
                    self.critic_optimizer,
                )

                # Compute Actor Loss
                advantages = all_advantages[:, firm_id]
                if self.normalize_advantages:
                    advantages = ((advantages - advantages.mean())
                                  / (advantages.std().clamp_min(1e-6)))

                with torch.no_grad():
                    actions = self.environment.restore_actions(
                        actions[:, firm_id].squeeze()
                    )
                log_probs_new = torch.concatenate(
                    policies[firm_id].get_log_probs(x[:, firm_id], actions), dim=-1)
                ratios = torch.exp(log_probs_new - log_probs_old[:, firm_id, :])
                actor_loss = - torch.minimum(advantages * ratios,
                                             advantages * ratios.clamp(1-self.cliprange,
                                                                       1+self.cliprange)).mean()
                entropy_loss = - log_probs_new.mean()
                # Optimize for Actor[firm_id]
                self.optimize(
                    actor_loss + entropy_loss * self.entropy_reg,
                    list(policies[firm_id].parameters()),
                    self.actor_optimizers[firm_id],
                )

                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": critic_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "reward": rewards_debug[firm_id].mean().item(),
                            "firm_id": firm_id,
                        }
                    )
        for actor_scheduler in self.actor_schedulers:
            actor_scheduler.step()
        self.critic_scheduler.step()
        self.entropy_reg *= self.entropy_gamma
        return pd.DataFrame(history).groupby("firm_id").mean()

    @torch.no_grad()
    def _clip_grad_norm(self, model, norm_type=2):
        try:
            nn.utils.clip_grad_norm_(
                model,
                self.max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=True,
            )
        except RuntimeError:
            return False
        return True

    def optimize(self, loss, params, optimizer):
        optimizer.zero_grad()
        loss.backward()

        if self._clip_grad_norm(params):
            optimizer.step()
        else:
            print("Got NaN gradients")

    @torch.no_grad()
    def get_trajectory(self, order=None, num_steps=50):
        self.environment.reset()
        if order is None:
            order = range(len(self.environment.firms))

        batch_size = self.environment.batch_size
        state_dim = self.environment.state_dim
        probs_dim = self.environment.probs_dim
        action_dim = self.environment.action_dim
        n_agents = self.environment.n_agents

        total_steps = num_steps + 1
        kwargs = dict(device=self.device)

        all_advantages = torch.empty((total_steps, batch_size, n_agents, 1), **kwargs)
        all_actions = torch.empty((total_steps, batch_size, n_agents, action_dim), **kwargs)
        all_states = torch.empty((total_steps+1, batch_size, n_agents, state_dim), **kwargs)
        all_rewards = torch.empty((total_steps, batch_size, n_agents, 1), **kwargs)
        all_log_probs = torch.empty((total_steps, batch_size, n_agents, probs_dim), **kwargs)
        all_values = torch.empty((total_steps + 1, batch_size, n_agents), **kwargs)

        # First Step
        for firm_id in order:
            state, actions, log_probs, revenue, costs = self.environment.step(firm_id)
            all_states[0, :, firm_id, :] = state
            all_log_probs[0, :, firm_id, :] = log_probs
            all_rewards[0, :, firm_id, :] = revenue - costs
            all_actions[0, :, firm_id] = actions
        all_values[0] = self.critic(all_states[0])
        # Other steps
        for step in range(1, total_steps):
            for firm_id in order:
                state, actions, log_probs, revenue, costs = self.environment.step(
                    firm_id
                )
                all_states[step, :, firm_id, :] = state
                all_log_probs[step, :, firm_id, :] = log_probs
                all_rewards[step, :, firm_id, :] = -costs
                all_actions[step, :, firm_id] = actions
                all_rewards[step - 1, :, firm_id, :] += revenue

            all_values[step] = self.critic(all_states[step])

        for firm_id in order:
            # state, _, _, _, _ = self.environment.step(
            #     firm_id
            # )
            state = get_state_log(self.environment.market,
                                  self.environment.firms[firm_id])
            all_states[-1, :, firm_id, :] = state
        all_values[-1] = self.critic(all_states[-1])
        all_values = all_values.unsqueeze(-1)
        rewards_to_show = all_rewards.clone()
        # all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std().clamp_min(1e-6))
        # all_rewards /= self.environment.market.start_gains
        # Compute GAE
        gae = 0
        lambda_ = 0.95
        for t in reversed(range(total_steps)):
            next_values = all_values[t + 1]
            current_rewards = all_rewards[t]
            current_values = all_values[t]

            delta = current_rewards + self.gamma * next_values - current_values
            all_advantages[t] = gae = delta + self.gamma * lambda_ * gae

        all_value_targets = all_advantages + all_values[:-1]
        trajectory = dict(
            x=all_states[:-1],
            x_next=all_states[1:],
            actions=all_actions,
            values=all_values[:-1],
            advantages=all_advantages,
            value_targets=all_value_targets,
            log_probs=all_log_probs
        )
        # Permute Batch
        for key in trajectory:
            trajectory[key] = trajectory[key].flatten(0, 1)
        self.buffer.add_batch(trajectory)
        return trajectory, rewards_to_show.permute(0, 2, 1, 3)