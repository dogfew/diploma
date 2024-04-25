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
            entropy_reg=0.01,
            learning_rates=(3e-4, 3e-4),
            buffer_size=8192 * 16,
            batch_size=32,
            max_grad_norm=1,
            device="cuda",
    ):
        super().__init__(environment)
        critic_lr, actor_lr = learning_rates

        market = self.environment.market
        state_dim, action_dim = self.environment.state_dim, get_action_dim(market, limit=environment.limit)
        n_firms = market.n_firms
        # Replay Buffer
        self.buffer = ReplayBuffer(
            state_dim=state_dim, action_dim=action_dim, n_firms=n_firms, size=buffer_size
        )

        # Critic that returns Q-value (1)
        self.critic_loss = nn.MSELoss()
        self.q_critic = q_critic(
            state_dim=state_dim, action_dim=action_dim, n_agents=n_firms
        ).to(device)
        self.q_critic_target = deepcopy(self.q_critic)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=critic_lr, weight_decay=1e-6
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
            self.q_critic2.parameters(), lr=critic_lr, weight_decay=1e-6
        )
        self.q_critic2_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.q_critic2_optimizer, 0.995
        )

        self.policies = self.environment.policies
        for policy in self.environment.policies:
            policy.to(device)

        # Actors
        self.actor_optimizers = [
            torch.optim.Adam(policy.parameters(), lr=actor_lr, weight_decay=1e-6)
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
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.device = device

    def train(self, n_episodes, episode_length=100, debug_period=5, change_order=False):
        pbar = tqdm(range(n_episodes))
        order = list(range(self.n_agents))
        for _ in pbar:
            df = self.train_epoch(max_episode_length=episode_length, order=order)
            df["episode"] = self.episode
            self.df_list.append(df)
            if self.episode % debug_period == 0:
                self.plot_loss(self.df_list)
            self.episode += 1
            if change_order:
                random.shuffle(order)
            pbar.set_postfix(
                {"LR": self.q_critic_scheduler.get_last_lr()[0],
                 "Buffer Index": self.buffer.index,
                 "Order": str(order)
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
                x, x_next, actions, rewards = map(
                    lambda i: i.to(self.device), self.buffer.sample(self.batch_size)
                )
                policies = self.policies

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
                        log_probs_firm.sum(dim=1) * self.entropy_reg - q_values
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
    def collect_experience(self, order, num_steps=50, do_not_skip=False):
        if order is None:
            order = range(len(self.environment.firms))
        rewards_lst = []
        to_add = {
            "actions": [],
            "states": [],
            "log_probs": [],
            'revenues': [],
            "costs": [],
            "next_states": [],
        }
        for firm_id in order:
            state, actions, log_probs, _, costs = self.environment.step(firm_id)
            to_add['states'].append(state)
            to_add['actions'].append(actions)
            to_add['log_probs'].append(log_probs)
            to_add['costs'].append(costs)
        tensor_lst = deque([to_add], maxlen=2)
        for step in range(num_steps):
            tensor_lst.append({
                "actions": [],
                "states": [],
                "log_probs": [],
                'revenues': [],
                "costs": [],
                "next_states": [],
            })
            for firm_id in order:
                state, actions, log_probs, prev_revenue, costs = self.environment.step(firm_id)

                tensor_lst[-2]['revenues'].append(prev_revenue)
                tensor_lst[-2]['next_states'].append(state)
                tensor_lst[-1]['states'].append(state)
                tensor_lst[-1]['actions'].append(actions)
                tensor_lst[-1]['log_probs'].append(log_probs)
                tensor_lst[-1]['costs'].append(costs)

            to_buffer = tensor_lst[-2]
            rewards = torch.stack(to_buffer["revenues"]) - torch.stack(to_buffer['costs'])
            rewards = rewards / self.environment.market.start_gains
            self.buffer.add_batch(
                x=torch.stack(to_buffer["states"], dim=1),
                x_next=torch.stack(to_buffer["next_states"], dim=1),
                actions=torch.stack(to_buffer["actions"], dim=1),
                rewards=rewards.permute(1, 0, 2),
            )
            rewards_lst.append(rewards)
        return rewards_lst

    @torch.no_grad()
    def collect_experience(self, order, num_steps=50, do_not_skip=False):
        if order is None:
            order = range(len(self.environment.firms))
        batch_size = self.environment.batch_size
        action_dim = self.environment.action_dim
        state_dim = self.environment.state_dim
        n_agents = self.environment.n_agents

        rewards_lst = []
        kwargs = dict(device=self.device)

        to_add = {
            "states": torch.empty((batch_size, n_agents, state_dim), **kwargs),
            "actions": torch.empty((batch_size, n_agents, action_dim), **kwargs),
            'rewards': torch.empty((batch_size, n_agents, 1),  **kwargs),
            "next_states": torch.empty((batch_size, n_agents, state_dim),  **kwargs ),
        }
        # First Steps
        for firm_id in order:
            state, actions, log_probs, _, costs = self.environment.step(firm_id)
            to_add['states'][:, firm_id] = state
            to_add['actions'][:, firm_id] = actions
            to_add['rewards'][:, firm_id] = -costs
        tensor_lst = deque([to_add], maxlen=2)
        # Other Steps. We do not record final step
        for step in range(num_steps):
            tensor_lst.append({
                'rewards': torch.empty((batch_size, n_agents, 1), **kwargs ),
                "next_states": torch.empty((batch_size, n_agents, state_dim), **kwargs ),
                "actions": torch.empty((batch_size, n_agents, action_dim),  **kwargs ),
                "states": torch.empty((batch_size, n_agents, state_dim), **kwargs ),
            }
            )
            for firm_id in order:
                state, actions, log_probs, prev_revenue, costs = self.environment.step(firm_id)

                tensor_lst[-2]['rewards'][:, firm_id, :] += prev_revenue
                tensor_lst[-2]['next_states'][:, firm_id] = state
                tensor_lst[-1]['states'][:, firm_id] = state
                tensor_lst[-1]['actions'][:, firm_id] = actions
            to_buffer = tensor_lst[-2]
            rewards = to_buffer['rewards'] / self.environment.market.start_gains
            self.buffer.add_batch(
                x=to_buffer['states'],
                x_next=to_buffer['next_states'],
                actions=to_buffer['actions'],
                rewards=rewards,
            )
            rewards_lst.append(rewards)
        return rewards_lst