import random
from collections import deque

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy

from environment_batched.utils import get_state_log
from models.critic import CentralizedCriticV
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
            entropy_reg=0.01,
            learning_rates=(3e-4, 3e-4),
            critic_hidden_dim=64,
            buffer_size=8192 * 16,
            batch_size=512,
            gamma=0.99,
            lr_gamma=0.99,
            entropy_gamma=0.999,
            max_grad_norm=0.5,
            shared_weights=True,
            social_planner=False,
            normalize_advantages=True,
            common_optimizer=True,
            critic_loss=F.smooth_l1_loss,
            use_buffer=False,
            device="cuda",
    ):
        super().__init__(environment,
                         entropy_gamma=entropy_gamma,
                         gamma=gamma,
                         max_grad_norm=max_grad_norm,
                         entropy_reg=entropy_reg,
                         critic_loss=critic_loss,
                         batch_size=batch_size,
                         device=device)
        critic_lr, actor_lr = learning_rates

        market = self.environment.market
        state_dim, action_dim = self.environment.state_dim, get_action_dim(
            market, limit=environment.limit
        )
        n_firms = market.n_firms
        self.use_buffer = use_buffer
        if self.use_buffer:
            # Replay Buffer
            self.buffer = ReplayBufferPPO(
                state_dim=state_dim,
                action_dim=action_dim,
                prob_dim=self.environment.probs_dim,
                n_firms=n_firms,
                size=buffer_size,
                device=device
            )

        # Critic that returns V-value
        self.critic_loss = critic_loss
        self.shared_weights = shared_weights
        if self.shared_weights:
            self.critic = CentralizedCriticV(
                state_dim=state_dim,
                n_agents=n_firms,
                hidden_dim=critic_hidden_dim,
                social_planner=social_planner
            ).to(device)
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=critic_lr, weight_decay=0
            )
            self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.critic_optimizer, gamma=lr_gamma
            )
        else:
            self.critics = [
                CentralizedCriticV(
                    state_dim=state_dim,
                    n_agents=n_firms,
                    hidden_dim=critic_hidden_dim,
                    social_planner=social_planner
                ).to(device) for _ in self.policies
            ]
            self.critic_optimizers = [
                torch.optim.Adam(
                    critic.parameters(), lr=critic_lr, weight_decay=0
                ) for critic in self.critics
            ]

        self.policies = self.environment.policies
        for policy in self.environment.policies:
            policy.to(device)

        # Actors
        if shared_weights:
            actor_parameters = [
                list(policy.parameters()) + list(self.critic.parameters()) for policy in self.policies
            ]
        else:
            actor_parameters = [
                list(policy.parameters()) + list(critic.parameters()) for critic, policy in zip(self.critics,
                                                                                                self.policies)
            ]
        self.actor_optimizers = [
            torch.optim.Adam(
                params
                # if common_optimizer else policy.parameters()
                ,
                lr=actor_lr, weight_decay=0)
            for params in actor_parameters
        ]
        self.actor_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=lr_gamma)
            for actor_optimizer in self.actor_optimizers
        ]
        self.actor_schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='max', factor=0.33)
            for actor_optimizer in self.actor_optimizers
        ]
        # Hyperparams
        self.common_optimizer = common_optimizer
        self.normalize_advantages = normalize_advantages
        self.cliprange = 0.2
        self.lambda_ = 0.95


    @property
    def last_lr(self):
        return self.actor_schedulers[0].get_last_lr()[0]

    def train_epoch(self, max_episode_length=32, order: list[int] = None):
        self.environment.reset()
        history = []
        trajectory, rewards_lst = self.get_trajectory(num_steps=max_episode_length, order=order)
        policies = self.policies

        for idx in range(max_episode_length):
            if self.use_buffer:
                # Sample from buffer
                sampled = self.buffer.sample(self.batch_size)
            else:
                sampled = map(
                    lambda y: trajectory.get(y)[idx * self.batch_size: (idx + 1) * self.batch_size],
                    ['x', 'x_next', 'actions', 'values', 'advantages', 'value_targets', 'log_probs']
                )
            x, x_next, actions, values, all_advantages, value_targets, log_probs_old = sampled
            for firm_id in range(self.n_agents):
                # Compute Critic Loss
                if self.shared_weights:
                    critic = self.critic
                    critic_optimizer = self.critic_optimizer
                else:
                    critic = self.critics[firm_id]
                    critic_optimizer = self.critic_optimizers[firm_id]
                v = critic(x).unsqueeze(-1)[:, firm_id]
                v_old = values[:, firm_id]
                v_hat = value_targets[:, firm_id]
                critic_loss = torch.maximum(
                    self.critic_loss(v, v_hat, reduction='none'),
                    self.critic_loss(v.clamp(v_old - self.cliprange, v_old + self.cliprange), v_hat, reduction='none')
                ).mean()

                # Compute Actor Loss
                advantages = all_advantages[:, firm_id]
                if self.normalize_advantages:
                    advantages = ((advantages - advantages.mean())
                                  / (advantages.std().clamp_min(1e-6)))

                actions_restored = self.environment.restore_actions(
                    actions[:, firm_id].squeeze()
                )
                log_probs_new = torch.concatenate(
                    policies[firm_id].get_log_probs(
                        state=x[:, firm_id],
                        actions=actions_restored),
                    dim=-1)
                ratios = torch.exp(log_probs_new - log_probs_old[:, firm_id, :])
                actor_loss = - torch.minimum(advantages * ratios,
                                             advantages * ratios.clamp(1 - self.cliprange,
                                                                       1 + self.cliprange)).mean()
                entropy_loss = - log_probs_new.mean()
                if self.common_optimizer:
                    self.optimize(
                        loss=critic_loss + actor_loss + entropy_loss * self.entropy_reg,
                        params=list(policies[firm_id].parameters()) + list(critic.parameters()),
                        optimizer=self.actor_optimizers[firm_id],
                    )
                else:
                    self.optimize(critic_loss,
                                  critic.parameters(),
                                  critic_optimizer)
                    self.optimize(actor_loss + entropy_loss * self.entropy_reg,
                                  policies[firm_id].parameters(),
                                  self.actor_optimizers[firm_id])

                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": critic_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "firm_id": firm_id,
                        }
                    )
        rewards = rewards_lst.mean(dim=(0, 2)).flatten().cpu().numpy()

        for actor_scheduler, reward in zip(self.actor_schedulers, rewards):
            actor_scheduler.step(reward)
        if not self.common_optimizer:
            self.critic_scheduler.step()
        self.entropy_reg *= self.entropy_gamma

        df_out = pd.DataFrame(history).groupby("firm_id").mean()
        df_out['reward'] = rewards
        return df_out

    def get_critic_output(self, state):
        if self.shared_weights:
            return self.critic(state)
        else:
            critics_out = torch.stack([self.critics[i](state)[:, i] for i in range(len(self.critics))])
            return critics_out.T

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
        all_states = torch.empty((total_steps + 1, batch_size, n_agents, state_dim), **kwargs)
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
        all_values[0] = self.get_critic_output(all_states[0])
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
            all_values[step] = self.get_critic_output(all_states[step])

        for firm_id in order:
            state = self.environment.get_state(firm_id)
            if self.environment.mode == 'finance':
                all_rewards[-1, :, firm_id] += self.environment.market.gains[:, firm_id].unsqueeze(-1)
            all_states[-1, :, firm_id, :] = state
        if self.environment.mode == 'finance':
            all_rewards /= self.environment.market.start_gains
        all_values[-1] = self.get_critic_output(all_states[-1])
        all_values = all_values.unsqueeze(-1)
        rewards_to_show = all_rewards.clone()
        # Compute GAE
        gae = 0
        lambda_ = self.lambda_
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
        permutation = torch.randperm(total_steps * batch_size)
        for key in trajectory:
            trajectory[key] = trajectory[key].flatten(0, 1)[permutation]
        if self.use_buffer:
            self.buffer.add_batch(trajectory)
        return trajectory, rewards_to_show.permute(0, 2, 1, 3)
