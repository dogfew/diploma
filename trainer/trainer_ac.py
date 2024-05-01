import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from models.critic import CentralizedCriticQ
from trainer.base_trainer import BaseTrainer
from trainer.replay_buffer import ReplayBuffer
from models.utils.preprocessing import get_state_dim, get_action_dim


class TrainerAC(BaseTrainer):
    """
    Soft Actor Critic Algorithm
    """

    def __init__(
            self,
            environment,
            gamma=0.99,
            tau=0.95,
            entropy_reg=0.05,
            learning_rates=(3e-4, 3e-4),
            critic_hidden_dim=64,
            buffer_size=8192,
            batch_size=512,
            lr_gamma=0.98,
            entropy_gamma=0.999,
            max_grad_norm=1,
            critic_loss=F.smooth_l1_loss,
            device="cuda",
    ):
        super().__init__(environment,
                         entropy_gamma=entropy_gamma,
                         gamma=gamma,
                         tau=tau,
                         critic_loss=critic_loss,
                         max_grad_norm=max_grad_norm,
                         entropy_reg=entropy_reg,
                         batch_size=batch_size,
                         device=device)
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

        # Critic that returns Q-value
        self.critic_loss = critic_loss
        self.critic = CentralizedCriticQ(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_firms,
            hidden_dim=critic_hidden_dim
        ).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=0
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=lr_gamma
        )

        self.policies = self.environment.policies
        self.target_policies = self.environment.target_policies

        for policy in self.environment.policies:
            policy.to(device)
        for policy in self.environment.target_policies:
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

    def train_epoch(self, max_episode_length=10, order: list[int] = None):
        self.environment.reset()
        history = []
        rewards_lst = self.collect_experience(num_steps=max_episode_length, order=order)
        for _ in range(max_episode_length):
            for firm_id in range(self.n_agents):
                # Extract Batch and move it to device
                x, x_next, actions, rewards = self.buffer.sample(self.batch_size)
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
                target_policies = self.target_policies
                policies = self.policies

                # Q-values
                with torch.no_grad():
                    actions_next, log_probs_next = self.get_actions(
                        x, target_policies, firm_id=firm_id
                    )
                    next_entropy = log_probs_next[:, firm_id].mean(dim=-1)
                    next_q_values = (self.critic_target(x_next, actions_next)[:, firm_id]
                                     - self.entropy_reg * next_entropy
                                     )
                    q_values_target = (
                            rewards.squeeze(-1)[:, firm_id]
                            + self.gamma * next_q_values
                    )
                q_values = self.critic(x, actions)[:, firm_id]
                critic_loss = self.critic_loss(q_values, q_values_target)

                # Optimize for Centralized Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()

                actions, log_probs = self.get_actions(x, policies, firm_id=firm_id)
                actions_firm, log_probs_firm = (
                    actions[:, firm_id, :],
                    log_probs[:, firm_id, :],
                )
                q_values = self.critic(x, actions)[:, firm_id]
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
                            "entropy_loss": -log_probs_firm.mean().item(),
                            "firm_id": firm_id,
                        }
                    )
            self._soft_update_target_network()

        for actor_scheduler in self.actor_schedulers:
            actor_scheduler.step()
        self.critic_scheduler.step()
        self.entropy_reg *= self.entropy_gamma
        df_out = pd.DataFrame(history).groupby("firm_id").mean()
        df_out['reward'] = rewards_lst.mean(dim=(0, 2)).flatten().cpu().numpy()
        return df_out

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
                self.critic_target.parameters(),
                self.critic.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )
