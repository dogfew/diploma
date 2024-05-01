import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from environment_batched.utils import get_state_log
from models.critic import CentralizedCriticV
from models.critic.centralized_q_critic import CentralizedCriticQ
from trainer.base_trainer import BaseTrainer
from trainer.replay_buffer import ReplayBuffer
from models.utils.preprocessing import get_state_dim, get_action_dim


class TrainerMASAC(BaseTrainer):
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
            batch_size=32,
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
                         max_grad_norm=max_grad_norm,
                         entropy_reg=entropy_reg,
                         batch_size=batch_size,
                         critic_loss=critic_loss,
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

        # Critic that returns V-value
        self.v_critic = CentralizedCriticV(state_dim=state_dim,
                                           n_agents=n_firms,
                                           hidden_dim=critic_hidden_dim).to(device)
        self.v_critic_target = deepcopy(self.v_critic)
        self.v_critic_optimizer = torch.optim.Adam(
            self.v_critic.parameters(), lr=critic_lr, weight_decay=0
        )
        self.v_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.v_critic_optimizer, gamma=lr_gamma
        )

        # Critic that returns Q-value (1)
        self.critic_loss = nn.MSELoss()
        self.q_critic = CentralizedCriticQ(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_firms,
            hidden_dim=critic_hidden_dim
        ).to(device)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=critic_lr, weight_decay=0
        )
        self.q_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.q_critic_optimizer, lr_gamma
        )

        # Critic that returns Q-value (2)
        self.q_critic2 = CentralizedCriticQ(
            state_dim=state_dim, action_dim=action_dim, n_agents=n_firms
        ).to(device)
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
            torch.optim.Adam(policy.parameters(),
                             lr=actor_lr,
                             weight_decay=0)
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
        policies = self.policies
        for idx in range(max_episode_length):
            rewards_debug = rewards_lst[idx]
            x, x_next, actions, rewards = self.buffer.sample(self.batch_size)
            for firm_id in range(self.n_agents):
                # Extract Batch and move it to device
                # Compute Values
                actions_new, log_probs_new = self.get_actions(x, policies, firm_id=firm_id)
                entropy = log_probs_new[:, firm_id].mean(dim=-1)
                q1_pred = self.q_critic(x, actions)[:, firm_id]
                q2_pred = self.q_critic2(x, actions)[:, firm_id]
                q_pred = torch.minimum(
                    self.q_critic(x, actions_new)[:, firm_id],
                    self.q_critic2(x, actions_new)[:, firm_id]
                )
                v_pred = self.v_critic(x)[:, firm_id]
                advantage = q_pred - v_pred.detach()
                with torch.no_grad():
                    q_target = (
                            rewards[:, firm_id].squeeze(-1)
                            + self.gamma * self.v_critic_target(x_next)[:, firm_id]
                    )
                    v_target = q_pred - self.entropy_reg * entropy

                # Compute Critic Loss (1)
                q1_loss = self.critic_loss(q1_pred, q_target)
                q2_loss = self.critic_loss(q2_pred, q_target)
                v_loss = self.critic_loss(v_pred, v_target)
                actor_loss = (
                        entropy * self.entropy_reg - advantage
                ).mean()

                # Optimize for Actor[firm_id]
                self.optimize(
                    actor_loss,
                    policies[firm_id].parameters(),
                    self.actor_optimizers[firm_id],
                )
                # Optimize for Centralized Q-Critic (1)
                self.optimize(
                    q1_loss,
                    self.q_critic.parameters(),
                    self.q_critic_optimizer,
                )

                # Optimize for Centralized Q-Critic (2)
                self.optimize(
                    q2_loss,
                    self.q_critic2.parameters(),
                    self.q_critic2_optimizer,
                )

                # Optimize for Centralized V-Critic
                self.optimize(
                    v_loss,
                    self.v_critic.parameters(),
                    self.v_critic_optimizer,
                )

                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": (v_loss + q1_loss + q2_loss).item(),
                            "entropy_loss": - entropy.mean().item(),
                            "firm_id": firm_id,
                        }
                    )
            self._soft_update_target_network()

        for actor_scheduler in self.actor_schedulers:
            actor_scheduler.step()
        self.q_critic_scheduler.step()
        self.entropy_reg *= self.entropy_gamma
        df_out = pd.DataFrame(history).groupby("firm_id").mean()
        df_out['reward'] = rewards_lst.mean(dim=(0, 2)).flatten().cpu().numpy()
        return df_out

    @torch.no_grad()
    def _soft_update_target_network(self):
        for target_param, param in zip(
                self.v_critic_target.parameters(),
                self.v_critic.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )
