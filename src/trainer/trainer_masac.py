from copy import deepcopy

import pandas as pd
import torch
import torch.nn.functional as F

# from src.models.critic import DecentralizedCriticV
from src.models.critic.centralized_q_critic import DecentralizedCriticQ
from src.models.critic.centralized_v_critic import DecentralizedCriticV

# from src.models.utils.preprocessing import get_action_dim
from src.trainer.base_trainer import BaseTrainer
from src.trainer.replay_buffer import ReplayBuffer


class TrainerMASAC(BaseTrainer[ReplayBuffer]):
    """
    Soft Actor Critic Algorithm
    """

    def __init__(
        self,
        environment,
        gamma=0.99,
        tau=0.005,
        entropy_reg=-10,  # аккуратно с этим параметром.
        learning_rates=(3e-4, 3e-4),
        critic_hidden_dim=64,
        buffer_size=8192 * 16,
        batch_size: int = 1024,
        max_grad_norm=1,
        critic_loss=F.smooth_l1_loss,
        device="cuda",
    ):
        super().__init__(
            environment,
            gamma=gamma,
            tau=tau,
            max_grad_norm=max_grad_norm,
            entropy_reg=entropy_reg,
            critic_loss=critic_loss,
            batch_size=batch_size,
            device=device,
        )
        critic_lr, actor_lr = learning_rates
        market = self.environment.market
        state_dim, action_dim = self.environment.state_dim, self.environment.action_dim
        n_firms = market.n_firms
        # Replay Buffer
        self.buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            n_firms=n_firms,
            size=buffer_size,
            device=device,
        )

        # Critic that returns V-value
        self.v_critic = DecentralizedCriticV(
            state_dim=state_dim,
            n_agents=n_firms,
            hidden_dim=critic_hidden_dim,
        ).to(device)
        self.v_critic_target = deepcopy(self.v_critic)
        self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=critic_lr, weight_decay=0)
        # self.v_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.v_critic_optimizer,
        #     gamma=lr_gamma,
        # )

        # Critic that returns Q-value (1)
        self.critic_loss = critic_loss
        self.q_critic = DecentralizedCriticQ(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_firms,
            hidden_dim=critic_hidden_dim,
        ).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=critic_lr, weight_decay=0)
        # self.q_critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.q_critic_optimizer, lr_gamma)

        # Critic that returns Q-value (2)
        self.q_critic2 = DecentralizedCriticQ(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            n_agents=n_firms,
        ).to(
            device,
        )
        self.q_critic2_optimizer = torch.optim.Adam(self.q_critic2.parameters(), lr=critic_lr, weight_decay=0)
        # self.q_critic2_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.q_critic2_optimizer,
        #     gamma=lr_gamma,
        # )

        self.policies = self.environment.policies
        for policy in self.environment.policies:
            policy.to(device)

        # Actors
        self.actor_optimizer = torch.optim.Adam(
            [{"params": policy.parameters()} for policy in self.policies],
            lr=actor_lr,
            weight_decay=0,
        )
        # self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=lr_gamma)
        # self.actor_optimizers = [
        #     torch.optim.Adam(policy.parameters(), lr=actor_lr, weight_decay=0) for policy in self.policies
        # ]
        # self.actor_schedulers = [
        #     torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=lr_gamma)
        #     for actor_optimizer in self.actor_optimizers
        # ]
        # Alpha
        self.log_alphas = [torch.zeros((), device=device, requires_grad=True) for _ in range(n_firms)]
        self.alpha_optimizers = [torch.optim.Adam([la], lr=3e-4) for la in self.log_alphas]
        # self.log_alpha = torch.zeros(n_firms, device=device, requires_grad=True)
        # self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = entropy_reg

    def train_epoch(self, max_episode_length=10, order: list[int] | None = None):
        self.environment.reset()
        history = []
        rewards_lst = self.collect_experience(num_steps=max_episode_length, order=order)
        policies = self.policies
        for _idx in range(max_episode_length):
            # rewards_debug = rewards_lst[idx]
            x, x_next, actions, rewards = self.buffer.sample(self.batch_size)
            rewards = (rewards - rewards.mean(dim=0, keepdims=True)) / (
                rewards.std(dim=0, keepdims=True).clamp(min=1e-6)
            )
            total_q1_loss = total_q2_loss = total_v_loss = total_actor_loss = 0
            q1_all = self.q_critic(x, actions)
            q2_all = self.q_critic2(x, actions)
            v_all = self.v_critic(x)
            with torch.no_grad():
                q_target_all = rewards.squeeze(-1) + self.gamma * self.v_critic_target(x_next)

            for firm_id in range(self.n_agents):
                # Extract Batch and move it to device
                # # # Normalize rewards
                # rewards = (rewards - rewards.mean(dim=0, keepdims=True)) / (
                #     rewards.std(dim=0, keepdims=True).clamp(min=1e-6)
                #

                # Compute Values
                actions_new, log_probs_new = self.get_actions(x, policies, firm_id=firm_id)
                # entropy = log_probs_new[:, firm_id].mean(dim=-1)
                log_pi = log_probs_new[:, firm_id].sum(dim=-1)
                alpha = self.log_alphas[firm_id].exp().detach()
                q1_pred = q1_all[:, firm_id]
                q2_pred = q2_all[:, firm_id]
                # q1_pred = self.q_critic(x, actions)[:, firm_id]
                # q2_pred = self.q_critic2(x, actions)[:, firm_id]
                q_pred = torch.minimum(
                    self.q_critic(x, actions_new)[:, firm_id],
                    self.q_critic2(x, actions_new)[:, firm_id],
                )
                v_pred = v_all[:, firm_id]
                # v_pred = self.v_critic(x)[:, firm_id]
                advantage = q_pred - v_pred.detach()
                q_target = q_target_all[:, firm_id]
                with torch.no_grad():
                    # q_target = (
                    #     rewards[:, firm_id].squeeze(-1)
                    #     + self.gamma * self.v_critic_target(x_next)[:, firm_id]
                    # )
                    v_target = q_pred - alpha * log_pi

                # Compute Critic Loss (1)
                q1_loss = self.critic_loss(q1_pred, q_target)
                q2_loss = self.critic_loss(q2_pred, q_target)
                v_loss = self.critic_loss(v_pred, v_target)
                actor_loss = (alpha * log_pi - advantage).mean()
                alpha_loss = -(self.log_alphas[firm_id] * (log_pi.detach() + self.target_entropy)).mean()
                # Optimize for Actor[firm_id]

                total_actor_loss += actor_loss
                total_q1_loss += q1_loss
                total_q2_loss += q2_loss
                total_v_loss += v_loss

                # self.optimize(
                #     actor_loss,
                #     # list(policies[firm_id].parameters()),
                #     self.actor_optimizer[firm_id],
                # )
                self.optimize(alpha_loss, self.alpha_optimizers[firm_id])
                # total_q1_loss += q1_loss
                # total_q2_loss += q2_loss
                # total_v_loss += v_loss

                with torch.no_grad():
                    history.append(
                        {
                            "actor_loss": actor_loss.item(),
                            "critic_loss": (v_loss + q1_loss + q2_loss).item(),
                            "entropy_loss": -log_pi.mean().item(),
                            "firm_id": firm_id,
                        },
                    )
            # Optimize for Actors
            self.optimize(total_actor_loss, self.actor_optimizer)
            # Optimize for Decentralized Q-Critic (1)
            self.optimize(total_q1_loss, self.q_critic_optimizer)
            # Optimize for Decentralized Q-Critic (2)
            self.optimize(total_q2_loss, self.q_critic2_optimizer)
            # Optimize for Decentralized V-Critic
            self.optimize(total_v_loss, self.v_critic_optimizer)

            self._soft_update_target_network()

        # for actor_scheduler in self.actor_schedulers:
        # self.actor_scheduler.step()
        # self.q_critic_scheduler.step()
        # self.q_critic2_scheduler.step()
        # self.v_critic_scheduler.step()
        df_out = pd.DataFrame(history).groupby("firm_id").mean()
        df_out["reward"] = rewards_lst.mean(dim=(0, 2)).flatten().cpu().numpy()
        return df_out

    @torch.no_grad()
    def _soft_update_target_network(self):
        for target_param, param in zip(
            self.v_critic_target.parameters(),
            self.v_critic.parameters(),
            strict=True,
        ):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
