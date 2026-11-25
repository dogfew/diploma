import pickle
import random
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import clear_output
from matplotlib.patheffects import withStroke
from torch import nn
from tqdm import tqdm

from src.environment.environment import BatchedEnvironment

if TYPE_CHECKING:
    from trainer.replay_buffer import ReplayBuffer, ReplayBufferPPO


# def in_notebook():
#     try:
#         from IPython import get_ipython

#         if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
#             return False
#     except ImportError:
#         return False
#     except AttributeError:
#         return False
#     return True


class BaseTrainer[B: ReplayBufferPPO | ReplayBuffer]:
    def __init__(
        self,
        environment: BatchedEnvironment,
        gamma=0.99,
        critic_loss=torch.nn.functional.smooth_l1_loss,
        entropy_gamma=0.999,
        max_grad_norm=0.5,
        entropy_reg=0.01,
        batch_size: int = 1024,
        tau=0.95,
        device="cuda",
    ):
        self.environment = environment
        self.batch_size = batch_size
        self.n_agents = self.environment.n_agents
        self.buffer: B | None = None
        self.policies = self.environment.policies
        self.target_policies = self.environment.target_policies
        # self.in_notebook = in_notebook()

        # colors = plt.cm.get_cmap("Set1").colors
        cmap = mpl.colormaps["Set1"]
        colors = cmap(range(cmap.N))
        self.color_map = {firm_id: colors[firm_id % len(colors)] for firm_id in range(len(environment.firms))}
        self.device = device
        self.critic_scheduler = None
        self.critic_loss = critic_loss

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
        # self.batch_size = self.environment.batch_size

    @property
    def last_lr(self):
        if self.critic_scheduler is not None:
            return self.critic_scheduler.get_last_lr()[0]
        if self.critic_scheduler is not None:
            return self.q_critic_scheduler.get_last_lr()[0]
        return None

    def train_epoch(self, max_episode_length, order):
        raise NotImplementedError

    def train(self, n_epochs, episode_length=100, debug_period=5, *, shuffle_order=False):
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
            if self.episode % debug_period == 0 and debug_period < n_epochs:
                self.plot_loss(self.df_list)
            self.episode += 1
            if shuffle_order:
                random.shuffle(order)
            pbar.set_postfix(
                {
                    "LR": self.last_lr,
                    "Order": str(order),
                },
            )
        if debug_period < n_epochs:
            self.plot_loss(self.df_list)

    @torch.no_grad()
    def plot_loss(self, df_list) -> None:
        clear_output(wait=True)
        # _fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
        _fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))
        df = pd.concat(df_list).reset_index()
        for firm_id, group in df.groupby("firm_id"):
            color = self.color_map[firm_id]

            ax[0, 0].plot(
                group["episode"],
                group["actor_loss"],
                label=f"Firm {firm_id}",
                path_effects=[withStroke(linewidth=2, foreground="black")],
                color=color,
            )
            ax[0, 1].plot(
                group["episode"],
                group["critic_loss"],
                label=f"Firm {firm_id}",
                path_effects=[withStroke(linewidth=2, foreground="black")],
                color=color,
            )
            ax[1, 0].plot(
                group["episode"],
                group["reward"],
                label=f"Firm {firm_id}",
                path_effects=[withStroke(linewidth=2, foreground="black")],
                color=color,
            )
            ax[1, 1].plot(
                group["episode"],
                group["entropy_loss"],
                label=f"Firm {firm_id}",
                path_effects=[withStroke(linewidth=2, foreground="black")],
                color=color,
            )
            if "approx_kl" in group:
                ax[2, 0].plot(
                    group["episode"],
                    group["approx_kl"],
                    label=f"Firm {firm_id}",
                    path_effects=[withStroke(linewidth=2, foreground="black")],
                    color=color,
                )
                ax[2, 1].plot(
                    group["episode"],
                    group["lr"],
                    label=f"Firm {firm_id}",
                    path_effects=[withStroke(linewidth=2, foreground="black")],
                    color=color,
                )

        ax[0, 0].set_xlabel("Epoch")
        ax[0, 0].set_ylabel("Actor Loss")
        ax[0, 0].set_title("Actor Loss Dynamics")
        ax[0, 0].legend()
        ax[0, 0].grid(True)

        ax[0, 1].set_xlabel("Epoch")
        ax[0, 1].set_title("Critic Loss Dynamics")
        ax[0, 1].set_yscale("log", base=2)
        ax[0, 1].legend()
        ax[0, 1].grid(True)

        ax[1, 0].set_xlabel("Epoch")
        ax[1, 0].set_ylabel("Mean Reward")
        ax[1, 0].set_title("Mean Reward Dynamics")
        ax[1, 0].legend()
        ax[1, 0].grid(True)
        ax[1, 0].axhline(y=0, color="grey", linestyle="--", label="Zero Profit")

        ax[1, 1].set_xlabel("Epoch")
        ax[1, 1].set_ylabel("Entropy Loss")
        ax[1, 1].set_title("Entropy Loss Dynamics")
        ax[1, 1].legend()
        ax[1, 1].set_yscale("log", base=2)
        ax[1, 1].grid(True)

        ax[2, 0].set_xlabel("Epoch")
        ax[2, 0].set_ylabel("Approx KL")
        ax[2, 0].set_title("Policy KL Dynamics")
        ax[2, 0].legend()
        ax[2, 0].set_yscale("log", base=2)
        ax[2, 0].grid(True)

        ax[2, 1].set_xlabel("Epoch")
        ax[2, 1].set_ylabel("Actor LR")
        ax[2, 1].set_title("Actor LR Dynamics")
        ax[2, 1].legend()
        ax[2, 1].set_yscale("log", base=10)
        ax[2, 1].grid(True)

        # ax[2, 1].axis("off")  # шестая ячейка пустая
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def collect_experience(self, order, num_steps=50):
        if order is None:
            order = range(len(self.environment.firms))

        batch_size: int = self.environment.batch_size
        action_dim: int = self.environment.action_dim
        state_dim: int = self.environment.state_dim
        n_agents: int = self.environment.n_agents

        total_steps: int = num_steps + 1

        all_states = torch.empty((total_steps, batch_size, n_agents, state_dim), device=self.device)
        all_actions = torch.empty((total_steps, batch_size, n_agents, action_dim), device=self.device)
        all_rewards = torch.empty((total_steps, batch_size, n_agents, 1), device=self.device)

        # First Step
        for firm_id in order:
            state, actions, _log_probs, revenue, costs = self.environment.step(firm_id)
            all_states[0, :, firm_id, :] = state
            all_actions[0, :, firm_id, :] = actions
            all_rewards[0, :, firm_id, :] = revenue - costs

        # Other steps
        for step in range(1, total_steps):
            for firm_id in order:
                state, actions, _log_probs, revenue, costs = self.environment.step(firm_id)
                all_states[step, :, firm_id, :] = state
                all_actions[step, :, firm_id, :] = actions
                all_rewards[step, :, firm_id, :] = -costs

                all_rewards[step - 1, :, firm_id, :] += revenue
        self.buffer.add_batch(
            x=all_states[:-1].flatten(0, 1),
            x_next=all_states[1:].flatten(0, 1),
            actions=all_actions[:-1].flatten(0, 1),
            rewards=all_rewards[:-1].flatten(0, 1),
        )
        return all_rewards.permute(0, 2, 1, 3)

    def get_actions(self, x, policies, firm_id=None):
        all_actions = []
        all_log_probs = []
        for j in range(self.n_agents):
            if j == firm_id:
                action_j, log_prob_j = policies[j](x[:, j, :])
            else:
                with torch.no_grad():
                    action_j, log_prob_j = policies[j](x[:, j, :])
            all_actions.append(torch.concatenate(action_j, dim=-1))
            all_log_probs.append(torch.concatenate(log_prob_j, dim=-1))
        actions = torch.stack(all_actions).transpose(1, 0)
        log_probs = torch.stack(all_log_probs).transpose(1, 0)
        return actions, log_probs

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    @torch.no_grad()
    def _clip_grad_norm(self, parameters, norm_type=2):
        try:
            for param in parameters:
                if torch.isnan(param.grad).any():
                    param.grad = torch.where(
                        torch.isnan(param.grad),
                        torch.zeros_like(param.grad),
                        param.grad,
                    )
                    print(param.shape)
                    print(param.min().item(), param.max().item())
            nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=True,
            )
        except RuntimeError:
            return False
        return True

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        if all(self._clip_grad_norm(group["params"]) for group in optimizer.param_groups):
            optimizer.step()
        else:
            print("Got NaN gradients")

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename, "rb"))
