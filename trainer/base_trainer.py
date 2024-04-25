import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from IPython.display import clear_output
from copy import deepcopy, copy

from trainer.replay_buffer import ReplayBuffer, ReplayBufferOld


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


class BaseTrainer:
    def __init__(self, environment, device="cuda"):
        self.environment = environment
        self.n_agents = len(self.environment.firms)
        self.buffer = ReplayBufferOld(50_000)
        self.policies = self.environment.policies
        self.target_policies = self.environment.target_policies
        self.in_notebook = in_notebook()

        colors = plt.cm.get_cmap("Set1").colors
        self.color_map = {
            firm_id: colors[firm_id % len(colors)]
            for firm_id in range(len(environment.firms))
        }
        self.device = device

    @torch.no_grad()
    def plot_loss(self, df_list) -> None:
        clear_output(True)
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
        df = pd.concat(df_list).reset_index()
        for firm_id, group in df.groupby("firm_id"):
            color = self.color_map[firm_id]

            ax[0].plot(
                group["episode"],
                group["actor_loss"],
                label=f"Firm {firm_id}",
                color=color,
            )
            ax[1].plot(
                group["episode"],
                group["critic_loss"],
                label=f"Firm {firm_id}",
                color=color,
            )
            ax[2].plot(
                group["episode"],
                group["reward"],
                label=f"Firm {firm_id}",
                alpha=0.5,
                color=color,
                linestyle="--",
            )

            window_size = self.window_size if hasattr(self, "window_size") else 1
            ax[2].plot(
                group["episode"],
                group["reward"].rolling(window=window_size).mean(),
                color=color,
                linewidth=3,
            )

        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Actor Loss")
        ax[0].set_title("Actor Loss Dynamics")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].set_xlabel("Epoch")
        ax[1].set_title("Critic Loss Dynamics")
        ax[1].legend()
        # ax[1].set_yscale("log")
        ax[1].grid(True, axis="y")

        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Mean Reward")
        ax[2].set_title("Mean Reward Dynamics")
        ax[2].legend()
        ax[2].grid(True)
        # ax[2].set_yscale("symlog")
        ax[2].axhline(y=0, color="grey", linestyle="--", label="Zero Profit")
        plt.grid()
        plt.show()

    @torch.no_grad()
    def collect_experience(self, order, do_not_skip=False):
        if order is None:
            order = range(len(self.environment.firms))
        to_add = {
            "actions": [],
            "states": [],
            "log_probs": [],
            "rewards": [],
            "next_states": [],
        }
        for firm_id in order:
            state, actions, log_probs, _, costs = self.environment.step(firm_id)
            to_add["actions"].append(actions)
            to_add["states"].append(state)
            to_add["log_probs"].append(log_probs)
        if do_not_skip:
            old_firms_data = [
                self.environment.firms[firm_id].copy() for firm_id in order
            ]
            old_market_data = self.environment.market.copy()
        for firm_id in order:
            next_state, _, _, revenue, _ = self.environment.step(firm_id)
            to_add["next_states"].append(next_state)
            if isinstance(revenue, int):
                to_add["rewards"].append(
                    torch.tensor([revenue - costs], device=self.device)
                )
            else:
                to_add["rewards"].append(revenue - costs)
        if do_not_skip:
            for firm_id, old_data in zip(order, old_firms_data):
                self.environment.firms[firm_id].set(*old_data)
            self.environment.market.set(*old_market_data)
        rewards = torch.stack(to_add["rewards"])
        # normalize rewards
        rewards = rewards / self.environment.market.start_gains
        if isinstance(revenue, int):
            self.buffer.add(
                x=torch.stack(to_add["states"]),
                x_next=torch.stack(to_add["next_states"]),
                actions=torch.stack(to_add["actions"]),
                rewards=rewards,
            )
        else:
            self.buffer.add_batch(
                x=torch.stack(to_add["states"], dim=1),
                x_next=torch.stack(to_add["next_states"], dim=1),
                actions=torch.stack(to_add["actions"], dim=1),
                rewards=rewards.permute(1, 0, 2),
            )
        return rewards

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

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename, "rb"))
