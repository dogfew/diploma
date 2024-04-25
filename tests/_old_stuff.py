@torch.no_grad()
def collect_experience(self, order, num_steps=50, do_not_skip=False):
    if order is None:
        order = range(len(self.environment.firms))
    batch_size = self.environment.batch_size
    action_dim = self.environment.action_dim
    state_dim = self.environment.state_dim
    n_agents = self.environment.n_agents

    rewards_lst = []
    kwargs = dict(fill_value=torch.nan, device=self.device)

    new_data = {
        "states": torch.full((batch_size, n_agents, state_dim), **kwargs),
        "actions": torch.full((batch_size, n_agents, action_dim), **kwargs),
        "rewards": torch.full((batch_size, n_agents, 1), **kwargs),
        "next_states": torch.full((batch_size, n_agents, state_dim), **kwargs),
    }
    # First Steps
    for firm_id in order:
        state, actions, log_probs, _, costs = self.environment.step(firm_id)
        new_data["states"][:, firm_id, :] = state
        new_data["actions"][:, firm_id, :] = actions
        new_data["rewards"][:, firm_id, :] = -costs
    # Other Steps. We do not record final step
    for step in range(num_steps):
        new_data, old_data = {
            "rewards": torch.full((batch_size, n_agents, 1), **kwargs),
            "next_states": torch.full((batch_size, n_agents, state_dim), **kwargs),
            "actions": torch.full((batch_size, n_agents, action_dim), **kwargs),
            "states": torch.full((batch_size, n_agents, state_dim), **kwargs),
        }, new_data
        for firm_id in order:
            state, actions, log_probs, prev_revenue, costs = self.environment.step(
                firm_id
            )

            old_data["rewards"][:, firm_id, :] += prev_revenue
            old_data["next_states"][:, firm_id, :] = state

            new_data["rewards"][:, firm_id, :] = -costs
            new_data["states"][:, firm_id, :] = state
            new_data["actions"][:, firm_id, :] = actions

        self.buffer.add_batch(
            x=old_data["states"],
            x_next=old_data["next_states"],
            actions=old_data["actions"],
            rewards=old_data["rewards"] / self.environment.market.start_gains,
        )
        rewards_lst.append(old_data["rewards"].permute(1, 0, 2))
    return rewards_lst

    @torch.no_grad()
    def collect_experience(self, order, num_steps=50, do_not_skip=False):
        if order is None:
            order = range(len(self.environment.firms))
        rewards_lst = []
        to_add = {
            "actions": [],
            "states": [],
            "log_probs": [],
            "revenues": [],
            "costs": [],
            "next_states": [],
        }
        for firm_id in order:
            state, actions, log_probs, _, costs = self.environment.step(firm_id)
            to_add["states"].append(state)
            to_add["actions"].append(actions)
            to_add["log_probs"].append(log_probs)
            to_add["costs"].append(costs)
        tensor_lst = deque([to_add], maxlen=2)
        for step in range(num_steps):
            tensor_lst.append(
                {
                    "actions": [],
                    "states": [],
                    "log_probs": [],
                    "revenues": [],
                    "costs": [],
                    "next_states": [],
                }
            )
            for firm_id in order:
                state, actions, log_probs, prev_revenue, costs = self.environment.step(
                    firm_id
                )

                tensor_lst[-2]["revenues"].append(prev_revenue)
                tensor_lst[-2]["next_states"].append(state)
                tensor_lst[-1]["states"].append(state)
                tensor_lst[-1]["actions"].append(actions)
                tensor_lst[-1]["log_probs"].append(log_probs)
                tensor_lst[-1]["costs"].append(costs)

            to_buffer = tensor_lst[-2]
            rewards = torch.stack(to_buffer["revenues"]) - torch.stack(
                to_buffer["costs"]
            )
            rewards = rewards / self.environment.market.start_gains
            self.buffer.add_batch(
                x=torch.stack(to_buffer["states"], dim=1),
                x_next=torch.stack(to_buffer["next_states"], dim=1),
                actions=torch.stack(to_buffer["actions"], dim=1),
                rewards=rewards.permute(1, 0, 2),
            )
            rewards_lst.append(rewards)
        return rewards_lst
