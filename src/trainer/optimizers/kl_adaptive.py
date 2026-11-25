class KLAdaptiveLR:
    def __init__(
        self,
        optimizer,
        target_kl=0.01,
        factor=1.5,
        min_lr=1e-5,
        max_lr=1e-3,
    ):
        self.optimizer = optimizer
        self.target_kl = target_kl
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_kl = target_kl * factor

    def step(self, kls):
        for group, kl in zip(self.optimizer.param_groups, kls, strict=True):
            if kl > 2 * self.target_kl:
                group["lr"] = max(group["lr"] / self.factor, self.min_lr)
            elif 0 < kl < self.target_kl / 2:
                group["lr"] = min(group["lr"] * self.factor, self.max_lr)

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
