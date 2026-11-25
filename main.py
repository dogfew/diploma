from src.models.policy.beta_policy import LogisticNormalPolicyNetwork
import argparse
import os
from pathlib import Path

from src.environment.prod_functions import MoneyPrintingLeontief

os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("OMP_PLACES", "cores")
import torch
import yaml

from src.environment import BatchedEnvironment, BatchedLeontief
from src.models.policy import BetaPolicyNetwork
from src.trainer import TrainerPPO
from src.trainer.trainer_masac import TrainerMASAC
from src.utils.plotting_mean import plot_actions_batch, plot_environment_batch, plot_volumes_batch


def load_functions(config_list, device):
    if config_list is None:
        return None
    funcs = []
    for item in config_list:
        if item["type"] == "Leontief":
            inputs = torch.tensor(item["inputs"], device=device)
            outputs = torch.tensor(item["outputs"], device=device)
            funcs.append(BatchedLeontief(inputs, outputs, device=device))
        elif item["type"] == "MoneyPrintingLeontief":
            inputs = torch.tensor(item["inputs"], device=device)
            outputs = torch.tensor(item["outputs"], device=device)
            funcs.append(MoneyPrintingLeontief(inputs, outputs, money_delta=item["delta"], device=device))
    return funcs


def main(config_path, *, no_inference=False, validate_args=False):
    torch.distributions.Distribution.set_default_validate_args(validate_args)
    path = Path("configs/") / config_path
    with path.open() as f:
        config = yaml.load(f, Loader=yaml.CSafeLoader)
        print(config)

    device = config["device"]
    if device == "cpu":
        torch.set_flush_denormal(True)
    torch.manual_seed(config["seed"])

    print(f"Starting experiment: {config['experiment_name']} on {device}")

    prod_funcs = load_functions(config["prod_functions"], device)
    invest_funcs = load_functions(config.get("invest_functions"), device)
    policy_map = {
        "BetaPolicyNetwork": BetaPolicyNetwork,
        "LogisticNormalPolicyNetwork": LogisticNormalPolicyNetwork,
    }

    env_cfg = config["environment"]
    env = BatchedEnvironment(
        market_kwargs=config["market"],
        policy_class=policy_map[env_cfg.pop("policy_class")],
        prod_functions=prod_funcs,
        invest_functions=invest_funcs,
        device=device,
        **env_cfg,
    )

    trainer_cfg = config["trainer"]

    trainer_map = {"TrainerPPO": TrainerPPO, "TrainerMASAC": TrainerMASAC}
    trainer_class = trainer_map[trainer_cfg.pop("trainer_class")]
    trainer = trainer_class(
        env,
        learning_rates=tuple(trainer_cfg.pop("learning_rates")),
        device=device,
        **trainer_cfg,
    )

    # 5. Обучение
    train_cfg = config["training_params"]
    trainer.train(
        n_epochs=train_cfg["n_epochs"],
        episode_length=train_cfg["episode_length"],
        debug_period=train_cfg["debug_period"],
        shuffle_order=train_cfg.get("shuffle_order", False),
    )

    eval_cfg = config["evaluation"]
    env.reset()
    env.change_batch_size(eval_cfg["eval_batch_size"])

    n_periods = eval_cfg["n_periods"]
    for i in range(n_periods):
        env.step_and_record_batch(i % env.n_agents)
    if no_inference or n_periods == 0:
        return
    print("Plotting results...")
    plot_environment_batch(env.state_history)
    plot_volumes_batch(env.state_history)
    for k, v in env.actions_history.items():
        plot_actions_batch(v, f"policy{k}", mode="median")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiment from YAML config")
    parser.add_argument("--config", default="test.yaml", type=str, help="Path to the YAML configuration file")
    parser.add_argument(
        "--no_inference",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--validate_args",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Проверять аргументы и значения у torch.distributions (медленнее, для отладки)",
    )
    args = parser.parse_args()

    main(args.config, no_inference=args.no_inference, validate_args=args.validate_args)
