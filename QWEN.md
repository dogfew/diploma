# Diploma Project — QWEN Context

## Project Overview

This is a **multi-agent reinforcement learning** simulation of a multi-branch economy. Firms (agents) operate in market(s), buying raw materials, producing goods, investing in fixed capital, and setting prices. The core algorithm is **Multi-Agent Proximal Policy Optimization (MAPPO)** trained with Generalized Advantage Estimation (GAE).

The project is a research/experimentation codebase for a diploma thesis on agent-based economic modeling. It contains multiple experimental scripts (`main.py`, `main2.py`, `main3.py`, `main4.py`, `main5.py`) and Jupyter notebooks (`experiments*.ipynb`) that test different reward functions, firm types, and market configurations.

### Key Concepts

- **Firms**: Agents that buy inputs, produce outputs, invest in capital, and sell products.
- **Market**: A mechanism connecting firms through price and volume matrices. Two market types: `BatchedMarket` (full transparency) and `BlindMarket` (quantity-only visibility).
- **Production Functions**: Leontief (fixed-ratio) and Cobb-Douglas functions.
- **Policies**: Neural networks (BetaPolicyNetwork) that output action distributions via Beta/Dirichlet distributions.
- **Reward types**: Production-based, finance-based, or mixed rewards.

### Architecture

```
src/
├── environment/          # Economic simulation
│   ├── environment.py    # BatchedEnvironment, BlindMarket
│   ├── market.py         # BatchedMarket, BlindMarket
│   ├── firm.py           # BatchedFirm, BatchedLimitFirm, etc.
│   ├── prod_functions.py # BatchedLeontief, CobbDouglas
│   └── utils.py          # Helpers (dimension calculation, action processing)
├── models/               # Neural network components
│   ├── policy/           # BetaPolicyNetwork (main policy class)
│   ├── critic/           # CentralizedCriticV, CentralizedQ_Critic
│   └── loss/             # Loss function implementations
├── trainer/              # RL training algorithms
│   ├── base_trainer.py   # BaseTrainer (plotting, common utilities)
│   ├── trainer_ppo.py    # TrainerPPO (MAPPO with GAE)
│   ├── trainer_sac.py    # TrainerSAC
│   ├── trainer_masac.py  # Multi-Agent SAC
│   └── replay_buffer.py  # ReplayBufferPPO
└── utils/                # Visualization and helpers
    ├── plotting_mean.py  # Plotting utilities
    └── plotting_tikz.py  # TikZ output for LaTeX
```

## File Structure

| Path | Description |
|-------|------------|
| `main.py`, `main2.py`, ... | Standalone experiment scripts (one firm, different configs) |
| `experiments.py`, `experiments*.ipynb` | Jupyter notebooks with extensive experiment history |
| `src/environment/` | Core economic simulation (market, firms, production) |
| `src/models/` | Neural network policy and critic implementations |
| `src/trainer/` | RL trainers (PPO, SAC, MASAC) |
| `src/utils/` | Plotting and utility functions |
| `requirements.txt` | Python dependencies (PyTorch, torchrl, numpy, matplotlib, etc.) |

## Running Experiments

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running a Script

Each `main*.py` file is a self-contained experiment. Example:
```bash
python main5.py
```

Key parameters are defined at the top of each script:
- `market_kwargs`: Initial volumes, prices, gains, deprecation settings
- `prod_functions`: Production functions for each firm
- `invest_functions`: Investment functions (optional, for capital-limited firms)
- `target`: Reward type — `"production"` or `"finance"`
- `device`: Computation device — `"cpu"`, `"cuda"`, or `"xpu"`
- `batch_size`: Number of parallel simulations (default 512 or 1024)

### Running Notebooks

```bash
jupyter notebook experiments.py
# or
jupyter notebook experiments_latex.ipynb
jupyter notebook experiments_old.ipynb
```

## Experiment Configuration Guide

### Market Parameters (`market_kwargs`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_volumes` | 10 | Initial goods reserves per firm |
| `base_price` | 50 | Initial market prices |
| `start_gains` | 500 | Initial financial resources per firm |
| `deprecation_steps` | 2 | Capital depreciation period |
| `min_price` | 1 | Minimum allowed price |
| `max_price` | 100 | Maximum allowed price |

### Reward Types (`target`)

- **`"production"`**: Reward = production volume. Firms cooperate, production grows.
- **`"finance"`**: Reward = profit (revenue - costs). Leads to Nash equilibrium where firms stop trading.
- **Mixed**: `"finance"` with `production_reg > 0` adds production bonus.

### Firm Types

| Class | Description |
|-------|-------------|
| `BatchedFirm` | Basic firm, no fixed capital |
| `BatchedProductionFirm` | Production-focused reward |
| `BatchedLimitFirm` | With fixed capital/investment |
| `BatchedLimitProductionFirm` | Capital + production reward |

### Policy Network

- **BetaPolicyNetwork**: 2-layer MLP (hidden_dim=64) with Tanh activations.
- Outputs 4 action groups: buy percentages, sale percentages, use (production/investment) percentages, prices.
- Each action group uses appropriate distribution: Dirichlet for buy, Beta for sale/prices, Dirichlet for use.
- Weights initialized per Stable Baselines paper (https://arxiv.org/pdf/2006.05990).

## Common Development Patterns

1. **Define environment**: Create `BatchedEnvironment` with market kwargs, policy class, production functions.
2. **Define trainer**: Create `TrainerPPO` with learning rates, batch size, entropy regulation.
3. **Train**: `trainer.train(n_epochs, episode_length, shuffle_order=False)`.
4. **Evaluate**: Reset environment, run simulation steps, plot results with `plot_*` utilities.

## Key Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `learning_rates` | (3e-3, 3e-4) | (actor_lr, critic_lr) |
| `batch_size` | 512 or 1024 | Parallel simulations |
| `entropy_reg` | 1e-5 | Entropy regularization |
| `entropy_gamma` | 0.999 | Entropy decay |
| `lr_gamma` | 0.999 | Learning rate decay |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `episode_length` | 32 | Steps per episode |
| `production_reg` | 0-15 | Production bonus weight |

## Git Branches / History

Current branch: `eval-agents`. The project has gone through multiple refactoring iterations (see git history: "Большой рефакторинг", "Blind Market Experiments", etc.).

## Testing

You are writing tests now.

## Notes

- Seed values significantly affect training trajectories. Different seeds can produce very different outcomes.
- Fixed shuffle_order vs shuffled order produces different market dynamics.
- The project uses PyTorch with batched (vectorized) simulations for efficiency.
- Plotting utilities support both matplotlib (interactive) and TikZ (LaTeX publication) outputs.
