"""
rl/common/config.py

Two-level configuration system.

┌─────────────────────────────────────────────────────────────────┐
│  Config            base class — algorithm + training loop only  │
│  MinesweeperConfig subclass  — adds env-specific fields         │
└─────────────────────────────────────────────────────────────────┘

Why two levels
--------------
Config holds everything an RL algorithm needs to run: discount
factor, batch size, epsilon schedule, replay buffer size.  None of
these fields mention a grid, a mine, or a server URL.

MinesweeperConfig adds what the *environment* needs.  An agent that
trains on CartPole never sees these fields; an agent that trains on
Minesweeper uses MinesweeperConfig and gets everything in one object.

Adding a new environment means subclassing Config and adding fields —
the base class, all algorithms, and all common utilities stay untouched.

from_yaml works at every level
-------------------------------
from_yaml is defined once on Config and inherited by every subclass.
It uses cls.__dataclass_fields__ which includes inherited fields, so:

    Config.from_yaml("config_dqn.yml")            → Config
    MinesweeperConfig.from_yaml("config_dqn.yml") → MinesweeperConfig

Unknown keys in the YAML are silently ignored — a future-env YAML
loaded as Config will simply skip the env-specific keys.

Usage
-----
    # In train.py (Minesweeper)
    cfg = MinesweeperConfig.from_yaml("configs/config_dqn.yml")

    # In a future CartPole train.py
    cfg = Config.from_yaml("configs/config_cartpole.yml")

    # CLI override (same pattern for both)
    cfg.device   = args.device   or cfg.device
    cfg.run_name = args.run_name or cfg.run_name
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Self

import yaml


# ── Base configuration ─────────────────────────────────────────────────────────
@dataclass
class Config:
    """
    Algorithm-level configuration.

    Every field here is meaningful regardless of what environment or
    network architecture is used.  Environment-specific fields live
    in subclasses (see MinesweeperConfig below).
    """

    # ── Training loop ──────────────────────────────────────────────────────
    total_episodes: int = 10_000
    eval_every: int = 500  # run evaluation every N training episodes
    eval_episodes: int = 20  # episodes per evaluation pass
    seed: int = 42
    device: str = "cpu"

    # ── Persistence ────────────────────────────────────────────────────────
    log_dir: str = "runs"  # TensorBoard SummaryWriter root
    run_name: str = "experiment"  # sub-folder inside log_dir
    checkpoint_dir: str = "checkpoints"  # where .pt files are saved

    # ── RL core (shared by every algorithm) ────────────────────────────────
    gamma: float = 0.99  # discount factor
    batch_size: int = 64
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1_000  # update steps before cosine decay starts
    lr_total_updates: int = 50_000  # T_max for cosine: total_episodes × avg_steps

    # ── Exploration — ε-greedy (shared across value-based algorithms) ──────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 100_000
    epsilon_schedule: str = "linear"  # "linear" | "exponential"

    # ── DQN-specific ───────────────────────────────────────────────────────
    replay_capacity: int = 100_000
    min_replay_size: int = 1_000  # steps before first gradient update
    target_update_freq: int = 1_000  # hard copy online → target every N steps

    # ── Class method ───────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """
        Load config from a YAML file.

        Only keys present in cls.__dataclass_fields__ are used.
        Unknown keys are silently ignored — safe to load a
        MinesweeperConfig YAML as a base Config, or vice versa.

        Works identically for Config and all subclasses:
            Config.from_yaml(path)            → Config instance
            MinesweeperConfig.from_yaml(path) → MinesweeperConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # cls.__dataclass_fields__ includes inherited fields on subclasses
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__}:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k:<25} = {v!r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Minesweeper configuration ──────────────────────────────────────────────────


@dataclass
class MinesweeperConfig(Config):
    """
    Extends Config with Minesweeper environment fields.

    All base Config fields are inherited and remain available.
    Only fields that are meaningless outside a Minesweeper context
    live here — grid dimensions, mine count, server URL, and the
    training-only solve_tiles schedule.

    Example YAML (configs/dqn.yml):
        n: 8
        mines: 10
        solve_tiles: 6
        curriculum_enabled: true
        curriculum_hold_steps: 20000
        curriculum_end_steps: 80000
        env_url: "http://localhost:9090"
        # ... plus any base Config fields to override ...
    """

    # ── Environment identity ───────────────────────────────────────────────
    env_url: str = "http://localhost:9090"  # OpenEnv server base URL

    # ── Board geometry ─────────────────────────────────────────────────────
    n: int = 8  # grid side length (4–16)
    mines: int = 10  # number of mines on the board

    # ── Curriculum (training-only) ────────────────────────────────────────
    solve_tiles: int = 0  # starting solve_tiles for training episodes
    curriculum_enabled: bool = False
    curriculum_hold_steps: int = 20_000  # keep solve_tiles fixed up to this step
    curriculum_end_steps: int = 80_000  # linearly decay solve_tiles to 0 by this step
