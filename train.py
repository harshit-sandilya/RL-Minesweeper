"""
train.py  —  Entry point for all training runs.

Loads a YAML config, builds the Minesweeper env, constructs the agent,
and calls agent.train().  CLI flags override YAML values so you can
run sweeps without editing any file.

Usage
-----
  # Basic run using a config file
  python train.py --config rl/configs/config_dqn.yml

  # Override individual fields from the CLI
  python train.py --config rl/configs/config_dqn.yml --device mps --seed 7

  # Board-size sweep (mine density held constant at ~0.15)
  python train.py --config rl/configs/config_dqn.yml --n 6  --mines 5  --run_name dqn_6x6
  python train.py --config rl/configs/config_dqn.yml --n 8  --mines 10 --run_name dqn_8x8
  python train.py --config rl/configs/config_dqn.yml --n 10 --mines 15 --run_name dqn_10x10

Adding a new algorithm
----------------------
  1. Implement rl/algorithms/<algo>.py  and  rl/agents/<algo>_agent.py
  2. Add an entry to AGENT_REGISTRY below (algo_name → lazy import fn).
  3. Add a  rl/configs/config_<algo>.yml  with its hyperparams.
  4. Run:  python train.py --config rl/configs/config_<algo>.yml --algo <algo>

  Nothing else in train.py needs to change.
"""

import argparse
import sys

# ── Algorithm registry ────────────────────────────────────────────────────────
# Maps algo name → callable that returns the AgentClass (lazy import).
# Each agent owns its own network construction — train.py never builds a network.
# To add a new algorithm: add one entry here and implement the agent class.


def _get_agent_class(algo: str):
    """Lazy import so unused algorithm modules are never loaded."""
    if algo == "dqn":
        from rl.agents.dqn_agent import DQNAgent

        return DQNAgent
    raise ValueError(f"Unknown algorithm: '{algo}'. Available: {list(AGENT_REGISTRY)}")


# Registry: algo_name → (description string) — used only for argparse choices/help.
# The actual class is fetched lazily via _get_agent_class() to avoid circular imports.
AGENT_REGISTRY = {
    "dqn": "Deep Q-Network (Mnih et al. 2015)",
    # "ddqn": "Double DQN (van Hasselt et al. 2016)",   ← future
    # "ppo":  "Proximal Policy Optimisation",            ← future
}


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an RL agent on Minesweeper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML config file  (e.g. rl/configs/config_dqn.yml)",
    )
    p.add_argument(
        "--algo",
        default="dqn",
        choices=list(AGENT_REGISTRY),
        help="Algorithm to run.",
    )

    # ── Optional checkpoint to resume from ───────────────────────────────
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a .pt checkpoint to resume training from.",
    )

    # ── Fields that are useful to override from CLI without editing YAML ──
    p.add_argument("--run_name", default=None, help="Override Config.run_name")
    p.add_argument(
        "--device", default=None, help="Override Config.device  (cpu | cuda | mps)"
    )
    p.add_argument("--seed", default=None, type=int, help="Override Config.seed")
    p.add_argument("--n", default=None, type=int, help="Override board size n×n")
    p.add_argument("--mines", default=None, type=int, help="Override mine count")
    p.add_argument(
        "--episodes",
        default=None,
        type=int,
        dest="total_episodes",
        help="Override Config.total_episodes",
    )

    return p.parse_args()


# ── Config loader + CLI override ─────────────────────────────────────────────

# Fields that can be overridden from the CLI.
# All of these exist on MinesweeperConfig (n, mines are subclass fields;
# the rest are on the base Config).
_OVERRIDABLE_FIELDS = ("run_name", "device", "seed", "n", "mines", "total_episodes")


def _load_config(args: argparse.Namespace):
    """
    Load MinesweeperConfig from YAML then apply any CLI overrides.

    Priority:  CLI flags  >  YAML values  >  MinesweeperConfig dataclass defaults.
    """
    from rl.common.config import MinesweeperConfig

    cfg = MinesweeperConfig.from_yaml(args.config)

    for field in _OVERRIDABLE_FIELDS:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
            print(f"[train.py]  override  {field} = {val}")

    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)

    print(f"[train.py]  algo     = {args.algo}")
    print(f"[train.py]  config   = {args.config}")
    print(f"[train.py]  board    = {cfg.n}×{cfg.n}  mines={cfg.mines}")
    print(f"[train.py]  device   = {cfg.device}")
    print(f"[train.py]  run      = {cfg.run_name}")
    print()

    # Import the MinesweeperEnv client from the minesweeper_env package
    from minesweeper_env import MinesweeperEnv

    # Initialize the environment with the correct parameters
    # The MinesweeperEnv client takes base_url, not individual parameters
    env_url = getattr(cfg, "env_url", "http://localhost:8000")
    env = MinesweeperEnv(base_url=env_url)

    # FIX #2 + #1: Get AgentClass, pass (cfg, env) — not (cfg, network).
    # Drop the `with` context manager — DQNAgent has no __enter__/__exit__.
    AgentClass = _get_agent_class(args.algo)
    agent = AgentClass(cfg, env)

    # Optional: resume from a checkpoint before training starts
    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)

    # FIX #3: train() returns None; don't try to unpack it as a dict.
    # logger.dump_summary() is already called inside agent.train() after
    # all episodes finish — the CSV is written automatically.
    try:
        agent.train()
    except KeyboardInterrupt:
        print(
            "\n[train.py]  Interrupted by user. "
            f"Resume from  {cfg.checkpoint_dir}/{cfg.run_name}_latest.pt"
        )
        sys.exit(0)

    print("[train.py]  Training finished.")
    print(f"[train.py]  TensorBoard logs : {cfg.log_dir}/{cfg.run_name}/")
    print(f"[train.py]  Summary CSV      : {cfg.log_dir}/{cfg.run_name}/summary.csv")
    print(f"[train.py]  Checkpoints      : {cfg.checkpoint_dir}/")


if __name__ == "__main__":
    main()
