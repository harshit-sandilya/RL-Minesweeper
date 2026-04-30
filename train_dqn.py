import argparse

from minesweeper_env import MinesweeperEnv
from rl.agents.dqn_agent import DQNAgent
from rl.common.config import MinesweeperConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an RL agent on Minesweeper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML config file  (e.g. ./configs/dqn.yml)",
    )

    return p.parse_args()


def _load_config(args: argparse.Namespace) -> MinesweeperConfig:
    cfg = MinesweeperConfig.from_yaml(args.config)
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)

    print("[train.py]  algo     = dqn")
    print(f"[train.py]  config   = {args.config}")
    print(f"[train.py]  board    = {cfg.n}×{cfg.n}  mines={cfg.mines}")
    print(f"[train.py]  device   = {cfg.device}")
    print(f"[train.py]  run      = {cfg.run_name}")
    print()

    env_url = getattr(cfg, "env_url", "http://localhost:8000")
    env = MinesweeperEnv(base_url=env_url)
    agent = DQNAgent(cfg, env)
    agent.train()

    print("[train.py]  Training finished.")
    print(f"[train.py]  TensorBoard logs : {cfg.log_dir}/{cfg.run_name}/")
    print(f"[train.py]  Summary CSV      : {cfg.log_dir}/{cfg.run_name}/summary.csv")
    print(f"[train.py]  Checkpoints      : {cfg.checkpoint_dir}/")


if __name__ == "__main__":
    main()
