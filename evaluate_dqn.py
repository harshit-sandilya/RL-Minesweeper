import argparse

import numpy as np
import torch

from minesweeper_env import MinesweeperEnv
from rl.agents.dqn_agent import DQNAgent
from rl.common.checkpoint import load_checkpoint
from rl.common.config import MinesweeperConfig
from rl.networks.cnn_network import MinesweeperCNN


def _compute_stats(results: list[dict]) -> dict:
    rewards = np.array([r["reward"] for r in results])
    steps = np.array([r["steps"] for r in results])
    wins = np.array([r["win"] for r in results])
    return {
        "n_episodes": len(results),
        "win_rate": float(wins.mean()),
        "avg_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "avg_steps": float(steps.mean()),
    }


def _print_report(
    stats: dict,
    cfg: MinesweeperConfig,
    checkpoint_path: str,
    ckpt_info: dict,
) -> None:
    sep = "-" * 52
    meta = ckpt_info.get("metadata", {})
    print(sep)
    print("  Evaluation Results")
    print(sep)
    print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Saved at step  : {ckpt_info.get('step', '?'):,}")
    print(f"  Saved at ep    : {ckpt_info.get('episode', '?'):,}")
    if "win_rate" in meta:
        print(f"  Train win rate : {meta['win_rate']:.3f}  (checkpoint metadata)")
    print(
        f"  Board          : {cfg.n}x{cfg.n}  mines={cfg.mines}"
        f"  density={cfg.mines / cfg.n**2:.3f}"
    )
    print(sep)
    print(f"  Episodes       : {stats['n_episodes']:,}")
    print(f"  Win rate       : {stats['win_rate'] * 100:.1f}%")
    print(
        f"  Avg reward     : {stats['avg_reward']:.4f}  (std={stats['std_reward']:.4f})"
    )
    print(f"  Reward range   : [{stats['min_reward']:.4f},  {stats['max_reward']:.4f}]")
    print(f"  Avg steps/ep   : {stats['avg_steps']:.1f}")
    print(sep)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Minesweeper checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", metavar="PATH", help="Direct path to a .pt file.")
    p.add_argument(
        "--episodes",
        default=100,
        type=int,
        metavar="N",
        help="Number of greedy evaluation episodes.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = raw.get("config", {})
    valid = {
        k: v for k, v in cfg_dict.items() if k in MinesweeperConfig.__dataclass_fields__
    }
    cfg = MinesweeperConfig(**valid) if valid else MinesweeperConfig()

    network = MinesweeperCNN(n=cfg.n, hidden_dim=256).eval()
    ckpt_info = load_checkpoint(args.checkpoint, network)
    env = MinesweeperEnv(base_url=cfg.env_url)
    agent = DQNAgent(cfg, env)
    results = agent.evaluate(n_episodes=args.episodes)

    stats = _compute_stats(results)
    _print_report(stats, cfg, args.checkpoint, ckpt_info)


if __name__ == "__main__":
    main()
