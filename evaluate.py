"""
evaluate.py  —  Run a trained checkpoint in greedy evaluation mode.

Loads a saved .pt checkpoint, runs N greedy episodes, prints a results
table, and saves per-episode stats to CSV.

Config is read from the checkpoint payload by default — pass --config
only if you want to override the training configuration (e.g. evaluate
on a different board size).

Env is constructed exactly as in train.py:
    MinesweeperEnv(base_url=cfg.env_url)
and each episode runs inside  `with env.sync() as env_sync:`  — the
same context manager contract the agent uses during training.

Usage
-----
  # Evaluate the best checkpoint from a run
  python evaluate.py --checkpoint checkpoints/dqn_8x8_best.pt

  # Override board size or episode count at eval time
  python evaluate.py --checkpoint checkpoints/dqn_8x8_best.pt --n 10 --mines 15

  # Use a specific config file instead of the one embedded in the checkpoint
  python evaluate.py --checkpoint checkpoints/dqn_8x8_best.pt --config rl/configs/config_dqn.yml

  # Find and evaluate the latest checkpoint automatically
  python evaluate.py --checkpoint_dir checkpoints
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from minesweeper_env import MinesweeperEnv
from rl.common.checkpoint import latest_checkpoint, load_checkpoint
from rl.common.config import MinesweeperConfig
from rl.common.utils import get_device
from rl.networks.cnn_network import MinesweeperCNN

# ── Config resolution ──────────────────────────────────────────────────────────


def _get_agent_class(algo: str):
    """Lazy import so unused algorithm modules are never loaded."""
    if algo == "dqn":
        from rl.agents.dqn_agent import DQNAgent

        return DQNAgent
    raise ValueError(f"Unknown algorithm: '{algo}'. Available: {list(AGENT_REGISTRY)}")


AGENT_REGISTRY = {
    "dqn": "Deep Q-Network (Mnih et al. 2015)",
    # "ddqn": "Double DQN (van Hasselt et al. 2016)",   ← future
    # "ppo":  "Proximal Policy Optimisation",            ← future
}


def _resolve_config(
    checkpoint_path: str,
    yaml_path: Optional[str],
    args: argparse.Namespace,
) -> MinesweeperConfig:
    """
    Build a MinesweeperConfig in priority order:
      1. YAML file (if --config was passed)
      2. Config dict embedded in the checkpoint
      3. MinesweeperConfig dataclass defaults

    CLI flags (--n, --mines, --device, --seed, --solve_tiles) always win.
    """
    if yaml_path:
        cfg = MinesweeperConfig.from_yaml(yaml_path)
    else:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = raw.get("config", {})
        valid = {
            k: v
            for k, v in cfg_dict.items()
            if k in MinesweeperConfig.__dataclass_fields__
        }
        cfg = MinesweeperConfig(**valid) if valid else MinesweeperConfig()

    for field in ("n", "mines", "device", "seed", "solve_tiles"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    return cfg


# ── Stats + reporting ──────────────────────────────────────────────────────────


def _compute_stats(results: List[Dict]) -> Dict:
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
    stats: Dict,
    cfg: MinesweeperConfig,
    checkpoint_path: str,
    ckpt_info: Dict,
    csv_path: str,
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
    print(f"  Results saved  : {csv_path}")
    print(sep)


def _save_csv(results: List[Dict], stats: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "steps", "win"])
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({"episode": i, **r})
    with open(path, "a") as f:
        f.write(
            f"# win_rate={stats['win_rate']:.4f}"
            f"  avg_reward={stats['avg_reward']:.4f}"
            f"  avg_steps={stats['avg_steps']:.1f}\n"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Minesweeper checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--checkpoint", metavar="PATH", help="Direct path to a .pt file.")
    ckpt.add_argument(
        "--checkpoint_dir", metavar="DIR", help="Scan dir — uses latest .pt."
    )

    p.add_argument(
        "--algo", default="dqn", help="Algorithm the checkpoint was trained with."
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML config override (default: use config embedded in checkpoint).",
    )
    p.add_argument(
        "--episodes",
        default=100,
        type=int,
        metavar="N",
        help="Number of greedy evaluation episodes.",
    )
    p.add_argument(
        "--save_csv",
        default=None,
        metavar="PATH",
        help="CSV output path (default: <checkpoint_dir>/<stem>_eval.csv).",
    )

    p.add_argument("--n", default=None, type=int)
    p.add_argument("--mines", default=None, type=int)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--solve_tiles", default=None, type=int)
    return p.parse_args()


# ── Network factory ────────────────────────────────────────────────────────────


def _build_network(algo: str, cfg: MinesweeperConfig) -> MinesweeperCNN:
    """
    Mirrors DQNAgent.__init__ network construction so state_dict loads cleanly.
    hidden_dim=256 is a code-level architecture constant, not a YAML field.
    """
    if algo == "dqn":
        return MinesweeperCNN(n=cfg.n, hidden_dim=256)
    raise ValueError(f"Unknown algo: {algo!r}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print(f"[evaluate.py]  No .pt files found in {args.checkpoint_dir!r}")
            sys.exit(1)
        print(f"[evaluate.py]  Using latest checkpoint: {ckpt_path}")

    # Config (checkpoint-embedded → YAML → CLI overrides)
    cfg = _resolve_config(ckpt_path, args.config, args)
    device = get_device(cfg.device)

    print(f"[evaluate.py]  algo     = {args.algo}")
    print(
        f"[evaluate.py]  board    = {cfg.n}x{cfg.n}  mines={cfg.mines}"
        f"  density={cfg.mines / cfg.n**2:.3f}"
    )
    print(f"[evaluate.py]  device   = {device}")
    print(f"[evaluate.py]  episodes = {args.episodes}")

    # Build network + restore weights (no optimizer needed in eval)
    network = _build_network(args.algo, cfg).to(device)
    network.eval()
    ckpt_info = load_checkpoint(ckpt_path, network)
    print(f"[evaluate.py]  {network!r}")

    # Construct env exactly as train.py does — params go to reset(), not constructor
    env = MinesweeperEnv(base_url=cfg.env_url)
    AgentClass = _get_agent_class(args.algo)
    agent = AgentClass(cfg, env)
    results = agent.evaluate(n_episodes=args.episodes)

    stats = _compute_stats(results)

    csv_path = args.save_csv
    if csv_path is None:
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        csv_path = os.path.join(os.path.dirname(ckpt_path), f"{stem}_eval.csv")

    _save_csv(results, stats, csv_path)
    _print_report(stats, cfg, ckpt_path, ckpt_info, csv_path)


if __name__ == "__main__":
    main()
