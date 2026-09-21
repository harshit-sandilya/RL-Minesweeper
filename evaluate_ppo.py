import argparse

import torch

from minesweeper_env import MinesweeperEnv
from rl.agents.ppo_agent import PPOAgent
from rl.common.checkpoint import load_checkpoint
from rl.common.config import MinesweeperConfig
from rl.common.metrics import format_report, save_results_csv, summarize_results


def _print_checkpoint_header(
    cfg: MinesweeperConfig,
    checkpoint_path: str,
    ckpt_info: dict,
) -> None:
    sep = "-" * 52
    meta = ckpt_info.get("metadata", {})
    print(sep)
    print("  Checkpoint Info")
    print(sep)
    print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Saved at step  : {ckpt_info.get('step', '?'):,}")
    print(f"  Saved at ep    : {ckpt_info.get('episode', '?'):,}")
    if "win_rate" in meta:
        print(f"  Train win rate : {meta['win_rate']:.3f}  (checkpoint metadata)")
    if "eval_win_rate" in meta:
        print(
            f"  Eval win rate  : {meta['eval_win_rate']:.3f}  "
            f"(checkpoint metadata, at save time)"
        )
    print(
        f"  Board          : {cfg.n}x{cfg.n}  mines={cfg.mines}"
        f"  density={cfg.mines / cfg.n**2:.3f}"
    )
    print(sep)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Minesweeper PPO checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", metavar="PATH", required=True, help="Direct path to a .pt file."
    )
    p.add_argument(
        "--episodes",
        default=100,
        type=int,
        metavar="N",
        help="Number of greedy evaluation episodes.",
    )
    p.add_argument(
        "--algorithm",
        default="ppo",
        metavar="NAME",
        help="Label stored in results / used in report headers.",
    )
    p.add_argument(
        "--save-csv",
        default=None,
        metavar="PATH",
        help="If set, dump raw per-episode results to this CSV for later "
        "cross-algorithm / board-size analysis in pandas.",
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

    env = MinesweeperEnv(base_url=cfg.env_url)
    agent = PPOAgent(cfg, env)

    # Same contract as evaluate_a2c / evaluate_dqn: restore weights only
    # (no optimizer). Actor lives in model_state_dict; critic is stashed in
    # metadata["critic_state_dict"] (checkpoint.py has a single network slot).
    ckpt_info = load_checkpoint(args.checkpoint, agent.actor_net)
    critic_state = ckpt_info.get("metadata", {}).get("critic_state_dict")
    if critic_state is not None:
        agent.critic_net.load_state_dict(critic_state)

    _print_checkpoint_header(cfg, args.checkpoint, ckpt_info)

    results = agent.evaluate(n_episodes=args.episodes, algorithm=args.algorithm)

    summary = summarize_results(results, algorithm=args.algorithm)
    print(format_report(summary))

    if args.save_csv:
        save_results_csv(results, args.save_csv)
        print(f"  Raw per-episode results saved -> {args.save_csv}")


if __name__ == "__main__":
    main()
