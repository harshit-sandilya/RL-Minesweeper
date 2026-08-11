"""
rl/common/checkpoint.py

Save and load model checkpoints for any agent in the framework.

A checkpoint bundles everything needed to resume training or to run
evaluate.py on a saved model:

    {
        "model_state_dict":     network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # None if not saved
        "step":                 int,    # global env step counter
        "episode":              int,    # episode number at save time
        "config":               dict,   # Config.to_dict() snapshot
        "metadata":             dict,   # e.g. {"win_rate": 0.73, "algo": "dqn"}
    }

Typical usage
─────────────
  # train.py — periodic save + best-model save
  from rl.common.checkpoint import save_checkpoint, checkpoint_path

  ckpt_dir  = cfg.checkpoint_dir
  save_checkpoint(                                           # periodic
      path      = checkpoint_path(ckpt_dir, cfg.run_name, episode=ep),
      network   = agent.online_net,
      optimizer = agent.optimizer,
      step      = global_step,
      episode   = ep,
      config    = cfg,
      metadata  = {"win_rate": logger.current_win_rate(), "algo": "dqn"},
  )
  if logger.current_win_rate() > best_win_rate:            # best model
      save_checkpoint(
          path    = checkpoint_path(ckpt_dir, cfg.run_name, tag="best"),
          ...
      )

  # evaluate.py — load and run greedy
  from rl.common.checkpoint import load_checkpoint, latest_checkpoint

  ckpt_path = args.checkpoint or latest_checkpoint(cfg.checkpoint_dir)
  info = load_checkpoint(ckpt_path, network=net)
  print(f"Loaded step={info['step']}, win_rate={info['metadata'].get('win_rate')}")
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ── Internal helper ───────────────────────────────────────────────────────────


def _infer_device(network: nn.Module) -> torch.device:
    """
    Return the device of the first parameter in `network`.
    Falls back to CPU if the network has no parameters (e.g. empty shell
    created for loading before weights are restored).
    """
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ── Save ──────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    episode: int,
    config: Any,  # Config instance
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist a training checkpoint to disk.

    The config is serialised via .to_dict() (or stored directly if it is
    already a dict) so evaluate.py can reconstruct the exact Config that
    was used without needing the original YAML file.

    Args:
        path:      Destination .pt file path.
                   Parent directories are created automatically.
        network:   The online Q-network (or any nn.Module).
        optimizer: The optimizer whose state should be saved.
        step:      Global environment step counter at save time.
        episode:   Episode number at save time.
        config:    Config instance (or dict) used for this run.
        metadata:  Optional free-form dict for extra info such as
                   win_rate, algo name, board dimensions, etc.
                   Stored verbatim and returned by load_checkpoint.

    Example:
        >>> save_checkpoint(
        ...     path     = "checkpoints/dqn_8x8_best.pt",
        ...     network  = agent.online_net,
        ...     optimizer= agent.optimizer,
        ...     step     = 250_000,
        ...     episode  = 5_000,
        ...     config   = cfg,
        ...     metadata = {"win_rate": 0.74, "algo": "dqn"},
        ... )
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    payload = {
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "episode": int(episode),
        "config": config_dict,
        "metadata": metadata or {},
    }

    torch.save(payload, path)
    print(f"[Checkpoint] Saved  → {path}  (ep={episode}, step={step})")


# ── Load ──────────────────────────────────────────────────────────────────────


def load_checkpoint(
    path: str,
    network: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore weights (and optionally optimizer state).

    The state dict is loaded directly onto the device the network already
    lives on, so no explicit device management is needed by the caller —
    just build the network, move it to the right device, then call this.

    Args:
        path:      Path to the .pt checkpoint file.
        network:   Network whose weights will be restored in-place.
        optimizer: If provided, the optimizer state is also restored.
                   Pass None in evaluate.py (no training, no optimizer).

    Returns:
        Dict with keys:
            "step"     → int   global env step counter at save time
            "episode"  → int   episode number at save time
            "config"   → dict  serialised Config (use Config.from_dict())
            "metadata" → dict  free-form info saved alongside the model

    Raises:
        FileNotFoundError: If `path` does not exist.

    Example:
        >>> info = load_checkpoint("checkpoints/dqn_8x8_best.pt", network=net)
        >>> cfg  = Config.from_dict(info["config"])
        >>> print(f"Resumed from step {info['step']}, "
        ...       f"win_rate={info['metadata'].get('win_rate', 'n/a')}")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = _infer_device(network)

    # weights_only=False is required to load optimizer state (contains Python
    # objects).  The checkpoint files are produced by this framework only, so
    # the security trade-off is acceptable.
    payload = torch.load(path, map_location=device, weights_only=False)

    network.load_state_dict(payload["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    print(
        f"[Checkpoint] Loaded ← {path}  "
        f"(ep={payload.get('episode', '?')}, step={payload.get('step', '?')})"
    )

    return {
        "step": payload.get("step", 0),
        "episode": payload.get("episode", 0),
        "config": payload.get("config", {}),
        "metadata": payload.get("metadata", {}),
    }


# ── Path helpers ──────────────────────────────────────────────────────────────


def checkpoint_path(
    directory: str,
    run_name: str,
    episode: Optional[int] = None,
    tag: Optional[str] = None,
) -> str:
    """
    Build a canonical checkpoint file path.

    Exactly one of `episode` or `tag` should be provided.

    Args:
        directory: Root checkpoint directory (e.g. cfg.checkpoint_dir).
        run_name:  Identifier for this training run (e.g. cfg.run_name).
        episode:   Episode number → produces a timestamped filename.
        tag:       Descriptive tag → produces a stable filename.
                   Conventional tags: "best", "final", "latest".

    Returns:
        Full path string.

    Examples:
        >>> checkpoint_path("checkpoints", "dqn_8x8", episode=5000)
        "checkpoints/dqn_8x8_ep005000.pt"
        >>> checkpoint_path("checkpoints", "dqn_8x8", tag="best")
        "checkpoints/dqn_8x8_best.pt"
    """
    if episode is not None and tag is None:
        filename = f"{run_name}_ep{episode:06d}.pt"
    elif tag is not None and episode is None:
        filename = f"{run_name}_{tag}.pt"
    else:
        raise ValueError(
            "Provide exactly one of `episode` or `tag`, not both or neither."
        )
    return os.path.join(directory, filename)


def latest_checkpoint(directory: str) -> Optional[str]:
    """
    Return the most recently modified .pt file in `directory`, or None.

    Used by evaluate.py as a convenience when no explicit checkpoint path
    is passed via CLI:
        path = args.checkpoint or latest_checkpoint(cfg.checkpoint_dir)

    Args:
        directory: Directory to search (non-recursive).

    Returns:
        Absolute path to the newest .pt file, or None if none exist.
    """
    if not os.path.isdir(directory):
        return None

    candidates = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")
    ]
    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


# ── Top-k checkpoint manager ─────────────────────────────────────────────────


class TopKCheckpointManager:
    """
    Keeps only the top-k checkpoints ranked by an eval metric (higher is
    better) — mirrors PyTorch Lightning's ModelCheckpoint(save_top_k=k).

    Unlike periodic "save every N episodes" checkpointing, a checkpoint is
    only written if it would rank in the current top-k; whenever a new one
    is saved and the set is already full, the worst-ranked checkpoint is
    evicted (both from tracking and from disk).

    This class only knows "higher metric = better" — it doesn't know what
    the metric means (win rate, CI lower bound, negative loss, ...), so the
    same manager works for DQN, A2C, PPO, or anything else that calls
    maybe_save() after an eval pass.

    Usage:
        ckpt_mgr = TopKCheckpointManager(
            cfg.checkpoint_dir, cfg.run_name, k=cfg.checkpoint_top_k
        )
        ...
        ckpt_mgr.maybe_save(
            metric=eval_win_rate_ci_lower,
            network=agent.online_net,
            optimizer=agent.optimizer,
            step=global_step,
            episode=ep,
            config=cfg,
            metadata={"eval_win_rate": eval_win_rate, "algo": "dqn"},
        )
    """

    def __init__(self, directory: str, run_name: str, k: int = 1):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.directory = directory
        self.run_name = run_name
        self.k = k
        # (metric, path, episode, step), kept sorted ascending — worst first
        self._kept: list = []
        self._discover_existing()

    def _discover_existing(self) -> None:
        """
        Rebuild tracking state from any top-k checkpoints for this run_name
        already on disk, so resuming an interrupted training run doesn't
        lose track of what's kept or silently exceed k files.
        """
        if not os.path.isdir(self.directory):
            return
        prefix = f"{self.run_name}_topk_ep"
        for fname in os.listdir(self.directory):
            if not (fname.startswith(prefix) and fname.endswith(".pt")):
                continue
            path = os.path.join(self.directory, fname)
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            metric = payload.get("metadata", {}).get("rank_metric")
            if metric is None:
                continue
            episode = payload.get("episode", 0)
            step = payload.get("step", 0)
            self._kept.append((float(metric), path, episode, step))
        self._kept.sort(key=lambda t: t[0])

        # If a previous run left more than k files (e.g. k was lowered
        # since), trim immediately.
        while len(self._kept) > self.k:
            _, evict_path, evict_ep, _ = self._kept.pop(0)
            if os.path.exists(evict_path):
                os.remove(evict_path)
                print(
                    f"[Checkpoint] Evicted (stale, over top-{self.k}) "
                    f"→ {evict_path}  (ep={evict_ep})"
                )

    def maybe_save(
        self,
        metric: float,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        episode: int,
        config: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save a checkpoint for this eval result IF it belongs in the current
        top-k (higher metric = better), evicting the current worst kept
        checkpoint (tracking + disk file) if the set is already full.

        Returns True if saved, False if discarded (didn't rank in top-k).
        """
        if len(self._kept) >= self.k and metric <= self._kept[0][0]:
            return False

        path = checkpoint_path(
            self.directory, self.run_name, tag=f"topk_ep{episode:06d}"
        )
        meta = dict(metadata or {})
        meta["rank_metric"] = float(metric)
        save_checkpoint(
            path=path,
            network=network,
            optimizer=optimizer,
            step=step,
            episode=episode,
            config=config,
            metadata=meta,
        )

        self._kept.append((float(metric), path, episode, step))
        self._kept.sort(key=lambda t: t[0])

        if len(self._kept) > self.k:
            _, evict_path, evict_ep, _ = self._kept.pop(0)
            if evict_path != path and os.path.exists(evict_path):
                os.remove(evict_path)
                print(f"[Checkpoint] Evicted (stale) → {evict_path}  (ep={evict_ep})")

        return True

    @property
    def best(self) -> Optional[tuple]:
        """(metric, path) of the current best kept checkpoint, or None."""
        if not self._kept:
            return None
        m, p, _, _ = max(self._kept, key=lambda t: t[0])
        return (m, p)

    def __repr__(self) -> str:
        kept = ", ".join(
            f"{m:.3f}" for m, *_ in sorted(self._kept, key=lambda t: -t[0])
        )
        return f"TopKCheckpointManager(k={self.k}, kept_metrics=[{kept}])"
