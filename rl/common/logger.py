"""
rl/common/logger.py

Minimal, generic TensorBoard logger.

Philosophy
----------
The logger knows nothing about environments, algorithms, or metrics.
It has one job: write (key, value, step) triples to TensorBoard and
optionally flush them to a summary CSV at the end of a run.

What to log and when is entirely the caller's responsibility:

    # In dqn_agent.py
    logger.log_scalar("train/episode_reward", ep_reward,  episode)
    logger.log_scalar("train/steps_per_ep",   ep_steps,   episode)

    # In algorithm update step
    logger.log_scalar("train/loss", loss, global_step)
    logger.log_scalar("train/lr", lr, global_step)

This keeps all metric naming decisions close to the code that
produces the values, not buried in a logger that has to import
domain concepts to know what to log.

Usage
-----
    # As an object
    logger = Logger(log_dir="runs", run_name="dqn_8x8")
    logger.log_scalar("train/loss", 0.42, step=1000)
    logger.dump_summary("runs/dqn_8x8/summary.csv")
    logger.close()

    # As a context manager (auto-closes)
    with Logger(log_dir="runs", run_name="dqn_8x8") as logger:
        logger.log_scalar("train/loss", 0.42, step=1000)
    # writer flushed + closed here
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Thin wrapper around TensorBoard SummaryWriter.

    Exposes a single log_scalar interface.  All metric naming and
    logging frequency decisions are left to the caller.

    Args:
        log_dir:  Root directory for TensorBoard event files.
        run_name: Sub-folder name for this run.
                  Final write path: log_dir/run_name/

    Attributes:
        run_dir: Resolved path where TensorBoard events are written.
    """

    def __init__(self, log_dir: str, run_name: str) -> None:
        self.run_dir = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.run_dir))

        # Stores the most recent (value, step) for every key.
        # Used by dump_summary() to write a flat CSV at run end.
        self._last: Dict[str, Tuple[float, int]] = {}

    # ── Core interface ─────────────────────────────────────────────────────────

    def log_scalar(
        self,
        key: str,
        value: Union[float, int],
        step: int,
    ) -> None:
        """
        Write a scalar value to TensorBoard.

        Args:
            key:   Metric name, e.g. "train/loss" or "perf/inference_ms".
                   Use "/" to create TensorBoard sections.
            value: Numeric value.
            step:  Global step counter (episode number, update step, etc.).
        """
        self._writer.add_scalar(key, value, global_step=step)
        self._last[key] = (float(value), step)

    def log_scalars(
        self,
        scalars: Dict[str, Union[float, int]],
        step: int,
    ) -> None:
        """
        Write multiple scalars in one call.

        Args:
            scalars: {key: value} mapping.
            step:    Shared global step for all keys in this call.

        Example:
            logger.log_scalars({
                "train/loss":     loss,
                "train/lr":       lr,
                "train/q_mean":   q_mean,
            }, step=global_step)
        """
        for key, value in scalars.items():
            self.log_scalar(key, value, step)

    # ── Persistence ────────────────────────────────────────────────────────────

    def dump_summary(self, path: Optional[str] = None) -> Path:
        """
        Write the last logged value for every key to a CSV file.

        Useful for compare.py: load summary.csv from each run and
        build a cross-algorithm comparison table without parsing
        TensorBoard event files.

        Args:
            path: Output CSV path.
                  Defaults to run_dir/summary.csv.

        Returns:
            Path to the written CSV file.

        CSV columns: key, value, step
        """
        out = Path(path) if path else self.run_dir / "summary.csv"
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value", "step"])
            for key, (value, step) in sorted(self._last.items()):
                writer.writerow([key, value, step])
        return out

    def flush(self) -> None:
        """Force-write pending events to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self._writer.flush()
        self._writer.close()

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Logger(run_dir='{self.run_dir}', keys_logged={len(self._last)})"
