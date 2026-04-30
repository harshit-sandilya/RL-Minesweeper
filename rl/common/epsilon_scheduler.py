"""
rl/common/epsilon_scheduler.py

Epsilon-greedy exploration schedules for value-based RL agents.

Both schedulers are stateless — they compute ε purely from the step
argument, so they can be called multiple times at the same step (e.g.
for logging) without side effects.  The agent owns the step counter.

Usage:
    scheduler = make_scheduler(cfg)

    # inside the agent's step loop:
    eps = scheduler.get(global_step)
    action = agent.select_action(obs, eps)

    # both schedulers guarantee ε_end ≤ ε ≤ ε_start at every step
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.common.config import Config


# ── Linear schedule ───────────────────────────────────────────────────────────


class LinearEpsilonScheduler:
    """
    Linearly anneal ε from `start` down to `end` over `decay_steps` steps.

    Formula:
        ε(t) = max(ε_end,  ε_start − t × (ε_start − ε_end) / decay_steps)

    After `decay_steps` steps ε stays fixed at ε_end forever.

    Args:
        start:       Initial ε (e.g. 1.0  — fully random at the beginning).
        end:         Final ε   (e.g. 0.1  — 10 % random exploration at the end).
        decay_steps: Number of environment steps over which to anneal.

    Example:
        >>> sched = LinearEpsilonScheduler(1.0, 0.1, 100_000)
        >>> sched.get(0)           # 1.0
        >>> sched.get(50_000)      # 0.55
        >>> sched.get(100_000)     # 0.1
        >>> sched.get(999_999)     # 0.1  (clipped at end)
    """

    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        assert 0.0 <= end <= start <= 1.0, (
            f"Need 0 ≤ end ≤ start ≤ 1, got start={start}, end={end}"
        )
        assert decay_steps > 0, f"decay_steps must be positive, got {decay_steps}"

        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self._slope = (start - end) / decay_steps  # pre-computed

    def get(self, step: int) -> float:
        """
        Return ε at global environment step `step`.

        Args:
            step: Total number of environment steps taken so far (≥ 0).

        Returns:
            ε ∈ [end, start].
        """
        return max(self.end, self.start - step * self._slope)

    def __repr__(self) -> str:
        return (
            f"LinearEpsilonScheduler("
            f"start={self.start}, end={self.end}, "
            f"decay_steps={self.decay_steps})"
        )


# ── Exponential schedule ──────────────────────────────────────────────────────


class ExponentialEpsilonScheduler:
    """
    Exponentially decay ε from `start` down to `end` over `decay_steps` steps.

    The decay rate is derived automatically so ε reaches exactly `end`
    after `decay_steps` steps, matching the LinearScheduler's interface —
    same three constructor arguments, same .get(step) call.

    Formula:
        decay_rate = (ε_end / ε_start) ^ (1 / decay_steps)
        ε(t) = max(ε_end,  ε_start × decay_rate ^ t)

    Exponential decay stays high for longer than linear then drops sharply,
    which can be useful when the agent needs more early exploration on
    large boards where random play rarely reaches interesting states.

    Args:
        start:       Initial ε (e.g. 1.0).
        end:         Final ε   (e.g. 0.05).
        decay_steps: Steps until ε reaches `end`.

    Example:
        >>> sched = ExponentialEpsilonScheduler(1.0, 0.05, 100_000)
        >>> sched.get(0)           # 1.0
        >>> sched.get(50_000)      # ~0.224   (lower than linear's 0.525 — drops faster early)
        >>> sched.get(100_000)     # 0.05
        >>> sched.get(999_999)     # 0.05  (clipped at end)
    """

    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        assert 0.0 < end <= start <= 1.0, (
            f"Need 0 < end ≤ start ≤ 1 for exponential, "
            f"got start={start}, end={end}  "
            f"(end must be > 0 to avoid log(0))"
        )
        assert decay_steps > 0, f"decay_steps must be positive, got {decay_steps}"

        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self._log_decay = math.log(end / start) / decay_steps

    def get(self, step: int) -> float:
        """
        Return ε at global environment step `step`.

        Args:
            step: Total number of environment steps taken so far (≥ 0).

        Returns:
            ε ∈ [end, start].
        """
        return max(self.end, self.start * math.exp(self._log_decay * step))

    def __repr__(self) -> str:
        decay_rate = math.exp(self._log_decay)
        return (
            f"ExponentialEpsilonScheduler("
            f"start={self.start}, end={self.end}, "
            f"decay_steps={self.decay_steps}, "
            f"rate_per_step={decay_rate:.6f})"
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def make_scheduler(
    cfg: "Config",
) -> LinearEpsilonScheduler | ExponentialEpsilonScheduler:
    """
    Build the correct scheduler from a Config object.

    Reads cfg.epsilon_schedule ("linear" | "exponential"),
    cfg.epsilon_start, cfg.epsilon_end, and cfg.epsilon_decay_steps.
    Config.__post_init__ already validated these values so no extra
    checks are needed here.

    Args:
        cfg: Fully validated Config instance.

    Returns:
        Ready-to-use scheduler.

    Raises:
        ValueError: If cfg.epsilon_schedule is not a recognised schedule name.
    """
    if cfg.epsilon_schedule == "linear":
        return LinearEpsilonScheduler(
            start=cfg.epsilon_start,
            end=cfg.epsilon_end,
            decay_steps=cfg.epsilon_decay_steps,
        )
    elif cfg.epsilon_schedule == "exponential":
        return ExponentialEpsilonScheduler(
            start=cfg.epsilon_start,
            end=cfg.epsilon_end,
            decay_steps=cfg.epsilon_decay_steps,
        )
    else:
        raise ValueError(
            f"Unknown epsilon_schedule '{cfg.epsilon_schedule}'. "
            f"Choose 'linear' or 'exponential'."
        )
