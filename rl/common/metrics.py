"""
rl/common/metrics.py

Algorithm-agnostic evaluation metrics/reporting for the Minesweeper RL study.
Any agent's evaluate() (DQN, A2C, PPO, ...) should return a list[dict] where
each dict has at least these keys, so results are directly comparable across
algorithms and board sizes:

    reward, steps, win, revealed_safe, total_safe, wall_clock_sec

Optional keys (algo-specific, e.g. q_mean, entropy) are passed through untouched.
"""
from __future__ import annotations

import csv
import math
import statistics as stats
from pathlib import Path
from typing import Sequence

# Ratio edges (fraction of total_safe revealed at episode end). Top edge = win.
DEFAULT_COMPLETION_BINS = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. More reliable than
    a normal approximation when win rate is near 0 or 1, which happens a lot
    on small boards / early checkpoints / small eval budgets."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def bucket_by_completion(
    results: Sequence[dict],
    bin_edges: Sequence[float] = DEFAULT_COMPLETION_BINS,
) -> list[dict]:
    """
    Buckets episodes by revealed_safe/total_safe at episode end, so buckets
    are comparable across board sizes. Wins are always their own top bucket
    (ratio == 1.0 by construction: a win means every safe cell got revealed).

    Returns buckets worst -> best, e.g. for an 8x8/10-mine board
    (total_safe=54):
        [{"label": "<25% (0-13 cells)",    "count": 7,  "pct": 0.13, "win": False},
         ...,
         {"label": "Won (54 cells)",        "count": 42, "pct": 0.42, "win": True}]
    """
    n = len(results)
    if n == 0:
        return []

    total_safe = results[0].get("total_safe")
    wins = [r for r in results if r["win"]]
    losses = [r for r in results if not r["win"]]

    buckets = []
    edges = list(bin_edges)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bucket = [
            r for r in losses
            if lo <= (r["revealed_safe"] / r["total_safe"]) < hi
        ]
        pct_label = f"{int(lo * 100)}-{int(hi * 100)}%" if lo > 0 else f"<{int(hi * 100)}%"
        if total_safe:
            lo_c, hi_c = int(lo * total_safe), int(hi * total_safe)
            label = f"{pct_label} ({lo_c}-{hi_c} cells)"
        else:
            label = pct_label
        buckets.append({
            "label": label,
            "count": len(in_bucket),
            "pct": len(in_bucket) / n,
            "win": False,
        })

    win_label = f"Won ({total_safe} cells)" if total_safe else "Won (100%)"
    buckets.append({
        "label": win_label,
        "count": len(wins),
        "pct": len(wins) / n,
        "win": True,
    })
    return buckets


def summarize_results(
    results: Sequence[dict],
    algorithm: str = "",
    bin_edges: Sequence[float] = DEFAULT_COMPLETION_BINS,
) -> dict:
    """Aggregate a list of per-episode result dicts (as returned by
    agent.evaluate()) into the summary stats a comparison table needs."""
    n = len(results)
    assert n > 0, "summarize_results() called with zero episodes"

    wins = sum(r["win"] for r in results)
    rewards = [r["reward"] for r in results]
    lengths = [r["steps"] for r in results]
    wall_clocks = [r.get("wall_clock_sec", 0.0) for r in results]
    total_wc = sum(wall_clocks)

    buckets = bucket_by_completion(results, bin_edges)
    loss_buckets = [b for b in buckets if not b["win"]]
    n_losses = n - wins
    early_death_pct = (loss_buckets[0]["count"] / n_losses) if loss_buckets and n_losses else 0.0
    late_death_pct = (loss_buckets[-1]["count"] / n_losses) if loss_buckets and n_losses else 0.0

    ci_lo, ci_hi = wilson_ci(wins, n)

    return {
        "algorithm": algorithm,
        "board_n": results[0].get("board_n"),
        "mines_count": results[0].get("mines_count"),
        "n_episodes": n,
        "win_rate": wins / n,
        "win_rate_ci": (ci_lo, ci_hi),
        "avg_reward": stats.mean(rewards),
        "std_reward": stats.pstdev(rewards) if n > 1 else 0.0,
        "avg_episode_len": stats.mean(lengths),
        "median_episode_len": stats.median(lengths),
        "std_episode_len": stats.pstdev(lengths) if n > 1 else 0.0,
        "avg_wall_clock_sec": total_wc / n if n else 0.0,
        "total_wall_clock_sec": total_wc,
        "episodes_per_sec": (n / total_wc) if total_wc > 0 else 0.0,
        "completion_buckets": buckets,
        "early_death_pct": early_death_pct,
        "late_death_pct": late_death_pct,
    }


def format_report(summary: dict) -> str:
    """Pretty console table, ready to paste into a paper/README."""
    header = (
        f"{summary['algorithm'].upper()} — {summary['board_n']}x{summary['board_n']}, "
        f"{summary['mines_count']} mines  (n={summary['n_episodes']} episodes)"
    )
    width = max(60, len(header) + 4)
    lines = ["=" * width, f"  {header}", "=" * width]

    ci_lo, ci_hi = summary["win_rate_ci"]
    lines.append(f"  Win rate        : {summary['win_rate']:.1%}  (95% CI: {ci_lo:.1%}-{ci_hi:.1%})")
    lines.append(f"  Avg reward      : {summary['avg_reward']:.3f}  (std={summary['std_reward']:.3f})")
    lines.append(
        f"  Episode length  : {summary['avg_episode_len']:.1f} avg / "
        f"{summary['median_episode_len']:.1f} median  (std={summary['std_episode_len']:.1f})"
    )
    lines.append(
        f"  Speed           : {summary['episodes_per_sec']:.1f} ep/s  "
        f"({summary['avg_wall_clock_sec'] * 1000:.1f} ms/ep, "
        f"{summary['total_wall_clock_sec']:.1f}s total)"
    )
    lines.append("-" * width)
    lines.append("  Board completion at episode end:")
    for b in summary["completion_buckets"]:
        lines.append(f"    {b['label']:>24} : {b['pct']:5.1%}  ({b['count']} episodes)")
    lines.append("-" * width)
    lines.append(f"  Early deaths (bottom bucket, of losses): {summary['early_death_pct']:.1%}")
    lines.append(f"  Late deaths  (near-win bucket, of losses): {summary['late_death_pct']:.1%}")
    lines.append("=" * width)
    return "\n".join(lines)


def save_results_csv(results: Sequence[dict], path: str) -> None:
    """Dump raw per-episode results to CSV for later cross-algorithm /
    board-size analysis in pandas (learning curves, completion histograms,
    significance tests across seeds, etc)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
