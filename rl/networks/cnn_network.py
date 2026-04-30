"""
rl/networks/cnn_network.py

Minesweeper-specific CNN Q-network.

This file is intentionally problem-coupled.  The rest of the library
(DQN algorithm, replay buffer, logger, config) is environment-agnostic.
This network and dqn_agent.py are the two files that know about
Minesweeper.

Observation contract
--------------------
Expects obs.spatial — the (1, n, n) float32 spatial representation
of the board, where each cell value is board_value / 9.0 ∈ [0, 1].

    Cell encoding (from MinesweeperObservation.spatial):
        0.0          → unrevealed cell (unknown)
        0.1 – 0.8   → revealed, clue digit 1–8  (neighbours / 9.0)
        0.0          → revealed, safe  (0 neighbours — same as unrevealed,
                        but the env distinguishes via action_mask)
        0.0          → flagged cell    (handled externally)

    Shape fed to the network: (B, 1, n, n)
    Shape returned:           (B, n*n)   — raw Q-value per cell

Why CNN and not MLP
-------------------
Minesweeper is a spatial deduction game.  Whether cell (r, c) is safe
depends entirely on the clue values in the 3×3 neighbourhood around it.
A 3×3 Conv2d with padding=1 computes exactly this neighbourhood
aggregation — it is the right inductive bias for the task.

An MLP receives a flat vector and has no structural knowledge that
index i and index i+1 are adjacent.  It would have to re-learn
spatial relationships through coincidental weight patterns.

Architecture
------------
    input : (B, 1, n, n)
    ↓  Conv2d(  1 → 32, 3×3, padding=1) + ReLU   # each cell sees its 8 neighbours
    ↓  Conv2d( 32 → 64, 3×3, padding=1) + ReLU   # 5×5 compound receptive field
    ↓  Flatten  →  (B, 64·n²)
    ↓  Linear(64·n², hidden_dim)         + ReLU
    ↓  Linear(hidden_dim, n²)
    output : (B, n²)  — one raw Q-value per board cell

padding=1 keeps spatial dimensions at n×n after every conv layer so
every cell — including corners and edges — is treated symmetrically.

No BatchNorm
------------
BatchNorm2d is incompatible with DQN's two forward-pass contexts:

    action selection   batch_size = 1     running stats are stale
    minibatch update   batch_size = 64    recomputes stats from batch
    target network     never updated      running stats freeze and diverge

Kaiming initialisation + ReLU keeps activations well-scaled without it.

Parameter counts
----------------
    n =  6, hidden = 256 :    618,148
    n =  8, hidden = 256 :  1,084,096
    n = 10, hidden = 256 :  1,683,172
    n = 16, hidden = 256 :  4,279,168
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class MinesweeperCNN(nn.Module):
    """
    CNN Q-network for Minesweeper.

    Takes a spatial board observation and returns a Q-value for every cell.
    Designed to work with MinesweeperObservation.spatial directly — no
    reshaping or normalisation required before calling forward().

    Args:
        n:          Board side length (4 ≤ n ≤ 16).
                    Determines input shape (1, n, n) and output size (n²).
        hidden_dim: Width of the fully-connected hidden layer.
                    256 works well up to 16×16.  Drop to 128 for faster
                    experiments on small boards.

    Example:
        >>> net = MinesweeperCNN(n=8, hidden_dim=256)
        >>> obs_batch = torch.zeros(32, 1, 8, 8)   # 32 obs.spatial stacked
        >>> q_values  = net(obs_batch)              # (32, 64)
        >>> action    = q_values[0].argmax().item() # greedy cell index
    """

    def __init__(self, n: int, hidden_dim: int = 256) -> None:
        super().__init__()

        assert 4 <= n <= 16, f"Board size n must be 4–16, got {n}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"

        self.n = n
        self.num_actions = n * n
        self.hidden_dim = hidden_dim

        # ── Convolutional trunk ───────────────────────────────────────────
        # No BatchNorm — see module docstring for why.
        # padding=1 preserves spatial dims at n×n after each layer.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 1, n, n) → (B, 32, n, n)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, padding=1
            ),  # (B, 32, n, n) → (B, 64, n, n)
            nn.ReLU(),
        )

        # ── Fully-connected head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(64 * n * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n * n),  # raw Q-values, no activation
        )

        # ── Weight initialisation ─────────────────────────────────────────
        # PyTorch default (Kaiming uniform) is correct for ReLU activations.
        # Zero the output bias so Q-values start near 0 at the beginning
        # of training, making the first gradient steps more stable.
        last_layer = cast(nn.Linear, self.head[-1])
        nn.init.zeros_(last_layer.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a batch of spatial observations.

        Args:
            x: (B, 1, n, n) float32.
               Pass obs.spatial directly — already (1, n, n) and
               normalised to [0, 1] by the environment.

        Returns:
            (B, n²) float32 — raw Q-values, one per board cell.
            No activation on the output (unbounded, can be negative).
        """
        x = self.conv(x)  # (B, 64, n, n)
        x = x.flatten(start_dim=1)  # (B, 64·n²)
        return self.head(x)  # (B, n²)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MinesweeperCNN("
            f"n={self.n}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={self.num_parameters():,})"
        )
