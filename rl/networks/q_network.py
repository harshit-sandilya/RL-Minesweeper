"""
rl/networks/q_network.py

Multi-layer perceptron Q-network for discrete-action value-based RL.

Takes a flat state vector and outputs one Q-value per action.
The architecture (number and width of hidden layers) is decided at
construction time in train.py — not stored in Config, because network
design is a code-level decision, not a hyperparameter swept over a YAML.

Input / output contract
───────────────────────
    input  : (B, state_dim)   float32, values in [0.0, 1.0]
                                → obs.flat from MinesweeperObservation
    output : (B, num_actions)  float32, raw Q-values (no activation)
                                → one value per possible cell click

Architecture
────────────
    Linear(state_dim → h₀) → ReLU
    Linear(h₀ → h₁)        → ReLU
    ...
    Linear(hₙ → num_actions)          ← no activation on output layer

Usage
─────
    net = QNetwork(state_dim=64, num_actions=64, hidden_dims=[128, 128])
    net = net.to(device)

    # single state (no batch dim):
    state = torch.from_numpy(obs.flat).unsqueeze(0).to(device)  # (1, 64)
    q_values = net(state)                                         # (1, 64)

    # action masking before argmax:
    mask = torch.from_numpy(obs.action_mask).to(device)          # (64,) bool
    q_values[0][~mask] = -float("inf")
    action = q_values.argmax(dim=1).item()                        # int
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    MLP that maps a flat state observation to per-action Q-values.

    Layers are built from `hidden_dims` so depth and width are fully
    configurable without touching any other file.  All hidden activations
    are ReLU; the output layer has no activation (raw Q-values can be
    any real number).

    PyTorch's default initialisation for nn.Linear is Kaiming-uniform
    with a=√5, which is appropriate for ReLU networks and requires no
    additional setup.

    Args:
        state_dim:   Length of the flat input vector.
                     For an n×n Minesweeper board: state_dim = n * n.
        num_actions: Number of discrete output actions.
                     For Minesweeper: num_actions = n * n
                     (one action per cell).
        hidden_dims: List of hidden layer widths, e.g. [128, 128].
                     Must contain at least one element.

    Example:
        >>> net = QNetwork(state_dim=64, num_actions=64,
        ...                hidden_dims=[128, 128])
        >>> x = torch.zeros(32, 64)    # batch of 32 states
        >>> net(x).shape
        torch.Size([32, 64])
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()

        assert state_dim > 0, f"state_dim must be positive, got {state_dim}"
        assert num_actions > 0, f"num_actions must be positive, got {num_actions}"
        assert len(hidden_dims) >= 1, "hidden_dims must have at least one element"
        assert all(h > 0 for h in hidden_dims), (
            f"all hidden_dims must be positive, got {hidden_dims}"
        )

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dims = list(hidden_dims)

        # ── Build layers ──────────────────────────────────────────────────
        # Interleave Linear + ReLU for every hidden layer, then a final
        # Linear with no activation for the Q-value outputs.
        layers: list[nn.Module] = []
        in_dim = state_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, num_actions))  # output — no activation

        self.net = nn.Sequential(*layers)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a batch of states.

        Args:
            x: Float32 tensor of shape (B, state_dim).
               Values should be pre-normalised to [0.0, 1.0]
               (obs.flat already satisfies this via /9.0).

        Returns:
            Float32 tensor of shape (B, num_actions) containing raw
            Q-values.  No activation is applied — the output can be
            any real number.
        """
        return self.net(x)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        """
        Total number of trainable parameters.

        Useful for logging at the start of training and for comparing
        model capacity across architecture sweeps.

        Example:
            QNetwork(64, 64, [128, 128]) →
                64×128 + 128  = 8_320
                128×128 + 128 = 16_512
                128×64  + 64  = 8_256
                total         = 33_088
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        dims = [self.state_dim] + self.hidden_dims + [self.num_actions]
        arch = " → ".join(str(d) for d in dims)
        return f"QNetwork({arch}, params={self.num_parameters():,})"
