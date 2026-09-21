import torch
import torch.nn as nn


class MinesweeperCNN(nn.Module):
    """
    Fully Convolutional Q-network for Minesweeper.

    Takes a spatial board observation and returns a Q-value for every cell.
    Designed to work with MinesweeperObservation.spatial directly — no
    reshaping or normalisation required before calling forward().

    Args:
        n:          Board side length (4 ≤ n ≤ 16).
                    Determines output size (n²).
        hidden_dim: Number of hidden channels in the conv layers.
                    64 works well for learning local patterns.

    Example:
        >>> net = MinesweeperCNN(n=8, hidden_dim=64)
        >>> obs_batch = torch.zeros(32, 3, 8, 8)    # 32 obs.spatial stacked (3 channels)
        >>> q_values  = net(obs_batch)              # (32, 64)
        >>> action    = q_values[0].argmax().item() # greedy cell index
    """

    def __init__(self, n: int, hidden_dim: int = 64) -> None:
        super().__init__()

        assert 4 <= n <= 16, f"Board size n must be 4–16, got {n}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"

        self.n = n
        self.num_actions = n * n
        self.hidden_dim = hidden_dim

        # ── Fully Convolutional Network ───────────────────────────────────
        # padding=1 preserves spatial dims at n×n after each layer.
        # The final 1x1 conv outputs exactly 1 channel (the Q-value per cell).
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a batch of spatial observations.

        Args:
            x: (B, 3, n, n) float32.
               Pass obs.spatial directly — already (3, n, n) and
               normalised to [0, 1] by the environment.

        Returns:
            (B, n²) float32 — raw Q-values, one per board cell.
            No activation on the output (unbounded, can be negative).
        """
        q_map = self.net(x)  # (B, 1, n, n)
        return q_map.flatten(start_dim=1)  # (B, n²)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MinesweeperCNN("
            f"n={self.n}, "
            f"hidden_channels={self.hidden_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={self.num_parameters():,})"
        )
