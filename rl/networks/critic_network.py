import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Fully convolutional state-value network for Minesweeper.

    Same 3-layer conv trunk as MinesweeperCNN / ActorNetwork, but the head
    differs: a critic outputs ONE scalar V(s) per board, not one value per
    cell, so the spatial feature map is global-average-pooled before a
    linear projection to a scalar.

    Args:
        n:          Board side length (4 ≤ n ≤ 16). Kept for API symmetry
                    with ActorNetwork/MinesweeperCNN even though the
                    critic's output size doesn't depend on it (pooling
                    removes the spatial dimension).
        hidden_dim: Hidden channel width of the conv trunk. Should match
                    the paired ActorNetwork's hidden_dim.

    Example:
        >>> critic = CriticNetwork(n=8, hidden_dim=256)
        >>> obs_batch = torch.zeros(32, 3, 8, 8)
        >>> values = critic(obs_batch)                        # (32, 1)
        >>> values = values.squeeze(-1)                        # (32,)
    """

    def __init__(self, n: int, hidden_dim: int = 256) -> None:
        super().__init__()

        assert 4 <= n <= 16, f"Board size n must be 4–16, got {n}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"

        self.n = n
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(
            1
        )  # (B, hidden_dim, n, n) -> (B, hidden_dim, 1, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, n, n) float32 — obs.spatial, stacked.

        Returns:
            (B, 1) float32 — V(s) for each board in the batch.
        """
        features = self.trunk(x)  # (B, hidden_dim, n, n)
        pooled = self.pool(features).flatten(start_dim=1)  # (B, hidden_dim)
        return self.value_head(pooled)  # (B, 1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CriticNetwork("
            f"n={self.n}, "
            f"hidden_channels={self.hidden_dim}, "
            f"params={self.num_parameters():,})"
        )
