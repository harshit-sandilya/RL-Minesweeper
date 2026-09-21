import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Fully convolutional policy network for Minesweeper.

    Same conv trunk as MinesweeperCNN (rl/networks/q_network.py) so
    DQN/A2C/PPO share an identical-capacity feature extractor for a fair
    algorithm comparison — only the output head's meaning differs: here
    the output is a raw logit per cell, not a Q-value.

    Outputs UNMASKED, UNNORMALIZED logits. Action masking (setting
    illegal-cell logits to -inf before softmax/sampling) is the caller's
    responsibility — done identically in both the agent (rollout time)
    and rl.algorithms.a2c.A2C.update() (recomputing log-probs), so both
    always sample from / evaluate the same masked distribution.

    Args:
        n:          Board side length (4 ≤ n ≤ 16).
        hidden_dim: Hidden channel width of the conv trunk. 256 to match
                    MinesweeperCNN's Minesweeper-DQN default.

    Example:
        >>> actor = ActorNetwork(n=8, hidden_dim=256)
        >>> obs_batch = torch.zeros(32, 3, 8, 8)
        >>> logits = actor(obs_batch)                        # (32, 64)
        >>> masked = logits.masked_fill(~mask, -1e8)
        >>> dist = torch.distributions.Categorical(logits=masked)
        >>> action = dist.sample()
    """

    def __init__(self, n: int, hidden_dim: int = 256) -> None:
        super().__init__()

        assert 4 <= n <= 16, f"Board size n must be 4–16, got {n}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"

        self.n = n
        self.num_actions = n * n
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, n, n) float32 — obs.spatial, stacked.

        Returns:
            (B, n²) float32 — raw per-cell logits. No masking, no softmax.
        """
        logit_map = self.net(x)  # (B, 1, n, n)
        return logit_map.flatten(start_dim=1)  # (B, n²)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ActorNetwork("
            f"n={self.n}, "
            f"hidden_channels={self.hidden_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={self.num_parameters():,})"
        )
