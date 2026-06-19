"""
Data models for the Minesweeper RL Environment.

Observation encoding (matches game_engine.py):
    0–8  : revealed safe cell — adjacency count
    9    : unrevealed (Cell.UNKNOWN) — safe and mine cells look identical to agent
    Range [0, 9] — normalize by /9.0 for all neural net inputs
"""

from typing import List

import numpy as np
from openenv.core.env_server import Action, Observation, State
from pydantic import computed_field


# ── Action ────────────────────────────────────────────────────────────────────
class MinesweeperAction(Action):
    """
    A single cell click. The only action type in Minesweeper.

    Attributes:
        row (int): Row coordinate (0 to n-1)
        col (int): Column coordinate (0 to n-1)

    Bounds are coarse-gated here (max grid = 10×10).
    Fine bounds (< n) are enforced by the engine's step().
    """

    row: int
    col: int


# ── Observation ───────────────────────────────────────────────────────────────
class MinesweeperObservation(Observation):
    """
    Everything the agent sees after each step.

    Attributes:
        board (List[List[int]]): 2D game board where:
            - 0-8: revealed safe cell (adjacency count)
            - 9: unrevealed cell (both safe and mines)
        n (int): Grid side length
        mines_count (int): Total mines on board
        unrevealed_count (int): Number of unrevealed cells
        status (int): Game status (0=ONGOING, 1=WON, 2=LOST)
        message (str): Human-readable step summary

    Inherited from Observation base:
        done   : bool
        reward : Optional[float]
    """

    # ── Core fields ──────────────────────────────────────────────────────────
    board: List[List[int]]
    n: int  # grid side length
    mines_count: int  # total mines on board
    unrevealed_count: int  # convenience: equals (obs == 9).sum()
    status: int  # 0=ONGOING, 1=WON, 2=LOST
    message: str  # human-readable step summary (server-generated)

    # ── Raw array ────────────────────────────────────────────────────────────
    @property
    def board_array(self) -> np.ndarray:
        """
        (n, n) uint8 — the canonical board observation.

        Values:
            0–8 : revealed safe cell (adjacency count)
            9   : unrevealed (Cell.UNKNOWN) — mine cells also show as 9
        """
        return np.array(self.board, dtype=np.uint8)

    # ── Agent-ready views ────────────────────────────────────────────────────
    @property
    def flat(self) -> np.ndarray:
        """
        (n*n,) float32 in [0.0, 1.0] — direct input for MLP / DQN.

        Usage:
            state_tensor = torch.from_numpy(obs.flat)
        """
        return self.board_array.flatten().astype(np.float32) / 9.0

    @property
    def spatial(self) -> np.ndarray:
        """
        (3, n, n) float32 spatial representation.

        Channel 0:
            hidden mask
                1.0 = hidden
                0.0 = revealed

        Channel 1:
            revealed clue values normalized to [0,1]
                hidden cells = 0.0
                revealed 0-8 = value / 8.0

        Channel 2:
            frontier mask
                1.0 = cell is hidden AND adjacent to at least one revealed cell
                0.0 = otherwise
        """
        board = self.board_array

        # ── Channel 0: Hidden mask ───────────────────────────────────────────
        hidden = board == 9
        hidden_float = hidden.astype(np.float32)

        # ── Channel 1: Clue values ───────────────────────────────────────────
        clues = np.where(hidden, 0.0, board / 8.0).astype(np.float32)

        # ── Channel 2: Frontier mask ─────────────────────────────────────────
        revealed = ~hidden

        # Pad with False to easily handle edge boundaries when shifting
        padded_revealed = np.pad(
            revealed, pad_width=1, mode="constant", constant_values=False
        )
        adjacent_to_revealed = np.zeros_like(revealed, dtype=bool)

        # Shift the padded mask in all 8 directions to find adjacencies
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adjacent_to_revealed |= padded_revealed[
                    1 + dr : 1 + dr + self.n, 1 + dc : 1 + dc + self.n
                ]

        frontier = (hidden & adjacent_to_revealed).astype(np.float32)

        return np.stack([hidden_float, clues, frontier], axis=0)

    @property
    def action_mask(self) -> np.ndarray:
        """
        (n*n,) bool — True where a click is meaningful (cell is unrevealed).

        Usage (DQN with invalid action masking):
            q_values[~obs.action_mask] = -inf   # mask out revealed cells
            action = q_values.argmax()
            row, col = divmod(action, obs.n)
        """
        return self.board_array.flatten() == 9  # 9 == Cell.UNKNOWN


# ── State ─────────────────────────────────────────────────────────────────────
class MinesweeperState(State):
    """
    Server-side episode metadata (not the board itself).

    Attributes:
        n (int): Grid side length for the active episode
        mines_count (int): Number of mines for the active episode
        solve_tiles (int): Number of safe tiles pre-revealed for the active episode

    Computed properties:
        mine_density (float): mines / total_cells
        safe_cells (int): total_safe_cells = n² - mines_count

    Inherited from State base:
        episode_id : Optional[str]
        step_count : int
    """

    n: int
    mines_count: int  # matches engine attribute name
    solve_tiles: int  # number of safe tiles pre-revealed at episode start

    @computed_field
    @property
    def mine_density(self) -> float:
        """
        mines / total_cells — the primary complexity metric.

        Standard Minesweeper difficulties for reference:
            Beginner     : 9×9,  10 mines → 0.123
            Intermediate : 16×16, 40 mines → 0.156
            Expert       : 16×30, 99 mines → 0.206
        """
        return round(self.mines_count / (self.n**2), 4)

    @computed_field
    @property
    def safe_cells(self) -> int:
        """Total non-mine cells — the 'problem size' the agent must solve."""
        return self.n**2 - self.mines_count
