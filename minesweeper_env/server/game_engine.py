# game_engine.py
from __future__ import annotations

import itertools
import random
from collections import deque
from enum import IntEnum
from typing import Optional, Tuple, List

import numpy as np


class Cell(IntEnum):
    MINE = 16
    UNKNOWN = 255


class GameStatus(IntEnum):
    ONGOING = 0
    WON = 1
    LOST = 2


class MinesweeperEngine:
    """
    Self-contained Minesweeper engine.

    Each instance is fully independent — instantiate one per agent
    for concurrent / multi-agent training.

    Args:
        n           : grid side length (4–10)
        mines       : number of mines
        solve_tiles : safe tiles pre-revealed at reset (curriculum difficulty)
        seed        : optional RNG seed for reproducibility
    """

    _DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(
        self,
        n: int,
        mines: int,
        solve_tiles: int = 0,
        seed: Optional[int] = None,
    ):
        assert 4 <= n <= 10, f"Grid size must be 4–10, got {n}"
        assert mines < n**2 - 1, f"Too many mines ({mines}) for a {n}×{n} grid"

        self.n = n
        self.mines_count = mines
        self.solve_tiles = solve_tiles
        self._rng = random.Random(seed)

        # Initialised properly in reset()
        self.grid: List[List[int]] = []
        self.view: List[List[int]] = []
        self.revealed: set[Tuple[int, int]] = set()
        self.status: GameStatus = GameStatus.ONGOING
        self.step_count: int = 0

        self._last_action: Optional[Tuple[int, int]] = None

        self._build()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Rebuild the board and return the initial observation."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._build()
        return self.observe()

    def step(self, row: int, col: int) -> Tuple[np.ndarray, float, bool]:
        """
        Uncover cell (row, col).

        Returns:
            observation : (n, n, 3) uint8 array
            reward      : float
            done        : bool
        """
        if self.status != GameStatus.ONGOING:
            raise RuntimeError("Game is over — call reset() first.")
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise ValueError(f"({row},{col}) out of bounds for {self.n}×{self.n} grid.")

        self._last_action = (row, col)

        # Penalise re-clicking a revealed cell without ending the episode
        if (row, col) in self.revealed:
            return self.observe(), -0.5, False

        self.step_count += 1

        if self.grid[row][col] == Cell.MINE:
            self._reveal(row, col)
            self.status = GameStatus.LOST
            return self.observe(), -1.0, True

        self._flood_fill(row, col)

        if self._check_win():
            self.status = GameStatus.WON
            return self.observe(), 1.0, True

        return self.observe(), 0.0, False

    def observe(self, last_action: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        (n, n, 3) uint8 observation tensor.

        Channel 0 — hidden mask  : 1 = still unknown, 0 = revealed
        Channel 1 — view values  : 0-8 revealed count; 255 = unknown; 16 = mine (post-loss)
        Channel 2 — action mask  : 1 at the most-recently clicked cell
        """
        n = self.n
        act = last_action or self._last_action

        hidden = np.array(
            [
                [1 if self.view[r][c] == Cell.UNKNOWN else 0 for c in range(n)]
                for r in range(n)
            ],
            dtype=np.uint8,
        )
        view_arr = np.array(self.view, dtype=np.uint8)

        action_mask = np.zeros((n, n), dtype=np.uint8)
        if act is not None:
            action_mask[act[0], act[1]] = 1

        return np.stack([hidden, view_arr, action_mask], axis=-1)  # (n, n, 3)

    def render(self) -> str:
        """ASCII view of the current board (for debugging / logging)."""
        symbols = {Cell.UNKNOWN: "·", Cell.MINE: "✸"}
        lines = []
        lines.extend(
            "  ".join(
                symbols.get(self.view[r][c], str(self.view[r][c]))
                for c in range(self.n)
            )
            for r in range(self.n)
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_done(self) -> bool:
        return self.status != GameStatus.ONGOING

    @property
    def won(self) -> bool:
        return self.status == GameStatus.WON

    @property
    def unrevealed_count(self) -> int:
        return sum(
            1
            for r in range(self.n)
            for c in range(self.n)
            if (r, c) not in self.revealed
        )

    # ------------------------------------------------------------------ #
    #  Board construction                                                  #
    # ------------------------------------------------------------------ #

    def _build(self):
        n = self.n
        self.grid = [[0] * n for _ in range(n)]
        self.view = [[Cell.UNKNOWN] * n for _ in range(n)]
        self.revealed = set()
        self.status = GameStatus.ONGOING
        self.step_count = 0
        self._last_action = None

        self._place_mines()
        self._compute_adjacency()

        if self.solve_tiles > 0:
            self._pre_reveal(self.solve_tiles)

    def _place_mines(self):
        indices = self._rng.sample(range(self.n * self.n), self.mines_count)
        for idx in indices:
            r, c = divmod(idx, self.n)
            self.grid[r][c] = Cell.MINE

    def _compute_adjacency(self):
        n = self.n
        for r, c in itertools.product(range(n), range(n)):
            if self.grid[r][c] == Cell.MINE:
                continue
            self.grid[r][c] = sum(
                0 <= r + dr < n
                and 0 <= c + dc < n
                and self.grid[r + dr][c + dc] == Cell.MINE
                for dr, dc in self._DIRS
            )

    def _pre_reveal(self, k: int):
        """
        Reveal k safe tiles, keeping exactly one safe cell always hidden.
        Guarantees the agent always has at least one move remaining at start.
        """
        safe = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.grid[r][c] != Cell.MINE
        ]
        # One permanently hidden safe cell so the game isn't pre-solved
        reserved = self._rng.choice(safe)
        candidates = [cell for cell in safe if cell != reserved]

        for r, c in self._rng.sample(candidates, min(k, len(candidates))):
            self._reveal(r, c)

    # ------------------------------------------------------------------ #
    #  Core mechanics                                                      #
    # ------------------------------------------------------------------ #

    def _flood_fill(self, start_r: int, start_c: int):
        """Iterative BFS — no recursion limit issues on any supported grid size."""
        n = self.n
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) in self.revealed:
                continue
            self._reveal(r, c)
            if self.grid[r][c] == 0:
                for dr, dc in self._DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in self.revealed:
                        queue.append((nr, nc))

    def _reveal(self, r: int, c: int):
        self.revealed.add((r, c))
        self.view[r][c] = self.grid[r][c]

    def _check_win(self) -> bool:
        return self.unrevealed_count == self.mines_count
