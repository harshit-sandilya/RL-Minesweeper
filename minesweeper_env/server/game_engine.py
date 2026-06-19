from __future__ import annotations

import itertools
import random
from collections import deque
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class Cell(IntEnum):
    UNKNOWN = 9
    MINE = 10


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
        solve_tiles : safe tiles pre-revealed on the first move of the episode
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
        self.revealed: set[Tuple[int, int]] = set()
        self.status: GameStatus = GameStatus.ONGOING
        self.step_count: int = 0
        self._mines_initialized = False

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

        Reward design:
            Re-click on revealed cell  →  -0.1
            Mine hit (loss)            →  -1.0
            Safe reveal                →  new_cells / total_safe
            Win                        →  10.0

        Returns:
            observation : (n, n) uint8 array
            reward      : float in [0.0, 1.0]
            done        : bool
        """
        if self.status != GameStatus.ONGOING:
            raise RuntimeError("Game is over — call reset() first.")
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise ValueError(f"({row},{col}) out of bounds for {self.n}×{self.n} grid.")

        self._last_action = (row, col)

        # ── Re-click: wasted move ──────────────────────────────────────────────
        if (row, col) in self.revealed:
            return self.observe(), -0.1, False
            # return self.observe(), 0, False

        self.step_count += 1
        prev_revealed = len(self.revealed)
        total_safe = self.n**2 - self.mines_count

        # ── First click safety & Board Generation ──────────────────────────────
        is_first_click = not self._mines_initialized
        if is_first_click:
            self._place_mines_excluding(row, col)
            self._compute_adjacency()
            self._mines_initialized = True

        # ── Mine hit: episode ends, zero reward ───────────────────────────────
        if self.grid[row][col] == Cell.MINE:
            self.revealed.add((row, col))
            self.status = GameStatus.LOST
            return self.observe(), -1.0, True
            # return self.observe(), 0.0, True

        # ── Safe reveal: flood-fill, reward proportional to new cells ─────────
        self._flood_fill(row, col)

        # ── Pre-revealed hint tiles: apply only on the first click ─────────────
        if is_first_click and self.solve_tiles > 0:
            self._pre_reveal(self.solve_tiles)

        newly_revealed = len(self.revealed) - prev_revealed
        progress_reward = newly_revealed / total_safe

        # ── Win ───────────────────────────────────────────────────────────────
        if self._check_win():
            self.status = GameStatus.WON
            return self.observe(), 10.0, True

        return self.observe(), progress_reward, False
        # return self.observe(), 0.2, False

    def observe(self) -> np.ndarray:
        """
        (n, n) uint8 observation tensor.

        Values:
            0–8  : revealed safe cell (adjacency count)
            9    : unrevealed — mine cells also render as 9 (agent can't see mines)

        Usage by agent type:
            MLP / DQN  : obs.flatten().astype(np.float32) / 9.0
            CNN        : obs[np.newaxis]  →  (1, n, n) for PyTorch
            GRPO/LLM   : obs.tolist()    →  JSON-serialisable nested list
        """
        obs = np.full((self.n, self.n), Cell.UNKNOWN, dtype=np.uint8)
        for r, c in self.revealed:
            v = self.grid[r][c]
            if v != Cell.MINE:
                obs[r, c] = v
        return obs

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
        return self.n * self.n - len(self.revealed)

    # ------------------------------------------------------------------ #
    #  Board construction                                                  #
    # ------------------------------------------------------------------ #

    def _build(self):
        n = self.n
        self.grid = [[0] * n for _ in range(n)]
        self.revealed = set()
        self.status = GameStatus.ONGOING
        self.step_count = 0
        self._mines_initialized = False

    def _place_mines(self):
        """Randomly place mines on the grid using the configured RNG."""
        indices = self._rng.sample(range(self.n * self.n), self.mines_count)
        for idx in indices:
            r, c = divmod(idx, self.n)
            self.grid[r][c] = Cell.MINE

    def _place_mines_excluding(self, start_r: int, start_c: int):
        exclude_cells = {(start_r, start_c)}
        for dr, dc in self._DIRS:
            r, c = start_r + dr, start_c + dc
            if 0 <= r < self.n and 0 <= c < self.n:
                exclude_cells.add((r, c))

        available_cells = [(r, c) for r in range(self.n) for c in range(self.n)]
        safe_candidates = [
            cell for cell in available_cells if cell not in exclude_cells
        ]

        if len(safe_candidates) < self.mines_count:
            exclude_cells = {(start_r, start_c)}
            safe_candidates = [
                cell for cell in available_cells if cell not in exclude_cells
            ]

        mine_cells = self._rng.sample(safe_candidates, self.mines_count)
        for r, c in mine_cells:
            self.grid[r][c] = Cell.MINE

    def _compute_adjacency(self):
        """Compute adjacency counts (number of neighboring mines) for all safe cells."""
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

        Args:
            k (int): Number of safe tiles to reveal
        """
        safe_unrevealed = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.grid[r][c] != Cell.MINE and (r, c) not in self.revealed
        ]

        if not safe_unrevealed:
            return

        reserved = self._rng.choice(safe_unrevealed)
        candidates = [cell for cell in safe_unrevealed if cell != reserved]

        for r, c in self._rng.sample(candidates, min(k, len(candidates))):
            self.revealed.add((r, c))

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
            self.revealed.add((r, c))
            if self.grid[r][c] == 0:
                for dr, dc in self._DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in self.revealed:
                        queue.append((nr, nc))

    def _check_win(self) -> bool:
        return self.unrevealed_count == self.mines_count
