# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Minesweeper Env Environment.

The minesweeper_env environment is a simple test environment that echoes back messages.
"""

from typing import List
from openenv.core.env_server import Action, Observation, State
import numpy as np


# ── Defaults ────────────────────────────────────────────────────────────────
_DEFAULT_N = 8
_DEFAULT_MINES = 10
_DEFAULT_SOLVE_TILES = 0


class MinesweeperAction(Action):
    row: int
    col: int


class MinesweeperObservation(Observation):
    # done: bool and reward: Optional[float] inherited from Observation
    board: List  # (n, n, 3) as nested list
    n: int
    mines_count: int
    unrevealed_count: int
    status: int  # GameStatus enum value
    message: str

    @property
    def board_array(self) -> np.ndarray:
        """Return board as a (n, n, 3) uint8 numpy array."""
        return np.array(self.board, dtype=np.uint8)

    @property
    def hidden_mask(self) -> np.ndarray:
        """(n, n) — 1 where cell is unrevealed."""
        return self.board_array[:, :, 0]

    @property
    def value_map(self) -> np.ndarray:
        """(n, n) — revealed counts (0-8), 255=unknown, 16=mine."""
        return self.board_array[:, :, 1]

    @property
    def action_channel(self) -> np.ndarray:
        """(n, n) — 1 at the last-clicked cell."""
        return self.board_array[:, :, 2]


class MinesweeperState(State):
    # episode_id: Optional[str] and step_count: int inherited from State
    n: int = _DEFAULT_N
    mines: int = _DEFAULT_MINES
