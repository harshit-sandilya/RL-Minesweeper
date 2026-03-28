# server/environment.py

"""
Minesweeper Environment Implementation.

Wraps MinesweeperEngine in the OpenEnv Environment interface.
Each instance is fully isolated — safe for concurrent multi-agent sessions.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from .game_engine import MinesweeperEngine, GameStatus
from ..models import (
    MinesweeperAction,
    MinesweeperObservation,
    MinesweeperState,
    _DEFAULT_N,
    _DEFAULT_MINES,
    _DEFAULT_SOLVE_TILES,
)


class MinesweeperEnvironment(Environment):
    """
    Minesweeper environment for training and evaluating policies.

    Grid size (n) and mine count are configurable per-episode via reset(),
    so a single environment instance can serve variable-difficulty curricula.

    Concurrent sessions are supported — each WebSocket client gets its own
    MinesweeperEnvironment instance with fully isolated engine state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._engine: Optional[MinesweeperEngine] = None
        self._state = MinesweeperState(
            episode_id=str(uuid4()),
            step_count=0,
        )

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        n: int = _DEFAULT_N,
        mines: int = _DEFAULT_MINES,
        solve_tiles: int = _DEFAULT_SOLVE_TILES,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> MinesweeperObservation:
        """
        Start a new episode.

        Args:
            n           : grid side-length (4–10)
            mines       : number of mines
            solve_tiles : safe tiles pre-revealed (curriculum difficulty knob)
            seed        : RNG seed for reproducibility
            episode_id  : explicit episode identifier (auto-generated if omitted)
        """
        self._engine = MinesweeperEngine(
            n=n,
            mines=mines,
            solve_tiles=solve_tiles,
            seed=seed,
        )

        self._state = MinesweeperState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            n=n,
            mines=mines,
        )

        obs_array = self._engine.observe()

        return MinesweeperObservation(
            done=False,
            reward=0.0,
            board=obs_array.tolist(),  # (n, n, 3) → nested list for JSON
            n=n,
            mines_count=mines,
            unrevealed_count=self._engine.unrevealed_count,
            status=GameStatus.ONGOING,
            message=f"New {n}×{n} board — {mines} mines. Good luck!",
        )

    # ------------------------------------------------------------------ #
    #  step                                                                #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: MinesweeperAction,
        **kwargs,
    ) -> MinesweeperObservation:
        """
        Uncover the cell at (action.row, action.col).

        Reward structure (delegated from the engine):
            +1.0  — won (all non-mine cells revealed)
            -1.0  — lost (hit a mine)
            -0.5  — re-clicked an already-revealed cell
             0.0  — valid reveal, game ongoing
        """
        if self._engine is None:
            raise RuntimeError("Environment not initialised — call reset() first.")

        self._state.step_count += 1

        obs_array, reward, done = self._engine.step(action.row, action.col)

        status = self._engine.status
        message = _status_message(status, action.row, action.col, self._engine)

        return MinesweeperObservation(
            done=done,
            reward=reward,
            board=obs_array.tolist(),
            n=self._engine.n,
            mines_count=self._engine.mines_count,
            unrevealed_count=self._engine.unrevealed_count,
            status=status,
            message=message,
        )

    # ------------------------------------------------------------------ #
    #  state property                                                      #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> MinesweeperState:
        return self._state


# ── Helpers ──────────────────────────────────────────────────────────────────


def _status_message(
    status: GameStatus,
    row: int,
    col: int,
    engine: MinesweeperEngine,
) -> str:
    if status == GameStatus.WON:
        return f"You won in {engine.step_count} steps!"
    if status == GameStatus.LOST:
        return f"Hit a mine at ({row},{col}). Game over."
    return f"Revealed ({row},{col}). {engine.unrevealed_count} cells remaining."
