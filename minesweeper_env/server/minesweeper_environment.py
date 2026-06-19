# server/environment.py

"""
Minesweeper Environment Implementation.

Wraps MinesweeperEngine in the OpenEnv Environment interface.

The environment itself is stateless with respect to curriculum progression:
`solve_tiles` for each episode is supplied explicitly by the caller on reset().
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from ..models import MinesweeperAction, MinesweeperObservation, MinesweeperState
from .game_engine import GameStatus, MinesweeperEngine


class MinesweeperEnvironment(Environment):
    """
    Minesweeper RL environment.

    The server applies whatever `(n, mines, solve_tiles)` the caller provides
    on `reset()` and does not mutate episode difficulty across resets.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._engine: Optional[MinesweeperEngine] = None
        self._state: Optional[MinesweeperState] = None

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> MinesweeperObservation:
        """
        Start a new episode.

        Args:
            n           : grid side-length (4–10), required in kwargs
            mines       : number of mines, required in kwargs
            solve_tiles : number of safe tiles pre-revealed for this episode,
                          required in kwargs and passed explicitly by the caller
            seed        : RNG seed for reproducibility
            episode_id  : explicit episode ID (auto-generated if omitted)
        """
        try:
            n = int(kwargs["n"])
            mines = int(kwargs["mines"])
            solve_tiles = int(kwargs["solve_tiles"])
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"Missing required reset parameter: {missing}") from exc

        effective_solve_tiles = solve_tiles

        self._engine = MinesweeperEngine(
            n=n,
            mines=mines,
            solve_tiles=effective_solve_tiles,
            seed=seed,
        )

        self._state = MinesweeperState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            n=n,
            mines_count=mines,
            solve_tiles=effective_solve_tiles,
        )

        obs_array = self._engine.observe()

        return MinesweeperObservation(
            done=False,
            reward=0.0,
            board=obs_array.tolist(),
            n=n,
            mines_count=mines,
            unrevealed_count=self._engine.unrevealed_count,
            status=int(GameStatus.ONGOING),
            message=(
                f"New {n}×{n} board | {mines} mines | "
                f"solve_tiles={effective_solve_tiles} "
                f"({'no pre-revealed hints' if effective_solve_tiles == 0 else f'{effective_solve_tiles} pre-revealed safe tiles'})"
            ),
        )

    # ------------------------------------------------------------------ #
    #  step                                                                #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: MinesweeperAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> MinesweeperObservation:
        """
        Uncover the cell at (action.row, action.col).

        Reward structure (from engine):
            1.0                      — won
            new_cells / total_safe   — safe reveal, proportional progress
            0.0                      — mine hit (done) or re-click
        """
        if self._engine is None or self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1

        obs_array, reward, done = self._engine.step(action.row, action.col)

        status = self._engine.status

        return MinesweeperObservation(
            done=done,
            reward=reward,
            board=obs_array.tolist(),
            n=self._engine.n,
            mines_count=self._engine.mines_count,
            unrevealed_count=self._engine.unrevealed_count,
            status=int(status),
            message=_status_message(
                status=status,
                row=action.row,
                col=action.col,
                engine=self._engine,
                solve_tiles=self._state.solve_tiles,
            ),
        )

    # ------------------------------------------------------------------ #
    #  state property                                                      #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> MinesweeperState:
        """
        Exposes episode metadata for the active board.
        `mine_density` and `safe_cells` are computed_fields on `MinesweeperState`.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._state


# ── Helpers ───────────────────────────────────────────────────────────────────


def _status_message(
    status: GameStatus,
    row: int,
    col: int,
    engine: MinesweeperEngine,
    solve_tiles: int,
) -> str:
    if status == GameStatus.WON:
        return f"Won in {engine.step_count} steps! solve_tiles={solve_tiles}."

    if status == GameStatus.LOST:
        return f"Mine at ({row},{col}). {engine.step_count} steps taken."

    return f"({row},{col}) revealed. {engine.unrevealed_count} cells remain."
