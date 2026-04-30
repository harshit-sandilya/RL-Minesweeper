# server/environment.py

"""
Minesweeper Environment Implementation.

Wraps MinesweeperEngine in the OpenEnv Environment interface.

Curriculum logic:
    solve_tiles starts at whatever reset() receives.
    Every time the agent wins _win_threshold episodes in a row,
    solve_tiles is reduced by _solve_tiles_step (floor: 0).
    At solve_tiles=0 the agent plays unassisted — full difficulty.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from ..models import (
    _DEFAULT_MINES,
    _DEFAULT_N,
    _DEFAULT_SOLVE_TILES,
    MinesweeperAction,
    MinesweeperObservation,
    MinesweeperState,
)
from .game_engine import GameStatus, MinesweeperEngine


class MinesweeperEnvironment(Environment):
    """
    Minesweeper RL environment with built-in solve_tiles curriculum.

    Curriculum behaviour:
        - reset() sets the starting solve_tiles for the run.
        - Each win increments an internal consecutive-win counter.
        - When counter reaches win_threshold, solve_tiles drops by
          solve_tiles_step (minimum 0) and the counter resets.
        - Losses leave the counter unchanged — agent must win to progress.

    Args:
        win_threshold    : consecutive wins needed to advance curriculum (default 3)
        solve_tiles_step : how much to reduce solve_tiles per advancement (default 1)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        win_threshold: int = 3,
        solve_tiles_step: int = 1,
    ):
        self._engine: Optional[MinesweeperEngine] = None

        # Curriculum state — persists across episodes, reset by reset()
        self._win_threshold = win_threshold
        self._solve_tiles_step = solve_tiles_step
        self._consecutive_wins = 0
        self._curriculum_solve_tiles = _DEFAULT_SOLVE_TILES  # managed internally

        self._state = MinesweeperState()

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        n: int = _DEFAULT_N,
        mines: int = _DEFAULT_MINES,
        solve_tiles: Optional[int] = None,
        **kwargs,
    ) -> MinesweeperObservation:
        """
        Start a new episode.

        Args:
            n           : grid side-length (4–10)
            mines       : number of mines
            solve_tiles : override curriculum solve_tiles for this episode.
                          Pass None (default) to let the curriculum manage it.
                          Pass an explicit int to force a specific difficulty
                          and also seed the curriculum counter from that value.
            seed        : RNG seed for reproducibility
            episode_id  : explicit episode ID (auto-generated if omitted)
        """
        # If caller passes an explicit solve_tiles, treat it as a curriculum reset
        if solve_tiles is not None:
            self._curriculum_solve_tiles = solve_tiles
            self._consecutive_wins = 0

        effective_solve_tiles = self._curriculum_solve_tiles

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
                f"({'full difficulty' if effective_solve_tiles == 0 else f'curriculum tier {effective_solve_tiles}'})"
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
        if self._engine is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1

        obs_array, reward, done = self._engine.step(action.row, action.col)

        # ── Curriculum advancement ────────────────────────────────────────
        if done:
            if self._engine.won:
                self._consecutive_wins += 1
                if self._consecutive_wins >= self._win_threshold:
                    self._curriculum_solve_tiles = max(
                        0, self._curriculum_solve_tiles - self._solve_tiles_step
                    )
                    self._consecutive_wins = 0
                    self._state.solve_tiles = self._curriculum_solve_tiles

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
                consecutive_wins=self._consecutive_wins,
                curriculum_solve_tiles=self._curriculum_solve_tiles,
                win_threshold=self._win_threshold,
            ),
        )

    # ------------------------------------------------------------------ #
    #  state property                                                      #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> MinesweeperState:
        """
        Exposes curriculum progress alongside episode metadata.
        mine_density and safe_cells are computed_fields on MinesweeperState.
        """
        return self._state


# ── Helpers ───────────────────────────────────────────────────────────────────


def _status_message(
    status: GameStatus,
    row: int,
    col: int,
    engine: MinesweeperEngine,
    consecutive_wins: int,
    curriculum_solve_tiles: int,
    win_threshold: int,
) -> str:
    if status == GameStatus.WON:
        streak_msg = (
            f" Streak: {consecutive_wins}/{win_threshold}."
            if consecutive_wins > 0
            else ""
        )
        tier_msg = (
            f" Next episode: solve_tiles={curriculum_solve_tiles}."
            if curriculum_solve_tiles >= 0
            else ""
        )
        return f"Won in {engine.step_count} steps!{streak_msg}{tier_msg}"

    if status == GameStatus.LOST:
        return (
            f"Mine at ({row},{col}). "
            f"{engine.step_count} steps taken. "
            f"Win streak: {consecutive_wins}/{win_threshold}."
        )

    return f"({row},{col}) revealed. {engine.unrevealed_count} cells remain."
