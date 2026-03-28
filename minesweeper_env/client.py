"""Minesweeper Environment Client."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import MinesweeperAction, MinesweeperObservation, MinesweeperState
from .server.game_engine import GameStatus


class MinesweeperEnv(
    EnvClient[MinesweeperAction, MinesweeperObservation, MinesweeperState]
):
    """
    Client for the Minesweeper environment.

    Maintains a persistent WebSocket connection to the server.
    Each instance gets its own isolated session — spin up N clients
    for N parallel agents with zero shared state.

    Example (context manager):
        >>> with MinesweeperEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(n=8, mines=10)
        ...     obs    = result.observation
        ...     board  = obs.board_array   # (8, 8, 3) numpy array
        ...
        ...     while not result.done:
        ...         action = MinesweeperAction(row=3, col=4)
        ...         result = env.step(action)

    Example (Docker):
        >>> env = MinesweeperEnv.from_docker_image("minesweeper-env:latest")
        >>> try:
        ...     result = env.reset(n=6, mines=6, solve_tiles=5)
        ...     result = env.step(MinesweeperAction(row=0, col=0))
        ... finally:
        ...     env.close()
    """

    # ------------------------------------------------------------------ #
    #  EnvClient abstract methods                                          #
    # ------------------------------------------------------------------ #

    def _step_payload(self, action: MinesweeperAction) -> Dict:
        """Serialise MinesweeperAction → wire dict."""
        return {
            "row": action.row,
            "col": action.col,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MinesweeperObservation]:
        obs_data = payload.get("observation", {})

        raw_status = obs_data.get("status", 0)
        status = GameStatus(int(raw_status))

        observation = MinesweeperObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            board=obs_data.get("board", []),
            n=obs_data.get("n", 8),
            mines_count=obs_data.get("mines_count", 0),
            unrevealed_count=obs_data.get("unrevealed_count", 0),
            status=status,
            message=obs_data.get("message", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MinesweeperState:
        """Deserialise server response → MinesweeperState."""
        return MinesweeperState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            n=payload.get("n", 8),
            mines=payload.get("mines", 10),
        )
