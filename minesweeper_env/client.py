# client.py
"""Minesweeper Environment Client."""

from __future__ import annotations

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (_DEFAULT_MINES, _DEFAULT_N, _DEFAULT_SOLVE_TILES,
                     MinesweeperAction, MinesweeperObservation,
                     MinesweeperState)


class MinesweeperEnv(
    EnvClient[MinesweeperAction, MinesweeperObservation, MinesweeperState]
):
    """
    WebSocket client for the Minesweeper RL environment.

    Each instance is a fully isolated session — spin up N clients
    for N parallel agents with zero shared state.

    The client only knows about models.py — it never imports from
    server/ (that code runs inside the Docker container).

    ── Sync usage (scripts / notebooks) ─────────────────────────────────
        with MinesweeperEnv(base_url="http://localhost:8000").sync as env:
            result = env.reset(n=8, mines=10, solve_tiles=3)
            obs    = result.observation          # MinesweeperObservation

            while not result.done:
                # DQN example — pick highest Q-value unrevealed cell
                import numpy as np
                action_idx = obs.action_mask.argmax()   # simplest valid policy
                row, col   = divmod(int(action_idx), obs.n)
                result     = env.step(MinesweeperAction(row=row, col=col))

    ── Async usage (parallel training) ──────────────────────────────────
        async with MinesweeperEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(n=6, mines=6)
            result = await env.step(MinesweeperAction(row=0, col=0))
            state  = await env.state          # MinesweeperState (curriculum info)

    ── Docker ────────────────────────────────────────────────────────────
        env = MinesweeperEnv.from_docker_image("minesweeper-env:latest")
        try:
            result = env.sync.reset(n=8, mines=10, solve_tiles=5)
        finally:
            env.close()

    ── HuggingFace Space ─────────────────────────────────────────────────
        env = MinesweeperEnv(base_url="https://<user>-minesweeper-env.hf.space")
    """

    # ------------------------------------------------------------------ #
    #  Required: serialise action → wire dict                             #
    # ------------------------------------------------------------------ #

    def _step_payload(self, action: MinesweeperAction) -> Dict:
        """
        MinesweeperAction → dict sent over WebSocket.
        Only include the fields the server's step() reads.
        """
        return {
            "row": action.row,
            "col": action.col,
        }

    # ------------------------------------------------------------------ #
    #  Required: deserialise server response → StepResult                 #
    # ------------------------------------------------------------------ #

    def _parse_result(self, payload: Dict) -> StepResult[MinesweeperObservation]:
        """
        Raw WebSocket payload → StepResult[MinesweeperObservation].

        Payload shape (from server environment.py):
            {
                "done":    bool,
                "reward":  float | null,
                "observation": {
                    "board":            [[int, ...], ...],   # (n, n)
                    "n":                int,
                    "mines_count":      int,
                    "unrevealed_count": int,
                    "status":           int,                 # 0/1/2
                    "message":          str,
                    "done":             bool,
                    "reward":           float | null,
                }
            }
        """
        obs_data = payload.get("observation", {})

        observation = MinesweeperObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            board=obs_data.get("board", []),
            n=obs_data.get("n", _DEFAULT_N),
            mines_count=obs_data.get("mines_count", _DEFAULT_MINES),
            unrevealed_count=obs_data.get("unrevealed_count", 0),
            status=int(obs_data.get("status", 0)),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ------------------------------------------------------------------ #
    #  Required: deserialise server state → MinesweeperState              #
    # ------------------------------------------------------------------ #

    def _parse_state(self, payload: Dict) -> MinesweeperState:
        """
        Raw WebSocket payload → MinesweeperState.

        Carries curriculum metadata (solve_tiles, mine_density, safe_cells)
        in addition to the base episode_id and step_count.
        mine_density and safe_cells are computed_fields — not sent over wire,
        derived automatically from n and mines_count after construction.
        """
        return MinesweeperState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            n=payload.get("n", _DEFAULT_N),
            mines_count=payload.get("mines_count", _DEFAULT_MINES),
            solve_tiles=payload.get("solve_tiles", _DEFAULT_SOLVE_TILES),
        )
