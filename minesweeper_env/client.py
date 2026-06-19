# client.py
"""Minesweeper Environment Client."""

from __future__ import annotations

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import MinesweeperAction, MinesweeperObservation, MinesweeperState


class MinesweeperEnv(
    EnvClient[MinesweeperAction, MinesweeperObservation, MinesweeperState]
):
    """
    WebSocket client for the Minesweeper RL environment.

    Each instance is a fully isolated session — spin up N clients
    for N parallel agents with zero shared state.

    The client only knows about `models.py` — it never imports from
    `server/` (that code runs inside the environment service).

    Every episode reset must provide explicit environment parameters:
    `n`, `mines`, and `solve_tiles`.

    ── Sync usage (scripts / notebooks) ─────────────────────────────────
        with MinesweeperEnv(base_url="http://localhost:9090").sync() as env:
            result = env.reset(n=8, mines=10, solve_tiles=3)
            obs = result.observation

            while not result.done:
                action_idx = int(obs.action_mask.argmax())
                row, col = divmod(action_idx, obs.n)
                result = env.step(MinesweeperAction(row=row, col=col))
                obs = result.observation

    ── Async usage (parallel training) ──────────────────────────────────
        async with MinesweeperEnv(base_url="http://localhost:9090") as env:
            result = await env.reset(n=6, mines=6, solve_tiles=0)
            result = await env.step(MinesweeperAction(row=0, col=0))
            state = await env.state          # MinesweeperState for active episode

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
        obs_data = payload["observation"]
        board = obs_data["board"]

        observation = MinesweeperObservation(
            done=payload["done"],
            reward=payload.get("reward"),
            board=board,
            n=int(obs_data["n"]),
            mines_count=int(obs_data["mines_count"]),
            unrevealed_count=int(obs_data["unrevealed_count"]),
            status=int(obs_data["status"]),
            message=obs_data["message"],
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload["done"],
        )

    # ------------------------------------------------------------------ #
    #  Required: deserialise server state → MinesweeperState              #
    # ------------------------------------------------------------------ #

    def _parse_state(self, payload: Dict) -> MinesweeperState:
        """
        Raw WebSocket payload → MinesweeperState.

        Carries episode metadata (`n`, `mines_count`, `solve_tiles`)
        in addition to the base `episode_id` and `step_count`.
        `mine_density` and `safe_cells` are computed_fields — not sent over
        the wire, derived automatically from `n` and `mines_count`.
        """
        return MinesweeperState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload["step_count"]),
            n=int(payload["n"]),
            mines_count=int(payload["mines_count"]),
            solve_tiles=int(payload["solve_tiles"]),
        )
