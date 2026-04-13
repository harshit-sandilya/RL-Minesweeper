"""
Smoke test for the Minesweeper environment.
Requires the server running at localhost:8000.
"""

from minesweeper_env.client import MinesweeperEnv
from minesweeper_env.models import MinesweeperAction

URL = "http://localhost:8000"

with MinesweeperEnv(base_url=URL).sync() as env:
    result = env.reset(n=8, mines=10, solve_tiles=3)
    obs = result.observation

    print(
        f"Board {obs.n}×{obs.n} | mines={obs.mines_count} | unrevealed={obs.unrevealed_count}"
    )
    print(f"Status: {obs.status} | Message: {obs.message}")

    step = 0
    while not result.done:
        action_idx = int(obs.action_mask.argmax())
        row, col = divmod(action_idx, obs.n)

        result = env.step(MinesweeperAction(row=row, col=col))
        obs = result.observation
        step += 1

        print(
            f"Step {step:3d} | action=({row},{col}) | reward={result.reward:.4f} | done={result.done}"
        )

    outcome = "WON 🎉" if obs.status == 1 else "LOST 💥"
    print(f"\n{outcome} after {step} steps | final reward={result.reward}")
