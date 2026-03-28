# test_env.py
"""
Contract test for the Minesweeper environment.

Only uses the public client API — no internal imports.
This is the pattern any user of this env would write.
"""


import contextlib
import random
from minesweeper_env.client import MinesweeperEnv
from minesweeper_env.models import MinesweeperAction

URL = "http://localhost:8000"


# ── Minimal policies (obs → action) ─────────────────────────────────────────


def random_policy(obs) -> MinesweeperAction:
    """Uniformly random unrevealed cell."""
    n = obs.n
    hidden = obs.board_array[:, :, 0]  # 1 = unrevealed
    candidates = [(r, c) for r in range(n) for c in range(n) if hidden[r][c] == 1]
    r, c = random.choice(candidates) if candidates else (0, 0)
    return MinesweeperAction(row=r, col=c)


def corner_first_policy(obs) -> MinesweeperAction:
    """Click corners then edges, then random — simple deterministic heuristic."""
    n = obs.n
    hidden = obs.board_array[:, :, 0]
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    for r, c in corners:
        if hidden[r][c] == 1:
            return MinesweeperAction(row=r, col=c)
    # Fall back to random
    return random_policy(obs)


# ── Test helpers ─────────────────────────────────────────────────────────────


def run_episode(env, policy, n=8, mines=10, seed=None, solve_tiles=0):
    """Run one episode with a policy. Returns (won, steps, final_reward)."""
    result = env.reset(n=n, mines=mines, seed=seed, solve_tiles=solve_tiles)
    steps = 0
    while not result.done:
        action = policy(result.observation)
        result = env.step(action)
        steps += 1
    won = result.reward == 1.0
    return won, steps, result.reward


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_basic_contract():
    print("── Basic observation contract ──────────────────")
    with MinesweeperEnv(base_url=URL).sync() as env:
        result = env.reset(n=6, mines=5)
        obs = result.observation

        # Shape contract
        assert obs.board_array.shape == (
            6,
            6,
            3,
        ), f"Expected (6,6,3), got {obs.board_array.shape}"

        # Field contract
        assert obs.n == 6
        assert obs.mines_count == 5
        assert obs.unrevealed_count == 36  # nothing revealed yet
        assert not result.done
        assert isinstance(obs.message, str) and len(obs.message) > 0

        print(f"  shape={obs.board_array.shape}  unrevealed={obs.unrevealed_count}")
        print(f"  message='{obs.message}'")
    print("PASSED\n")


def test_episode_terminates():
    print("── Episode terminates with valid reward ────────")
    with MinesweeperEnv(base_url=URL).sync() as env:
        won, steps, reward = run_episode(env, random_policy, n=4, mines=2, seed=7)
        assert reward in (1.0, -1.0), f"Terminal reward must be ±1.0, got {reward}"
        assert steps > 0
        print(f"  won={won}  steps={steps}  reward={reward}")
    print("PASSED\n")


def test_re_click_penalty():
    print("── Re-click returns -0.5 and keeps game alive ──")
    with MinesweeperEnv(base_url=URL).sync() as env:
        for seed in range(50):
            env.reset(n=6, mines=5, seed=seed)
            r1 = env.step(MinesweeperAction(row=0, col=0))
            if not r1.done:
                break
        else:
            raise RuntimeError(
                "Couldn't find a seed where (0,0) is safe after 50 tries"
            )
        r2 = env.step(MinesweeperAction(row=0, col=0))
        assert r2.reward == -0.5, f"Expected re-click penalty -0.5, got {r2.reward}"
        assert not r2.done, "Re-click should not end the episode"
        print(f"  seed={seed}  r1.reward={r1.reward}  r2.reward={r2.reward}")
    print("PASSED\n")


def test_solve_tiles_pre_reveals():
    print("── solve_tiles pre-reveals safe cells ─────────")
    with MinesweeperEnv(base_url=URL).sync() as env:
        hard = env.reset(n=8, mines=10, solve_tiles=0)
        easy = env.reset(n=8, mines=10, solve_tiles=20)
        revealed_hard = hard.observation.n**2 - hard.observation.unrevealed_count
        revealed_easy = easy.observation.n**2 - easy.observation.unrevealed_count
        assert revealed_hard == 0, f"solve_tiles=0  → got {revealed_hard}"
        assert revealed_easy >= 1, f"solve_tiles=20 → got {revealed_easy}"
        assert revealed_easy > revealed_hard
        print(f"  solve_tiles=0  → {revealed_hard} pre-revealed")
        print(f"  solve_tiles=20 → {revealed_easy} pre-revealed")
    print("PASSED\n")


def test_state_tracks_steps():
    print("── State tracks step count correctly ───────────")
    with MinesweeperEnv(base_url=URL).sync() as env:
        # Find a seed where neither (0,0) nor (1,1) is a mine
        for seed in range(100):
            env.reset(n=8, mines=10, seed=seed)
            r1 = env.step(MinesweeperAction(row=0, col=0))
            if r1.done:
                continue
            r2 = env.step(MinesweeperAction(row=1, col=1))
            if not r2.done:
                break
        else:
            raise RuntimeError("Couldn't find a safe seed after 100 tries")

        state = env.state()
        assert state.step_count == 2, f"Expected 2 steps, got {state.step_count}"
        assert state.episode_id is not None
        print(
            f"  seed={seed}  episode_id={str(state.episode_id)[:8]}…  step_count={state.step_count}"
        )
    print("PASSED\n")


def test_multi_session_isolation():
    print("── Multiple sessions are isolated ──────────────")
    envs = [MinesweeperEnv(base_url=URL).sync() for _ in range(4)]
    try:
        _extracted_from_test_multi_session_isolation_5(envs)
    finally:
        for e in envs:
            with contextlib.suppress(Exception):
                e.__exit__(None, None, None)
    print("PASSED\n")


# TODO Rename this here and in `test_multi_session_isolation`
def _extracted_from_test_multi_session_isolation_5(envs):
    for e in envs:
        e.__enter__()

    # Reset with different seeds
    for i, e in enumerate(envs):
        e.reset(n=8, mines=10, seed=i)

    # Step each env at the same cell — different mine placements
    # will produce different revealed neighborhoods
    results = [e.step(MinesweeperAction(row=3, col=3)) for e in envs]

    # Boards must not ALL be equal — different seeds → different mine layouts
    boards = [r.observation.board_array for r in results]
    all_same = all((boards[0] == b).all() for b in boards[1:])
    assert (
        not all_same
    ), "Different seeds should produce diverging boards after stepping"

    # Also verify episode IDs are distinct (each session is independent)
    states = [e.state() for e in envs]
    episode_ids = [s.episode_id for s in states]
    assert len(set(episode_ids)) == 4, "All 4 sessions must have unique episode IDs"

    print("  4 isolated sessions confirmed")
    print(f"  episode_ids: {[eid[:8] for eid in episode_ids]}")


def test_policy_comparison():
    print("── Policy comparison (5 episodes each) ─────────")
    N_EPISODES = 5
    for name, policy in [
        ("random", random_policy),
        ("corner_first", corner_first_policy),
    ]:
        wins = 0
        with MinesweeperEnv(base_url=URL).sync() as env:
            for seed in range(N_EPISODES):
                won, steps, _ = run_episode(env, policy, n=6, mines=4, seed=seed)
                wins += int(won)
        print(f"  {name:15s} — {wins}/{N_EPISODES} wins")
    print("PASSED\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_basic_contract()
    test_episode_terminates()
    test_re_click_penalty()
    test_solve_tiles_pre_reveals()
    test_state_tracks_steps()
    test_multi_session_isolation()
    test_policy_comparison()
    print("All tests passed ✓")
