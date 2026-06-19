---
title: Minesweeper RL Environment
emoji: 💣
colorFrom: purple
colorTo: orange
sdk: docker
pinned: false
app_port: 9090
base_path: /web
tags:
    - openenv
    - reinforcement-learning
    - minesweeper
    - dqn
---

# Minesweeper RL Environment

An OpenEnv-compatible Minesweeper environment for RL training and evaluation.

## What changed in the current design

This environment now follows a strict explicit-configuration contract:

- every `reset()` call must provide `n`, `mines`, and `solve_tiles`
- the server does **not** manage curriculum progression across episodes
- the training loop is responsible for scheduling `solve_tiles`
- evaluation should typically run with `solve_tiles=0`

That separation keeps the environment reusable and makes train/eval behavior easier to reason about.

## Quick start

### Connect to a running local server

```python
from minesweeper_env import MinesweeperAction, MinesweeperEnv

with MinesweeperEnv(base_url="http://localhost:9090").sync() as env:
    result = env.reset(n=8, mines=10, solve_tiles=3)
    obs = result.observation

    while not result.done:
        action_idx = int(obs.action_mask.argmax())
        row, col = divmod(action_idx, obs.n)
        result = env.step(MinesweeperAction(row=row, col=col))
        obs = result.observation

    print(obs.message)
```

### Start the local server

```bash
uv run server --port 9090
```

## Reset contract

`reset()` requires explicit episode parameters:

```python
result = env.reset(
    n=8,
    mines=10,
    solve_tiles=6,
)
```

### Parameter meanings

- `n` — board side length
- `mines` — number of mines on the board
- `solve_tiles` — number of safe tiles pre-revealed on the first move

## Observation

Each step returns a `MinesweeperObservation` with:

- `board` — nested list representing the visible board
- `n` — active board size
- `mines_count` — mine count for the active episode
- `unrevealed_count` — remaining hidden cells
- `status` — `0=ONGOING`, `1=WON`, `2=LOST`
- `message` — human-readable status message

Convenience views:

- `obs.flat` — normalized flattened board for MLP-style inputs
- `obs.spatial` — 3-channel tensor for CNN-style inputs
- `obs.action_mask` — valid action mask over unrevealed cells

## State metadata

`env.state` returns a `MinesweeperState` for the active episode:

- `episode_id`
- `step_count`
- `n`
- `mines_count`
- `solve_tiles`
- computed properties: `mine_density`, `safe_cells`

## Reward design

The current engine rewards are:

- re-click revealed cell → `-0.1`
- hit mine → `-1.0`
- safe reveal → `new_cells / total_safe`
- win → `10.0`

## Notes for training code

Typical training setup:

- training resets with scheduled `solve_tiles`
- evaluation resets with `solve_tiles=0`
- client code should not assume any default board settings

## Local development

```bash
# in the environment directory
uv run server --port 9090

# or
python -m minesweeper_env.server.app --port 9090
```

## Deployment

This environment is designed for OpenEnv-style deployment and local websocket use. If you package or deploy it remotely, keep the same explicit reset contract so training and evaluation remain reproducible.
