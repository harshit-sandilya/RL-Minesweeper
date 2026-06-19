# RL Minesweeper

A reinforcement-learning project for training DQN-style agents on a Minesweeper environment served through Meta's OpenEnv pattern.

## Repository layout

- `minesweeper_env/` — OpenEnv Minesweeper service and typed client
- `rl/` — agents, algorithms, replay buffer, logging, and networks
- `configs/` — training configs
- `logs/` — captured train/eval console outputs
- `train_dqn.py` — training entry point
- `evaluate_dqn.py` — checkpoint evaluation entry point
- `test_env.py` — quick environment smoke test

## Current environment contract

Episode parameters are now passed explicitly on every reset:

```python
result = env.reset(n=8, mines=10, solve_tiles=6)
```

Notes:

- `n`, `mines`, and `solve_tiles` are required per episode.
- The environment no longer owns curriculum progression.
- Training owns curriculum scheduling.
- Evaluation always runs at `solve_tiles=0`.

## Training curriculum policy

Current training schedule in `rl/agents/dqn_agent.py`:

- hold the configured `solve_tiles` until `curriculum_hold_steps`
- linearly decay to `0` by `curriculum_end_steps`
- keep `0` for the rest of training

Example current config from `configs/dqn.yml`:

- board: `8x8`
- mines: `10`
- starting `solve_tiles`: `6`
- hold: `20_000` steps
- decay end: `80_000` steps

## Local run commands

### Environment server

From `minesweeper_env/`:

```bash
conda activate rl
uv run server --port 9090
```

### Training

From repo root:

```bash
conda activate rl
python train_dqn.py --config configs/dqn.yml
```

### Quick smoke checks

```bash
conda activate rl
python test_env.py
python train_dqn.py --config configs/dqn.yml
```

For smoke checks, stop training as soon as it starts producing episode logs without errors.

## Experiment tracker

The detailed running experiment log lives in `docs/EXPERIMENTS.md`.

Current tracked entries:

- `EXP-12` — baseline diagnosis from `logs/train_output_12.log` and `logs/eval_output_12.log`
- `EXP-13` — planned rerun after explicit reset + training-owned curriculum refactor

## Recent codebase changes relevant to experiments

- removed shared hardcoded env constants from the runtime path
- made `n`, `mines`, and `solve_tiles` explicit reset inputs
- moved curriculum ownership fully into training code
- fixed eval to always use full difficulty (`solve_tiles=0`)
- aligned local smoke-test flow with the `9090` environment server setup
