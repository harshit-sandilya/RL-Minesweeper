# Experiments

This file is the running experiment tracker for improving the Minesweeper agent.

## How to use this file

For each meaningful run, add:

- experiment ID
- date
- algorithm
- config or overrides
- board size and mine count
- training curriculum schedule
- evaluation setup
- checkpoint selected
- main outcome
- suspected bottleneck / failure mode
- next change to try
- log file paths

## Summary table

| ID     | Status                         | Algorithm | Board             | Curriculum                                                                           | Eval mode       | Result summary                                                                                                                                                                               | Logs                                                  |
| ------ | ------------------------------ | --------- | ----------------- | ------------------------------------------------------------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| EXP-12 | Completed / baseline diagnosis | DQN       | `8x8`, `10` mines | Started with `solve_tiles=6`; old setup mixed assisted training with unassisted eval | `solve_tiles=0` | Checkpoint metadata reported train win rate `0.410`, but `logs/eval_output_12.log` showed eval win rate only `0.2%`. This pointed to a curriculum/data-flow mismatch between train and eval. | `logs/train_output_12.log`, `logs/eval_output_12.log` |
| EXP-13 | Planned                        | DQN       | `8x8`, `10` mines | Hold until `20k` steps, then linearly decay `solve_tiles` to `0` by `80k`            | `solve_tiles=0` | Re-run after explicit reset/data-flow refactor. Measure train/eval gap and full-difficulty generalization.                                                                                   | _to be added_                                         |

---

## Detailed notes

### EXP-12

- Date: existing logged run
- Algorithm: DQN
- Config: `configs/dqn.yml` from the pre-refactor setup
- Train setup:
    - board: `8x8`
    - mines: `10`
    - starting `solve_tiles`: `6`
    - environment still owned part of the difficulty behavior
- Eval setup:
    - unassisted evaluation on full difficulty
- Best checkpoint:
    - `checkpoints/dqn_8x8_baseline_ep100000.pt`
- Outcome:
    - checkpoint metadata showed training win rate around `0.410`
    - `logs/eval_output_12.log` showed eval win rate only `0.2%`
    - average eval reward was negative, despite much stronger training numbers
- Interpretation:
    - the agent likely overfit to assisted starts and/or inconsistent difficulty handling
    - train/eval data flow was not aligned tightly enough
- Logs:
    - `logs/train_output_12.log`
    - `logs/eval_output_12.log`
- Follow-up:
    - move curriculum ownership into training
    - make `n`, `mines`, and `solve_tiles` explicit on every reset
    - force eval to `solve_tiles=0`

### EXP-13

- Date: planned next run
- Algorithm: DQN
- Config: `configs/dqn.yml`
- Train setup:
    - board: `8x8`
    - mines: `10`
    - `solve_tiles=6`
    - `curriculum_hold_steps=20000`
    - `curriculum_end_steps=80000`
    - linear step-based decay implemented in `rl/agents/dqn_agent.py`
- Eval setup:
    - `solve_tiles=0`
    - same board/mines as training
- Best checkpoint:
    - _to be filled after run_
- Goal:
    - reduce the train/eval mismatch seen in EXP-12
    - test whether explicit reset parameters and training-owned curriculum improve full-difficulty generalization
- Metrics to watch:
    - train win rate
    - eval win rate
    - revealed ratio
    - remaining safe cells
    - best checkpoint by eval performance, not training reward alone
- Logs:
    - _to be added_
- Follow-up:
    - compare against Double DQN if DQN still stalls or overestimates

---

## Reusable template

```md
### EXP-XX

- Date:
- Algorithm:
- Config:
- Train setup:
- Eval setup:
- Best checkpoint:
- Outcome:
- Interpretation:
- Logs:
- Follow-up:
```
