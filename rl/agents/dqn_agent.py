"""
rl/agents/dqn_agent.py

DQN agent for Minesweeper.

This file is intentionally problem-coupled — it knows about:
    MinesweeperEnv, MinesweeperObservation, MinesweeperConfig,
    obs.spatial, obs.action_mask, action index → MinesweeperAction

Everything below this file in the stack is problem-agnostic:
    DQN algorithm, ReplayBuffer, Logger, Config, MinesweeperCNN

Responsibilities
----------------
    ✓  Build and wire all components from a MinesweeperConfig
    ✓  Run the training loop (episode → select action → step → store → update)
    ✓  ε-greedy action selection with Minesweeper action masking
    ✓  Convert obs.spatial → tensor for network input
    ✓  Convert network output index → MinesweeperAction for env
    ✓  Call algo.sync_target() every target_update_freq steps
    ✓  Log all metrics via logger
    ✓  Save / load checkpoints

Training loop (one episode)
----------------------------
    obs = env.reset()
    while not done:
        ε-greedy select action index from valid cells (obs.action_mask)
        step env → next_obs, reward, done, info
        store (obs.spatial, action, reward, next_obs.spatial,
               done, next_obs.action_mask) in replay buffer
        if buffer ready:
            batch = buffer.sample()
            result = algo.update(batch)
            if step % target_update_freq == 0: algo.sync_target()
            log update metrics
        obs = next_obs
    log episode metrics
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import from_numpy

from minesweeper_env import MinesweeperAction
from rl.algorithms.dqn import DQN
from rl.common.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from rl.common.config import MinesweeperConfig
from rl.common.epsilon_scheduler import make_scheduler
from rl.common.logger import Logger
from rl.common.replay_buffer import ReplayBuffer
from rl.common.utils import RollingMean, Timer, get_device, set_seed
from rl.networks.cnn_network import MinesweeperCNN


class DQNAgent:
    """
    DQN agent that trains on the Minesweeper environment.

    Assembles all components — network, algorithm, buffer, scheduler,
    logger — from a MinesweeperConfig and runs the training loop.

    Args:
        config:     MinesweeperConfig with all hyperparameters.
        env:        Minesweeper environment instance.
                    Must expose: env.reset() → obs
                                 env.step(action) → obs, reward, done, info

    Example:
        >>> cfg   = MinesweeperConfig.from_yaml("configs/config_dqn.yml")
        >>> env   = MinesweeperEnv(url=cfg.env_url, n=cfg.n, mines=cfg.mines)
        >>> agent = DQNAgent(cfg, env)
        >>> agent.train()
    """

    def __init__(self, config: MinesweeperConfig, env) -> None:
        self.config = config
        self.env = env
        self.device = get_device(config.device)

        # Store static environment parameters; solve_tiles is derived per mode.
        self._base_env_params = {
            "n": config.n,
            "mines": config.mines,
        }

        set_seed(config.seed)

        # ── Networks ──────────────────────────────────────────────────────
        self.online_net = MinesweeperCNN(
            n=config.n,
            hidden_dim=256,
        ).to(self.device)

        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=config.learning_rate,
        )

        # ── LR scheduler (warmup → cosine decay) ─────────────────────────
        # Built after optimizer. DQN.update() calls scheduler.step() only
        # AFTER optimizer.step(), so the ordering contract is satisfied.
        self.lr_scheduler = self._build_lr_scheduler()

        # ── DQN algorithm ─────────────────────────────────────────────────
        self.algo = DQN(
            online_net=self.online_net,
            target_net=self.target_net,
            optimizer=self.optimizer,
            gamma=config.gamma,
            grad_clip_norm=10.0,
            scheduler=self.lr_scheduler,
        )

        # ── Replay buffer ─────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            state_shape=(3, config.n, config.n),
            action_dim=config.n * config.n,
            obs_dtype=np.dtype(np.float32),
            use_action_mask=True,
        )

        # ── Epsilon scheduler ─────────────────────────────────────────────
        self.eps_scheduler = make_scheduler(config)

        # ── Logger ────────────────────────────────────────────────────────
        self.logger = Logger(
            log_dir=config.log_dir,
            run_name=config.run_name,
        )

        # ── Rolling metrics ───────────────────────────────────────────────
        self._win_rate = RollingMean(window=100)
        self._ep_timer = Timer()

        # ── Global counters ───────────────────────────────────────────────
        self._global_step = 0
        self._episode = 0

    # ── Training entry point ───────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the full training loop for config.total_episodes episodes.

        Evaluates every config.eval_every episodes.
        Saves a checkpoint after each evaluation pass.
        Dumps summary.csv once at the end of all training.
        """
        print(
            f"Training DQN on {self.config.n}×{self.config.n} Minesweeper "
            f"({self.config.mines} mines) for {self.config.total_episodes} episodes."
        )
        print(f"Device: {self.device}  |  Run: {self.config.run_name}")
        print(f"Network: {self.online_net}")
        print(f"Buffer:  {self.replay_buffer}")
        print()
        window_max_rev = 0
        window_max_rev_ratio = 0.0

        for ep in range(1, self.config.total_episodes + 1):
            self._episode = ep
            stats = self._train_episode()
            self._win_rate.push(float(stats["win"]))
            mine_density = self.config.mines / (self.config.n**2)
            window_max_rev = max(window_max_rev, stats["revealed_safe"])
            window_max_rev_ratio = max(window_max_rev_ratio, stats["revealed_ratio"])

            self.logger.log_scalars(
                {
                    "train/episode_reward": stats["reward"],
                    "train/win_rate": self._win_rate.mean(),
                    "train/mine_density": mine_density,
                    "train/steps_per_ep": stats["steps"],
                    "train/epsilon": self.eps_scheduler.get(self._global_step),
                    "train/solve_tiles": stats["solve_tiles"],
                    "train/revealed_ratio": stats["revealed_ratio"],
                    "train/final_unrevealed": stats["final_unrevealed"],
                    "train/remaining_safe": stats["remaining_safe"],
                },
                step=ep,
            )

            if ep % 100 == 0:
                print(
                    f"[Train Ep {ep:6d}]  "
                    f"R={stats['reward']:+.2f} | "
                    f"Win={self._win_rate.mean():.2f} | "
                    f"RevRatio={stats['revealed_ratio']:.3f} | "
                    f"RemSafe={int(stats['remaining_safe']):02d} | "
                    f"MaxRev={int(window_max_rev):02d} | "
                    f"MaxRatio={window_max_rev_ratio:.3f} | "
                    f"Q={stats['q_mean']:.2f} | "
                    f"Qmax={stats['q_max']:.2f} | "
                    f"ε={self.eps_scheduler.get(self._global_step):.3f} | "
                    f"solve={int(stats['solve_tiles'])} | "
                    f"lr={self.algo.current_lr:.2e} | "
                    f"upd={self.algo.update_count}"
                )
                window_max_rev = 0
                window_max_rev_ratio = 0.0

            if ep % self.config.eval_every == 0:
                eval_results = self.evaluate(self.config.eval_episodes)
                eval_reward = float(np.mean([r["reward"] for r in eval_results]))
                eval_win_rate = float(np.mean([r["win"] for r in eval_results]))
                eval_rev_ratio = float(
                    np.mean([r["revealed_ratio"] for r in eval_results])
                )
                eval_max_rev = float(
                    np.max([r["revealed_ratio"] for r in eval_results])
                )
                eval_rem_safe = float(
                    np.mean([r["remaining_safe"] for r in eval_results])
                )
                eval_unrevealed = float(
                    np.mean([r["final_unrevealed"] for r in eval_results])
                )
                eval_q_mean = float(np.mean([r["q_mean"] for r in eval_results]))
                eval_q_max = float(np.max([r["q_max"] for r in eval_results]))

                self.logger.log_scalars(
                    {
                        "eval/episode_reward": eval_reward,
                        "eval/win_rate": eval_win_rate,
                        "eval/revealed_ratio": eval_rev_ratio,
                        "eval/max_revealed_ratio": eval_max_rev,
                        "eval/remaining_safe": eval_rem_safe,
                        "eval/final_unrevealed": eval_unrevealed,
                        "eval/q_mean": eval_q_mean,
                        "eval/q_max": eval_q_max,
                    },
                    step=ep,
                )

                save_checkpoint(
                    path=checkpoint_path(
                        self.config.checkpoint_dir,
                        self.config.run_name,
                        episode=ep,
                    ),
                    network=self.online_net,
                    optimizer=self.optimizer,
                    episode=ep,
                    step=self._global_step,
                    config=self.config,
                    metadata={
                        "win_rate": self._win_rate.mean(),
                        "eval_win_rate": eval_win_rate,
                        "algo": "dqn",
                        "eval_rev_ratio": eval_rev_ratio,
                    },
                )

        self.logger.dump_summary()
        self.logger.close()
        print("Training complete.")

    # ── Single training episode ────────────────────────────────────────────────

    def _train_episode(self):
        """
        Run one training episode.

        Returns:
            Dictionary containing metrics for the episode.
        """
        # Use the sync context manager properly
        with self.env.sync() as env:
            current_solve_tiles = self._training_solve_tiles()
            obs_result = env.reset(
                **self._base_env_params,
                solve_tiles=current_solve_tiles,
            )
            obs = obs_result.observation
            done = False
            ep_reward = 0.0
            ep_steps = 0
            max_revealed = 0
            unique_actions = set()
            mine_hit = False
            ep_q_means = []
            ep_target_maxes = []

            with self._ep_timer:
                while not done:
                    epsilon = self.eps_scheduler.get(self._global_step)
                    action_idx = self._select_action(obs, epsilon)

                    unique_actions.add(action_idx)
                    action = self._idx_to_action(action_idx)
                    step_result = env.step(action)
                    next_obs = step_result.observation
                    reward = step_result.reward
                    done = step_result.done

                    revealed_safe = (
                        next_obs.n * next_obs.n
                        - next_obs.mines_count
                        - next_obs.unrevealed_count
                    )
                    max_revealed = max(max_revealed, revealed_safe)
                    if done and next_obs.status == 2:
                        mine_hit = True

                    self.replay_buffer.push(
                        state=obs.spatial,
                        action=action_idx,
                        reward=reward,
                        next_state=next_obs.spatial,
                        done=done,
                        next_action_mask=next_obs.action_mask,
                    )

                    obs = next_obs
                    ep_reward += reward
                    ep_steps += 1
                    self._global_step += 1

                    if (
                        len(self.replay_buffer) >= self.config.min_replay_size
                        and len(self.replay_buffer) >= self.config.batch_size
                    ):
                        batch = self.replay_buffer.sample(
                            self.config.batch_size, self.device
                        )
                        result = self.algo.update(batch)
                        ep_q_means.append(result.current_q_mean)
                        ep_target_maxes.append(result.target_q_max)

                        if self._global_step % self.config.target_update_freq == 0:
                            self.algo.sync_target()

                        self.logger.log_scalars(
                            {
                                "train/loss": result.loss,
                                "train/q_mean": result.q_mean,
                                "train/current_q_mean": result.current_q_mean,
                                "train/target_q_mean": result.target_q_mean,
                                "train/target_q_max": result.target_q_max,
                                "train/lr": self.algo.current_lr,
                            },
                            step=self._global_step,
                        )

        won = int(obs.status == 1)
        total_safe = obs.n * obs.n - obs.mines_count
        revealed_ratio = max_revealed / total_safe
        final_unrevealed = obs.unrevealed_count
        remaining_safe = total_safe - max_revealed
        revealed_ratio = max_revealed / total_safe if total_safe > 0 else 0.0

        self.logger.log_scalars(
            {"perf/wall_time_per_ep": self._ep_timer.elapsed_ms},
            step=self._episode,
        )

        return {
            "reward": ep_reward,
            "steps": ep_steps,
            "win": won,
            "solve_tiles": current_solve_tiles,
            "revealed_safe": max_revealed,
            "total_safe": total_safe,
            "revealed_ratio": revealed_ratio,
            "final_unrevealed": final_unrevealed,
            "remaining_safe": remaining_safe,
            "unique_actions": len(unique_actions),
            "mine_hit": mine_hit,
            "q_mean": float(np.mean(ep_q_means)) if ep_q_means else 0.0,
            "q_max": float(np.mean(ep_target_maxes)) if ep_target_maxes else 0.0,
        }

    # ── Action selection ───────────────────────────────────────────────────────
    def _select_action(self, obs, epsilon: float) -> int:
        """
        ε-greedy action selection with action masking.

        Only valid cells (obs.action_mask == True) are ever selected —
        both during random exploration and greedy exploitation.

        Args:
            obs:     MinesweeperObservation with .spatial and .action_mask.
            epsilon: Current exploration probability.

        Returns:
            Flat cell index in [0, n*n).
        """
        valid_actions = np.where(obs.action_mask)[0]

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state).cpu().numpy()[0]
        self.online_net.train()

        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked_q))

    def _greedy_action(self, obs) -> tuple[int, float]:
        """
        Pure greedy action selection (epsilon = 0) with action masking.
        Factored out of _select_action so evaluate() can call it directly
        without carrying epsilon or building a separate standalone function.
        Returns both the selected action and its corresponding Q-value.

        Args:
            obs: MinesweeperObservation with .spatial and .action_mask.

        Returns:
            (Flat cell index in [0, n*n), corresponding Q-value float)
        """
        valid_actions = np.where(obs.action_mask)[0]
        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state).cpu().numpy()[0]
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        best_action = int(np.argmax(masked_q))
        return best_action, float(masked_q[best_action])

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, n_episodes: int) -> list[dict]:
        """
        Run n_episodes greedy episodes and return per-episode stats.

        Public so evaluate.py can call agent.evaluate(args.episodes) directly
        without duplicating env interaction logic outside the agent.

        The network is put into eval() mode for the entire call and restored
        to train() mode afterwards — safe to call mid-training.

        Args:
            n_episodes: Number of greedy episodes to run.

        Returns:
            List of dicts, one per episode containing detailed metrics.
        """
        self.online_net.eval()
        results = []

        for ep in range(1, n_episodes + 1):
            with self.env.sync() as env:
                obs_result = env.reset(
                    **self._base_env_params,
                    solve_tiles=0,
                )
                obs = obs_result.observation
                done = False
                ep_reward = 0.0
                ep_steps = 0
                max_revealed = 0
                q_vals_ep = []

                while not done:
                    action_idx, q_val = self._greedy_action(obs)
                    q_vals_ep.append(q_val)
                    step_result = env.step(self._idx_to_action(action_idx))
                    obs = step_result.observation
                    ep_reward += step_result.reward
                    done = step_result.done
                    ep_steps += 1
                    revealed_safe = (
                        obs.n * obs.n - obs.mines_count - obs.unrevealed_count
                    )
                    max_revealed = max(max_revealed, revealed_safe)

                total_safe = obs.n * obs.n - obs.mines_count
                final_unrevealed = obs.unrevealed_count
                remaining_safe = total_safe - max_revealed
                revealed_ratio = max_revealed / total_safe if total_safe > 0 else 0.0
                results.append(
                    {
                        "reward": ep_reward,
                        "steps": ep_steps,
                        "win": int(obs.status == 1),
                        "revealed_safe": max_revealed,
                        "total_safe": total_safe,
                        "revealed_ratio": revealed_ratio,
                        "final_unrevealed": final_unrevealed,
                        "remaining_safe": remaining_safe,
                        "q_mean": float(np.mean(q_vals_ep)) if q_vals_ep else 0.0,
                        "q_max": float(np.max(q_vals_ep)) if q_vals_ep else 0.0,
                    }
                )

            if ep % max(1, n_episodes // 10) == 0 or ep == n_episodes:
                wr = sum(r["win"] for r in results) / len(results)
                w = len(str(n_episodes))
                avg_rev = np.mean([r["revealed_ratio"] for r in results])
                avg_rem = np.mean([r["remaining_safe"] for r in results])
                avg_q = np.mean([r["q_mean"] for r in results])
                avg_q_max = np.mean([r["q_max"] for r in results])
                max_rev = np.max([r["revealed_safe"] for r in results])
                max_rev_ratio = np.max([r["revealed_ratio"] for r in results])
                print(
                    f"  [Eval] ep {ep:{w}d}/{n_episodes} | "
                    f"Win={wr:.3f} | RevRatio={avg_rev:.3f} | "
                    f"RemSafe={avg_rem:.1f} | "
                    f"MaxRev={int(max_rev):02d} | MaxRatio={max_rev_ratio:.3f} | "
                    f"Q={avg_q:.2f} | Qmax={avg_q_max:.2f}",
                    end="\r",
                )

        print()
        self.online_net.train()
        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _training_solve_tiles(self) -> int:
        """
        Training curriculum schedule over global environment steps.

        - Hold `solve_tiles` constant until curriculum_hold_steps.
        - Linearly decay to 0 by curriculum_end_steps.
        - If curriculum is disabled, keep solve_tiles fixed.
        """
        start = max(0, int(self.config.solve_tiles))

        if not self.config.curriculum_enabled or start == 0:
            return start

        hold_steps = max(0, int(self.config.curriculum_hold_steps))
        end_steps = max(hold_steps, int(self.config.curriculum_end_steps))
        step = self._global_step

        if step <= hold_steps:
            return start
        if step >= end_steps:
            return 0

        progress = (step - hold_steps) / max(1, end_steps - hold_steps)
        value = start * (1.0 - progress)
        return max(0, int(np.ceil(value)))

    def _idx_to_action(self, idx: int) -> MinesweeperAction:
        """
        Convert a flat cell index to a MinesweeperAction.

        MinesweeperAction expects (row, col) derived from the flat index.
        This is the only place in the library that knows the index→action mapping.
        """
        row, col = divmod(idx, self.config.n)
        return MinesweeperAction(row=row, col=col)

    def _build_lr_scheduler(self):
        """Warmup → cosine decay over lr_total_updates steps."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

        warmup_steps = self.config.lr_warmup_steps
        total_updates = self.config.lr_total_updates

        warmup = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_updates - warmup_steps),
            eta_min=self.config.learning_rate / 10,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    # ── Checkpoint interface ───────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        """Resume training or run evaluation from a saved checkpoint."""
        meta = load_checkpoint(path, self.online_net, self.optimizer)
        self._global_step = meta.get("step", 0)
        self._episode = meta.get("episode", 0)
        self.algo.sync_target()
        print(
            f"Loaded checkpoint: {path}  "
            f"(episode={self._episode}, step={self._global_step})"
        )

    def __repr__(self) -> str:
        return (
            f"DQNAgent("
            f"n={self.config.n}, "
            f"mines={self.config.mines}, "
            f"device={self.device}, "
            f"episode={self._episode}, "
            f"step={self._global_step})"
        )
