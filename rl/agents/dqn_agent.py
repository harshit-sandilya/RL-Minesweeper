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

        # Store environment parameters for reset
        self.env_params = {
            "n": config.n,
            "mines": config.mines,
            "solve_tiles": config.solve_tiles,
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
            state_shape=(1, config.n, config.n),
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

        # FIX #1: Timer is a context manager (__enter__/__exit__).
        # We use it per-episode via `with self._ep_timer:` in _train_episode().
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

        for ep in range(1, self.config.total_episodes + 1):
            self._episode = ep
            ep_reward, ep_steps, won = self._train_episode()

            self._win_rate.push(float(won))
            mine_density = self.config.mines / (self.config.n**2)

            self.logger.log_scalars(
                {
                    "train/episode_reward": ep_reward,
                    "train/win_rate": self._win_rate.mean(),
                    "train/mine_density": mine_density,
                    "train/steps_per_ep": ep_steps,
                    "train/epsilon": self.eps_scheduler.get(self._global_step),
                },
                step=ep,
            )

            if ep % self.config.eval_every == 0:
                eval_results = self.evaluate(self.config.eval_episodes)
                eval_reward = float(np.mean([r["reward"] for r in eval_results]))
                eval_win_rate = float(np.mean([r["win"] for r in eval_results]))

                self.logger.log_scalars(
                    {
                        "eval/episode_reward": eval_reward,
                        "eval/win_rate": eval_win_rate,
                    },
                    step=ep,
                )

                # FIX #6: use checkpoint_path() for consistent zero-padded filenames
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
                    },
                )

                print(
                    f"[Ep {ep:6d}]  "
                    f"reward={ep_reward:+.2f}  "
                    f"win={self._win_rate.mean():.2f}  "
                    f"eval_win={eval_win_rate:.2f}  "
                    f"ε={self.eps_scheduler.get(self._global_step):.3f}  "
                    f"lr={self.algo.current_lr:.2e}  "
                    f"updates={self.algo.update_count}"
                )

        # FIX #5: dump_summary() and close() are OUTSIDE the for loop.
        # They must run exactly once after all episodes complete.
        self.logger.dump_summary()
        self.logger.close()
        print("Training complete.")

    # ── Single training episode ────────────────────────────────────────────────

    def _train_episode(self):
        """
        Run one training episode.

        Returns:
            (total_reward, steps_taken, won)
        """
        # Use the sync context manager properly
        with self.env.sync() as env:
            obs_result = env.reset(**self.env_params)
            obs = obs_result.observation
            done = False
            ep_reward = 0.0
            ep_steps = 0

            # FIX #2: initialise info before the loop so it's always defined,
            # even if the episode ends on the very first step (done=True on reset).
            info = {}

            # FIX #1: Timer used as context manager — starts on __enter__,
            # records elapsed_ms on __exit__. ep_wall_time is read after the block.
            with self._ep_timer:
                while not done:
                    epsilon = self.eps_scheduler.get(self._global_step)
                    action_idx = self._select_action(obs, epsilon)

                    action = self._idx_to_action(action_idx)
                    step_result = env.step(action)
                    next_obs = step_result.observation
                    reward = step_result.reward
                    done = step_result.done
                    info = getattr(step_result, "info", {})

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

                        if self._global_step % self.config.target_update_freq == 0:
                            self.algo.sync_target()

                        self.logger.log_scalars(
                            {
                                "train/loss": result.loss,
                                "train/q_mean": result.q_mean,
                                "train/lr": self.algo.current_lr,
                            },
                            step=self._global_step,
                        )

        won = info.get("won", False)

        self.logger.log_scalars(
            {"perf/wall_time_per_ep": self._ep_timer.elapsed_ms},
            step=self._episode,
        )

        return ep_reward, ep_steps, won

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

    def _greedy_action(self, obs) -> int:
        """
        Pure greedy action selection (epsilon = 0) with action masking.

        Factored out of _select_action so evaluate() can call it directly
        without carrying epsilon or building a separate standalone function.

        Identical to _select_action(obs, epsilon=0.0) but without the
        random branch — no np.random call, no network.train() toggle needed
        since evaluate() keeps the network in eval() mode throughout.

        Args:
            obs: MinesweeperObservation with .spatial and .action_mask.

        Returns:
            Flat cell index in [0, n*n).
        """
        valid_actions = np.where(obs.action_mask)[0]
        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state).cpu().numpy()[0]
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

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
            List of dicts, one per episode:
                {"reward": float, "steps": int, "win": int (0|1)}
        """
        self.online_net.eval()
        results = []

        for ep in range(1, n_episodes + 1):
            with self.env.sync() as env:
                obs_result = env.reset(**self.env_params)
                obs = obs_result.observation
                done = False
                ep_reward = 0.0
                ep_steps = 0
                info: dict = {}

                while not done:
                    action_idx = self._greedy_action(obs)
                    step_result = env.step(self._idx_to_action(action_idx))
                    obs = step_result.observation
                    ep_reward += step_result.reward
                    done = step_result.done
                    info = getattr(step_result, "info", {})
                    ep_steps += 1

            results.append(
                {
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "win": int(info.get("won", False)),
                }
            )

            if ep % max(1, n_episodes // 10) == 0 or ep == n_episodes:
                wr = sum(r["win"] for r in results) / len(results)
                w = len(str(n_episodes))
                print(
                    f"  [eval] ep {ep:{w}d}/{n_episodes}  win_rate={wr:.3f}", end="\r"
                )

        print()
        self.online_net.train()
        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

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
