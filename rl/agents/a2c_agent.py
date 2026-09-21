"""
rl/agents/a2c_agent.py

A2C agent for Minesweeper.

This file is intentionally problem-coupled — it knows about:
    MinesweeperEnv, MinesweeperObservation, MinesweeperConfig,
    obs.spatial, obs.action_mask, action index → MinesweeperAction

Everything below this file in the stack is problem-agnostic:
    A2C algorithm, Logger, Config, ActorNetwork, CriticNetwork

Mirrors DQNAgent's structure and public interface (train(), evaluate(),
load_checkpoint(), the same MinesweeperConfig, the same
TopKCheckpointManager, the same result-dict shape for metrics.py) so
results are directly comparable — only the parts that must differ for
an on-policy algorithm differ:

    ✗  No replay buffer  — A2C is on-policy; nothing here samples old
                            transitions from a different policy.
    ✗  No target network — A2C has no bootstrapped max_a' Q(s',a') term.
    ✗  No ε-greedy schedule — exploration comes from sampling the
                            stochastic policy + the entropy bonus.
    ✓  One rollout == one full episode (see rl.algorithms.a2c module
       docstring for why), fed to A2C.update() once per episode.

Training loop (one episode)
----------------------------
    obs = env.reset()
    states, actions, masks, rewards, dones = [], [], [], [], []
    while not done:
        sample action from masked Categorical(actor_net(obs.spatial))
        step env → next_obs, reward, done
        append transition to rollout lists
        obs = next_obs
    algo.update(rollout batch)   # ONE update per episode
    log episode metrics
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import from_numpy
from torch.distributions import Categorical

from minesweeper_env import MinesweeperAction
from rl.algorithms.a2c import A2C
from rl.common.checkpoint import TopKCheckpointManager, load_checkpoint
from rl.common.config import MinesweeperConfig
from rl.common.logger import Logger
from rl.common.metrics import wilson_ci
from rl.common.utils import RollingMean, Timer, get_device, set_seed
from rl.networks.actor_network import ActorNetwork
from rl.networks.critic_network import CriticNetwork


class A2CAgent:
    """
    A2C agent that trains on the Minesweeper environment.

    Args:
        config: MinesweeperConfig with all hyperparameters (same class
                DQNAgent uses — env fields are 100% shared; only 3 new
                algorithm fields are needed: value_coef, entropy_coef,
                grad_clip_norm — see configs/a2c.yml).
        env:    Minesweeper environment instance, same contract as DQNAgent.

    Example:
        >>> cfg   = MinesweeperConfig.from_yaml("configs/a2c.yml")
        >>> env   = MinesweeperEnv(base_url=cfg.env_url)
        >>> agent = A2CAgent(cfg, env)
        >>> agent.train()
    """

    def __init__(self, config: MinesweeperConfig, env) -> None:
        self.config = config
        self.env = env
        self.device = get_device(config.device)

        self._base_env_params = {
            "n": config.n,
            "mines": config.mines,
        }

        set_seed(config.seed)

        # ── Networks ──────────────────────────────────────────────────────
        self.actor_net = ActorNetwork(n=config.n, hidden_dim=256).to(self.device)
        self.critic_net = CriticNetwork(n=config.n, hidden_dim=256).to(self.device)

        # ── Optimizer (single, joint — matches Mnih et al. 2016) ───────────
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor_net.parameters(), self.critic_net.parameters()),
            lr=config.learning_rate,
        )

        # ── LR scheduler (warmup → cosine decay), same builder as DQN ──────
        self.lr_scheduler = self._build_lr_scheduler()

        # ── A2C algorithm ────────────────────────────────────────────────
        self.algo = A2C(
            actor_net=self.actor_net,
            critic_net=self.critic_net,
            optimizer=self.optimizer,
            gamma=config.gamma,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            grad_clip_norm=config.grad_clip_norm,
            scheduler=self.lr_scheduler,
        )

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
        Keeps only the top-config.checkpoint_top_k checkpoints by eval
        win-rate CI lower bound (same TopKCheckpointManager as DQNAgent).
        Dumps summary.csv once at the end of all training.
        """
        print(
            f"Training A2C on {self.config.n}×{self.config.n} Minesweeper "
            f"({self.config.mines} mines) for {self.config.total_episodes} episodes."
        )
        print(f"Device: {self.device}  |  Run: {self.config.run_name}")
        print(f"Actor:  {self.actor_net}")
        print(f"Critic: {self.critic_net}")
        print()

        ckpt_mgr = TopKCheckpointManager(
            directory=self.config.checkpoint_dir,
            run_name=self.config.run_name,
            k=self.config.checkpoint_top_k,
        )
        print(
            f"Checkpointing: keeping top-{ckpt_mgr.k} by eval win-rate CI lower bound"
        )

        for ep in range(1, self.config.total_episodes + 1):
            self._episode = ep
            stats = self._train_episode()
            self._win_rate.push(float(stats["win"]))
            mine_density = self.config.mines / (self.config.n**2)

            self.logger.log_scalars(
                {
                    "train/episode_reward": stats["reward"],
                    "train/win_rate": self._win_rate.mean(),
                    "train/mine_density": mine_density,
                    "train/steps_per_ep": stats["steps"],
                    "train/solve_tiles": stats["solve_tiles"],
                    "train/revealed_ratio": stats["revealed_ratio"],
                    "train/final_unrevealed": stats["final_unrevealed"],
                    "train/remaining_safe": stats["remaining_safe"],
                    "train/loss": stats["loss"],
                    "train/actor_loss": stats["actor_loss"],
                    "train/critic_loss": stats["critic_loss"],
                    "train/entropy": stats["entropy"],
                    "train/value_mean": stats["value_mean"],
                    "train/lr": self.algo.current_lr,
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
                    f"Loss={stats['loss']:.3f} | "
                    f"AL={stats['actor_loss']:+.3f} | "
                    f"CL={stats['critic_loss']:.3f} | "
                    f"Ent={stats['entropy']:.3f} | "
                    f"V={stats['value_mean']:.2f} | "
                    f"solve={int(stats['solve_tiles'])} | "
                    f"lr={self.algo.current_lr:.2e} | "
                    f"upd={self.algo.update_count}"
                )

            if ep % self.config.eval_every == 0:
                eval_results = self.evaluate(self.config.eval_episodes)
                eval_reward = float(np.mean([r["reward"] for r in eval_results]))
                eval_win_rate = float(np.mean([r["win"] for r in eval_results]))
                eval_rev_ratio = float(
                    np.mean([r["revealed_ratio"] for r in eval_results])
                )
                eval_rem_safe = float(
                    np.mean([r["remaining_safe"] for r in eval_results])
                )

                eval_wins = int(sum(r["win"] for r in eval_results))
                eval_ci_lo, eval_ci_hi = wilson_ci(eval_wins, len(eval_results))

                self.logger.log_scalars(
                    {
                        "eval/episode_reward": eval_reward,
                        "eval/win_rate": eval_win_rate,
                        "eval/win_rate_ci_lower": eval_ci_lo,
                        "eval/win_rate_ci_upper": eval_ci_hi,
                        "eval/revealed_ratio": eval_rev_ratio,
                        "eval/remaining_safe": eval_rem_safe,
                    },
                    step=ep,
                )

                saved = ckpt_mgr.maybe_save(
                    metric=eval_ci_lo,
                    network=self.actor_net,
                    optimizer=self.optimizer,
                    episode=ep,
                    step=self._global_step,
                    config=self.config,
                    metadata={
                        "win_rate": self._win_rate.mean(),
                        "eval_win_rate": eval_win_rate,
                        "eval_win_rate_ci": [eval_ci_lo, eval_ci_hi],
                        "algo": "a2c",
                        "eval_rev_ratio": eval_rev_ratio,
                        # save_checkpoint() only has one model_state_dict
                        # slot (built for single-network agents); stash the
                        # critic here rather than touching checkpoint.py's
                        # schema. See load_checkpoint() below.
                        "critic_state_dict": self.critic_net.state_dict(),
                    },
                )
                if not saved:
                    print(
                        f"[Checkpoint] Skipped ep={ep}  "
                        f"(CI-lower={eval_ci_lo:.3f} did not beat top-{ckpt_mgr.k})"
                    )

        self.logger.dump_summary()
        self.logger.close()
        print("Training complete.")

    # ── Single training episode (== one on-policy rollout) ─────────────────────

    def _train_episode(self) -> dict:
        """
        Play one full episode on-policy, then perform ONE A2C update using
        the entire episode as the rollout. See rl.algorithms.a2c module
        docstring for why rollout length == episode length here.
        """
        states, actions, masks, rewards, dones = [], [], [], [], []

        with self.env.sync() as env, self._ep_timer:
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

            while not done:
                action_idx, mask_tensor = self._sample_action(obs)
                action = self._idx_to_action(action_idx)
                step_result = env.step(action)
                next_obs = step_result.observation
                reward = step_result.reward
                done = step_result.done

                is_mine_hit_step = done and next_obs.status == 2
                revealed_safe = (
                    next_obs.n * next_obs.n
                    - next_obs.unrevealed_count
                    - int(is_mine_hit_step)
                )
                max_revealed = max(max_revealed, revealed_safe)

                states.append(obs.spatial)
                actions.append(action_idx)
                masks.append(mask_tensor)
                rewards.append(reward)
                dones.append(float(done))

                obs = next_obs
                ep_reward += reward
                ep_steps += 1
                self._global_step += 1

        batch = {
            "states": torch.from_numpy(np.stack(states)).float().to(self.device),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "action_masks": torch.stack(masks).to(self.device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
            "bootstrap_value": 0.0,  # every rollout here runs to a true terminal state
        }
        result = self.algo.update(batch)

        won = int(obs.status == 1)
        total_safe = obs.n * obs.n - obs.mines_count
        final_unrevealed = obs.unrevealed_count
        remaining_safe = total_safe - max_revealed
        revealed_ratio = max_revealed / total_safe if total_safe > 0 else 0.0

        self.logger.log_scalars(
            {"perf/wall_time_per_ep": self._ep_timer.elapsed_ms}, step=self._episode
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
            "loss": result.loss,
            "actor_loss": result.actor_loss,
            "critic_loss": result.critic_loss,
            "entropy": result.entropy,
            "value_mean": result.value_mean,
        }

    # ── Action selection ───────────────────────────────────────────────────────

    def _sample_action(self, obs) -> tuple[int, torch.Tensor]:
        """
        Stochastic on-policy action selection: sample from the masked
        categorical policy. No epsilon involved — A2C's exploration comes
        entirely from policy stochasticity + the entropy bonus in the loss.

        Returns:
            (sampled flat cell index, bool mask tensor used — stored so
             A2C.update() recomputes log-probs under the SAME masked
             distribution the action was actually sampled from)
        """
        mask = torch.from_numpy(obs.action_mask).to(self.device)
        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor_net(state).squeeze(0)  # (num_actions,)
            logits = logits.masked_fill(~mask, float("-inf"))
            dist = Categorical(logits=logits)
            action_idx = int(dist.sample().item())

        return action_idx, mask

    def _greedy_action(self, obs) -> int:
        """
        Deterministic action selection for evaluation: argmax of the
        masked policy (the mode of the distribution), not a sample —
        mirrors DQNAgent's greedy (epsilon=0) evaluation so DQN/A2C eval
        numbers are comparable under the same "no exploration noise"
        convention.
        """
        mask = torch.from_numpy(obs.action_mask).to(self.device)
        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor_net(state).squeeze(0)
            logits = logits.masked_fill(~mask, float("-inf"))
            action_idx = int(torch.argmax(logits).item())

        return action_idx

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int, algorithm: str = "a2c") -> list[dict]:
        """
        Run n_episodes greedy (argmax-policy) evaluation episodes on the
        FULL board (solve_tiles=0) — same result-dict contract as
        DQNAgent.evaluate(), so rl.common.metrics works unchanged.
        """
        self.actor_net.eval()
        self.critic_net.eval()
        results = []

        for ep in range(1, n_episodes + 1):
            with self.env.sync() as env, Timer() as ep_timer:
                obs_result = env.reset(**self._base_env_params, solve_tiles=0)
                obs = obs_result.observation
                done = False
                ep_reward = 0.0
                ep_steps = 0
                max_revealed = 0

                while not done:
                    action_idx = self._greedy_action(obs)
                    step_result = env.step(self._idx_to_action(action_idx))
                    obs = step_result.observation
                    ep_reward += step_result.reward
                    done = step_result.done
                    ep_steps += 1
                    is_mine_hit_step = done and obs.status == 2
                    revealed_safe = (
                        obs.n * obs.n - obs.unrevealed_count - int(is_mine_hit_step)
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
                    "wall_clock_sec": ep_timer.elapsed_ms / 1000.0,
                    "board_n": obs.n,
                    "mines_count": obs.mines_count,
                    "algorithm": algorithm,
                }
            )

            if ep % max(1, n_episodes // 10) == 0 or ep == n_episodes:
                wr = sum(r["win"] for r in results) / len(results)
                w = len(str(n_episodes))
                avg_rev = np.mean([r["revealed_ratio"] for r in results])
                avg_rem = np.mean([r["remaining_safe"] for r in results])
                max_rev = np.max([r["revealed_safe"] for r in results])
                max_rev_ratio = np.max([r["revealed_ratio"] for r in results])
                print(
                    f"  [Eval] ep {ep:{w}d}/{n_episodes} | "
                    f"Win={wr:.3f} | RevRatio={avg_rev:.3f} | "
                    f"RemSafe={avg_rem:.1f} | "
                    f"MaxRev={int(max_rev):02d} | MaxRatio={max_rev_ratio:.3f}",
                    end="\r",
                )

        print()
        self.actor_net.train()
        self.critic_net.train()
        return results

    # ── Curriculum (identical logic to DQNAgent._training_solve_tiles) ─────────

    def _training_solve_tiles(self) -> int:
        """
        Training curriculum schedule over global environment steps.
        Copied unchanged from DQNAgent so both algorithms get identical
        curriculum treatment for a fair comparison.
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

    # ── LR scheduler (identical builder to DQNAgent) ────────────────────────────

    def _build_lr_scheduler(self):
        """Warmup → cosine decay over lr_total_updates updates."""
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

    def load_checkpoint(self, path: str) -> dict:
        """
        Resume training from a saved checkpoint (restores optimizer too).

        save_checkpoint()'s payload only has one "model_state_dict" slot
        (built for single-network agents like DQN). A2C has two networks,
        so the critic's state_dict is stashed in
        metadata["critic_state_dict"] at save time (see train()) and
        restored here explicitly.

        evaluate_a2c.py does NOT use this — it mirrors evaluate_dqn and
        calls rl.common.checkpoint.load_checkpoint on the actor only
        (no optimizer), then restores the critic from metadata.
        """
        meta = load_checkpoint(path, self.actor_net, self.optimizer)
        critic_state = meta.get("metadata", {}).get("critic_state_dict")
        if critic_state is not None:
            self.critic_net.load_state_dict(critic_state)
        else:
            print(
                "[A2CAgent] WARNING: checkpoint has no critic_state_dict — "
                "critic weights were NOT restored (actor-only checkpoint?)."
            )
        self._global_step = meta.get("step", 0)
        self._episode = meta.get("episode", 0)
        print(
            f"Loaded checkpoint: {path}  "
            f"(episode={self._episode}, step={self._global_step})"
        )
        return meta

    def _idx_to_action(self, idx: int) -> MinesweeperAction:
        """Convert a flat cell index to a MinesweeperAction."""
        row, col = divmod(idx, self.config.n)
        return MinesweeperAction(row=row, col=col)

    def __repr__(self) -> str:
        return (
            f"A2CAgent("
            f"n={self.config.n}, "
            f"mines={self.config.mines}, "
            f"device={self.device}, "
            f"episode={self._episode}, "
            f"step={self._global_step})"
        )
