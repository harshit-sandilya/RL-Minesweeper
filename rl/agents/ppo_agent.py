"""
rl/agents/ppo_agent.py

PPO agent for Minesweeper.

This file is intentionally problem-coupled — it knows about:
    MinesweeperEnv, MinesweeperObservation, MinesweeperConfig,
    obs.spatial, obs.action_mask, action index → MinesweeperAction

Everything below this file in the stack is problem-agnostic:
    PPO algorithm, RolloutBuffer, Logger, Config, ActorNetwork, CriticNetwork

Mirrors A2CAgent's structure and public interface (train(), evaluate(),
load_checkpoint(), the same MinesweeperConfig, the same
TopKCheckpointManager, the same result-dict shape for metrics.py) —
same networks, same joint optimizer convention, same curriculum. The one
structural difference from A2CAgent: PPO's unit of "one update" is a
ROLLOUT of several full episodes, reused for several gradient epochs,
not a single episode with a single update.

    ✗  No replay buffer  — on-policy, same reasoning as A2C.
    ✗  No target network — same reasoning as A2C.
    ✗  No ε-greedy schedule — exploration = policy stochasticity + entropy bonus.
    ✓  One rollout == rollout_episodes full episodes (RolloutBuffer),
       reused for ppo_epochs passes of minibatch updates.

Training loop (one rollout)
----------------------------
    buf = RolloutBuffer()
    for _ in range(rollout_episodes):
        play one full episode, sampling from the masked policy,
        recording old_log_prob and V(s) at collection time
        buf.add_episode(...)
    buf.compute_advantages(gamma, gae_lambda)
    for epoch in range(ppo_epochs):
        for minibatch in buf.minibatches(minibatch_size, device):
            algo.update(minibatch)     # MANY updates per rollout
    log rollout metrics
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import from_numpy
from torch.distributions import Categorical

from minesweeper_env import MinesweeperAction
from rl.algorithms.ppo import PPO
from rl.common.checkpoint import TopKCheckpointManager, load_checkpoint
from rl.common.config import MinesweeperConfig
from rl.common.logger import Logger
from rl.common.metrics import wilson_ci
from rl.common.rollout_buffer import RolloutBuffer
from rl.common.utils import RollingMean, Timer, get_device, set_seed
from rl.networks.actor_network import ActorNetwork
from rl.networks.critic_network import CriticNetwork


class PPOAgent:
    """
    PPO agent that trains on the Minesweeper environment.

    Args:
        config: MinesweeperConfig with all hyperparameters (same class
                DQNAgent/A2CAgent use — env fields are 100% shared; the
                6 new PPO fields are in configs/ppo.yml).
        env:    Minesweeper environment instance, same contract as DQNAgent/A2CAgent.

    Example:
        >>> cfg   = MinesweeperConfig.from_yaml("configs/ppo.yml")
        >>> env   = MinesweeperEnv(base_url=cfg.env_url)
        >>> agent = PPOAgent(cfg, env)
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

        # ── Networks (identical shapes to A2C) ──────────────────────────────
        self.actor_net = ActorNetwork(n=config.n, hidden_dim=256).to(self.device)
        self.critic_net = CriticNetwork(n=config.n, hidden_dim=256).to(self.device)

        # ── Optimizer (single, joint) ────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor_net.parameters(), self.critic_net.parameters()),
            lr=config.learning_rate,
        )

        self.lr_scheduler = self._build_lr_scheduler()

        # ── PPO algorithm ────────────────────────────────────────────────
        self.algo = PPO(
            actor_net=self.actor_net,
            critic_net=self.critic_net,
            optimizer=self.optimizer,
            clip_epsilon=config.clip_epsilon,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            clip_value_loss=config.clip_value_loss,
            grad_clip_norm=config.grad_clip_norm,
            scheduler=self.lr_scheduler,
        )

        self.logger = Logger(log_dir=config.log_dir, run_name=config.run_name)

        self._win_rate = RollingMean(window=100)
        self._ep_timer = Timer()

        self._global_step = 0
        self._episode = 0

    # ── Training entry point ───────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the full training loop until config.total_episodes episodes
        have been played, one rollout of config.rollout_episodes episodes
        at a time.

        Evaluates whenever a rollout crosses a multiple of config.eval_every
        (rollout_episodes generally doesn't divide eval_every evenly, so
        this is checked as a boundary-crossing test, not `ep % eval_every`).
        Keeps only the top-config.checkpoint_top_k checkpoints by eval
        win-rate CI lower bound (same TopKCheckpointManager as DQN/A2C).
        """
        print(
            f"Training PPO on {self.config.n}×{self.config.n} Minesweeper "
            f"({self.config.mines} mines) for {self.config.total_episodes} episodes."
        )
        print(f"Device: {self.device}  |  Run: {self.config.run_name}")
        print(f"Actor:  {self.actor_net}")
        print(f"Critic: {self.critic_net}")
        print(
            f"Rollout: {self.config.rollout_episodes} episodes/rollout, "
            f"{self.config.ppo_epochs} epochs, minibatch={self.config.minibatch_size}, "
            f"clip_epsilon={self.config.clip_epsilon}"
        )
        print()

        ckpt_mgr = TopKCheckpointManager(
            directory=self.config.checkpoint_dir,
            run_name=self.config.run_name,
            k=self.config.checkpoint_top_k,
        )
        print(
            f"Checkpointing: keeping top-{ckpt_mgr.k} by eval win-rate CI lower bound"
        )

        ep = 0
        rollout_idx = 0

        while ep < self.config.total_episodes:
            rollout_idx += 1
            ep_before_rollout = ep

            buf = RolloutBuffer()
            rollout_stats = []

            n_this_rollout = min(
                self.config.rollout_episodes, self.config.total_episodes - ep
            )
            for _ in range(n_this_rollout):
                ep += 1
                self._episode = ep
                episode_arrays, ep_stat = self._collect_episode()
                buf.add_episode(**episode_arrays)
                rollout_stats.append(ep_stat)
                self._win_rate.push(float(ep_stat["win"]))

            buf.compute_advantages(self.config.gamma, self.config.gae_lambda)

            update_results = []
            for _epoch in range(self.config.ppo_epochs):
                for minibatch in buf.minibatches(
                    self.config.minibatch_size, self.device
                ):
                    update_results.append(self.algo.update(minibatch))

            # ── Rollout-level aggregate metrics ─────────────────────────────
            avg_reward = float(np.mean([s["reward"] for s in rollout_stats]))
            avg_rev_ratio = float(np.mean([s["revealed_ratio"] for s in rollout_stats]))
            avg_rem_safe = float(np.mean([s["remaining_safe"] for s in rollout_stats]))
            avg_steps = float(np.mean([s["steps"] for s in rollout_stats]))
            avg_loss = float(np.mean([r.loss for r in update_results]))
            avg_actor_loss = float(np.mean([r.actor_loss for r in update_results]))
            avg_critic_loss = float(np.mean([r.critic_loss for r in update_results]))
            avg_entropy = float(np.mean([r.entropy for r in update_results]))
            avg_approx_kl = float(np.mean([r.approx_kl for r in update_results]))
            avg_clip_frac = float(np.mean([r.clip_fraction for r in update_results]))

            self.logger.log_scalars(
                {
                    "train/episode_reward": avg_reward,
                    "train/win_rate": self._win_rate.mean(),
                    "train/mine_density": self.config.mines / (self.config.n**2),
                    "train/steps_per_ep": avg_steps,
                    "train/revealed_ratio": avg_rev_ratio,
                    "train/remaining_safe": avg_rem_safe,
                    "train/loss": avg_loss,
                    "train/actor_loss": avg_actor_loss,
                    "train/critic_loss": avg_critic_loss,
                    "train/entropy": avg_entropy,
                    "train/approx_kl": avg_approx_kl,
                    "train/clip_fraction": avg_clip_frac,
                    "train/lr": self.algo.current_lr,
                },
                step=ep,
            )

            # ── Periodic print, every ~100 episodes (crossing-check, same
            #    reasoning as the eval trigger below) ────────────────────────
            if ep // 100 != ep_before_rollout // 100:
                print(
                    f"[Train Ep {ep:6d}]  (rollout {rollout_idx}) "
                    f"R={avg_reward:+.2f} | "
                    f"Win={self._win_rate.mean():.2f} | "
                    f"RevRatio={avg_rev_ratio:.3f} | "
                    f"RemSafe={avg_rem_safe:.1f} | "
                    f"Loss={avg_loss:.3f} | "
                    f"AL={avg_actor_loss:+.3f} | "
                    f"CL={avg_critic_loss:.3f} | "
                    f"Ent={avg_entropy:.3f} | "
                    f"KL={avg_approx_kl:.4f} | "
                    f"ClipFrac={avg_clip_frac:.2f} | "
                    f"lr={self.algo.current_lr:.2e} | "
                    f"upd={self.algo.update_count}"
                )

            # ── Eval trigger: did this rollout cross a multiple of
            #    eval_every? (rollout_episodes generally does NOT divide
            #    eval_every evenly, so `ep % eval_every == 0` would silently
            #    skip evals forever — this checks for crossing instead.) ────
            if (
                ep // self.config.eval_every
                != ep_before_rollout // self.config.eval_every
            ):
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
                        "algo": "ppo",
                        "eval_rev_ratio": eval_rev_ratio,
                        # same trick as A2CAgent — checkpoint.py only has
                        # one model_state_dict slot, so the critic rides
                        # along in metadata. See load_checkpoint() below.
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

    # ── Single episode collection (no gradient update here) ────────────────────

    def _collect_episode(self) -> tuple[dict, dict]:
        """
        Play one full episode on-policy, recording old_log_prob and V(s)
        at collection time for every step (needed later by PPO's clipped
        ratio — those must stay fixed across all ppo_epochs of updates
        that reuse this episode).

        Returns:
            (episode_arrays, ep_stat) where episode_arrays is ready to pass
            straight into RolloutBuffer.add_episode(**episode_arrays), and
            ep_stat matches the same per-episode stat dict shape DQNAgent/
            A2CAgent use for logging.
        """
        states, actions, log_probs, values, rewards, dones, masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        with self.env.sync() as env, self._ep_timer:
            current_solve_tiles = self._training_solve_tiles()
            obs_result = env.reset(
                **self._base_env_params, solve_tiles=current_solve_tiles
            )
            obs = obs_result.observation
            done = False
            ep_reward = 0.0
            ep_steps = 0
            max_revealed = 0

            while not done:
                action_idx, log_prob, value, mask = self._sample_action(obs)
                step_result = env.step(self._idx_to_action(action_idx))
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
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(float(done))
                masks.append(mask)

                obs = next_obs
                ep_reward += reward
                ep_steps += 1
                self._global_step += 1

        episode_arrays = {
            "states": np.stack(states).astype(np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "old_log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "action_masks": np.stack([m.cpu().numpy() for m in masks]),
        }

        won = int(obs.status == 1)
        total_safe = obs.n * obs.n - obs.mines_count
        remaining_safe = total_safe - max_revealed
        revealed_ratio = max_revealed / total_safe if total_safe > 0 else 0.0

        ep_stat = {
            "reward": ep_reward,
            "steps": ep_steps,
            "win": won,
            "solve_tiles": current_solve_tiles,
            "revealed_safe": max_revealed,
            "total_safe": total_safe,
            "revealed_ratio": revealed_ratio,
            "remaining_safe": remaining_safe,
        }
        return episode_arrays, ep_stat

    # ── Action selection ───────────────────────────────────────────────────────

    def _sample_action(self, obs) -> tuple[int, float, float, torch.Tensor]:
        """
        Stochastic on-policy action selection, ALSO recording the log-prob
        and value under the current policy — both needed by PPO's rollout
        buffer since they're compared against the (changing) policy across
        several update epochs.

        Returns:
            (action_idx, old_log_prob, value, mask)
        """
        mask = torch.from_numpy(obs.action_mask).to(self.device)
        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor_net(state).squeeze(0)
            logits = logits.masked_fill(~mask, float("-inf"))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic_net(state).squeeze()

        return int(action.item()), float(log_prob.item()), float(value.item()), mask

    def _greedy_action(self, obs) -> int:
        """
        Deterministic action selection for evaluation: argmax of the
        masked policy — identical to A2CAgent._greedy_action, so DQN/A2C/
        PPO eval numbers are comparable under the same convention.
        """
        mask = torch.from_numpy(obs.action_mask).to(self.device)
        state = from_numpy(obs.spatial).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor_net(state).squeeze(0)
            logits = logits.masked_fill(~mask, float("-inf"))
            action_idx = int(torch.argmax(logits).item())

        return action_idx

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int, algorithm: str = "ppo") -> list[dict]:
        """
        Run n_episodes greedy evaluation episodes on the FULL board
        (solve_tiles=0) — identical result-dict contract to DQNAgent/
        A2CAgent.evaluate(), so rl.common.metrics works unchanged.
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

    # ── Curriculum (identical logic to DQNAgent/A2CAgent) ───────────────────────

    def _training_solve_tiles(self) -> int:
        """Training curriculum schedule over global environment steps."""
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

    # ── LR scheduler (identical builder to DQNAgent/A2CAgent) ───────────────────

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

        Same two-network-in-metadata approach as A2CAgent — checkpoint.py's
        schema only has one model_state_dict slot, so the critic rides in
        metadata["critic_state_dict"].

        evaluate_ppo.py does NOT use this — it mirrors evaluate_a2c/dqn and
        calls rl.common.checkpoint.load_checkpoint on the actor only
        (no optimizer), then restores the critic from metadata.
        """
        meta = load_checkpoint(path, self.actor_net, self.optimizer)
        critic_state = meta.get("metadata", {}).get("critic_state_dict")
        if critic_state is not None:
            self.critic_net.load_state_dict(critic_state)
        else:
            print(
                "[PPOAgent] WARNING: checkpoint has no critic_state_dict — "
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
        row, col = divmod(idx, self.config.n)
        return MinesweeperAction(row=row, col=col)

    def __repr__(self) -> str:
        return (
            f"PPOAgent("
            f"n={self.config.n}, "
            f"mines={self.config.mines}, "
            f"device={self.device}, "
            f"episode={self._episode}, "
            f"step={self._global_step})"
        )
