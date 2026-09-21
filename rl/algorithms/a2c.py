"""
rl/algorithms/a2c.py

Advantage Actor-Critic (A2C) update logic.

What lives here
---------------
Exactly the A2C update rule — nothing else.

    1. Compute discounted Monte Carlo returns for one full episode
       rollout (bootstrap = 0, since every rollout here runs to a
       terminal state — no truncation, so no value bootstrap is needed).
    2. Compute state values V(s; φ) from the critic.
    3. Advantage = returns - V(s)  (baseline-subtracted policy gradient).
    4. Actor (policy) loss:   -E[ log π(a|s; θ) · advantage ]
    5. Critic (value) loss:    E[ (V(s; φ) - returns)^2 ]
    6. Entropy bonus:          + entropy_coef · H(π(·|s; θ))
       encourages exploration; since A2C has no ε-greedy schedule,
       this is the only exploration pressure in the algorithm.
    7. Combined loss = actor_loss + value_coef · critic_loss - entropy_coef · entropy
    8. Single gradient step over BOTH actor and critic parameters
       (one joint optimizer, matching Mnih et al. 2016).

What does NOT live here
------------------------
- Rollout collection    → the agent plays one full episode and hands
                           the whole trajectory to update() as a batch
- Action sampling       → the agent samples from the masked policy
- Config object         → algorithm takes only the scalar values it uses
- Logging               → update() returns an UpdateResult; caller logs

Design note: rollout length == episode length
-----------------------------------------------
Unlike n-step/parallel-actor A2C (Mnih et al. 2016), this implementation
treats one full Minesweeper episode as one rollout. Episodes here are
short (≈10-15 steps), so this reduces to Monte-Carlo actor-critic with no
bootstrapping and no GAE (λ=1 is exact for a rollout that always reaches
a terminal state). This keeps the rest of the training loop symmetric
with DQNAgent (one episode → one update) and avoids introducing a
separate rollout buffer / cross-episode boundary logic. If the study
later wants classic fixed-length n-step A2C, update()'s batch contract
(states/actions/rewards/dones/masks + bootstrap_value) already supports
it — only the agent's collection loop would need to change.

Reference
---------
Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning"
ICML 2016. (A2C is the synchronous, single-network-update variant of
A3C described in that paper, as later popularized by OpenAI Baselines.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LRScheduler(Protocol):
    def step(self) -> None: ...


# ── Return type ────────────────────────────────────────────────────────────────


@dataclass
class UpdateResult:
    """
    Metrics produced by one A2C update (one full-episode rollout).

    Attributes:
        loss:         Combined loss (actor + value_coef*critic - entropy_coef*entropy).
        actor_loss:   Policy gradient loss alone.
        critic_loss:  Value MSE loss alone.
        entropy:      Mean policy entropy over the rollout (higher = more exploration).
        value_mean:   Mean V(s) predicted by the critic over the rollout.
        return_mean:  Mean discounted Monte-Carlo return over the rollout.
    """

    loss: float
    actor_loss: float
    critic_loss: float
    entropy: float
    value_mean: float
    return_mean: float


# ── Algorithm ──────────────────────────────────────────────────────────────────


class A2C:
    """
    Stateless (except for update counter) A2C algorithm.

    Holds the actor, critic, optimizer, and scheduler. The environment,
    rollout collection, and logging are all handled externally by the
    agent — exactly the same separation of concerns as rl.algorithms.dqn.DQN.

    Args:
        actor_net:      Policy network; outputs per-cell logits (B, n²).
        critic_net:     Value network; outputs a scalar V(s) per board (B,).
        optimizer:      Single optimizer over BOTH networks' parameters
                         (e.g. Adam(chain(actor.parameters(), critic.parameters()))).
        gamma:          Discount factor γ. Default 0.99.
        value_coef:     Weight on the critic loss in the combined loss. Default 0.5.
        entropy_coef:   Weight on the entropy bonus. Default 0.01.
        grad_clip_norm: If set, clip gradient L2-norm to this value before
                         each optimizer step. None = no clipping.
        scheduler:      Any torch LR scheduler, stepped once per update()
                         call after the optimizer step. None = constant LR.

    Example:
        >>> algo = A2C(actor, critic, opt, gamma=0.99, value_coef=0.5, entropy_coef=0.01)
        >>> result = algo.update(batch)
        >>> print(result.loss, result.entropy)
    """

    def __init__(
        self,
        actor_net: nn.Module,
        critic_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip_norm: Optional[float] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> None:
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler

        self._update_count: int = 0

    # ── Core update ────────────────────────────────────────────────────────────

    def update(self, batch: dict) -> UpdateResult:
        """
        Perform one gradient update from a single full-episode rollout.

        Batch contract (all tensors already on the target device):
            "states":          (T, *state_shape)  float32
            "actions":         (T,)               int64   — flat cell index taken each step
            "action_masks":    (T, num_actions)   bool    — legal-move mask AT THE TIME each
                                                             action was sampled (must match, so
                                                             log-probs are recomputed under the
                                                             same distribution that was sampled)
            "rewards":         (T,)               float32
            "dones":           (T,)               float32 (1.0 = terminal)
            "bootstrap_value": float — V(s_T) for the state after the last
                                        transition. 0.0 whenever the rollout
                                        ran to a true terminal state (the
                                        common case here, since rollout ==
                                        full episode).

        Returns:
            UpdateResult for this rollout.
        """
        states = batch["states"]  # (T, *state_shape)
        actions = batch["actions"]  # (T,)
        action_masks = batch["action_masks"]  # (T, num_actions) bool
        rewards = batch["rewards"]  # (T,)
        dones = batch["dones"]  # (T,)
        bootstrap_value = float(batch.get("bootstrap_value", 0.0))

        T = states.shape[0]

        # ── Monte Carlo discounted returns (backward recursion) ────────────
        returns = torch.zeros(T, dtype=torch.float32, device=states.device)
        running_return = bootstrap_value
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        # ── Critic: V(s) for every state in the rollout ─────────────────────
        values = self.critic_net(states).squeeze(-1)  # (T,)

        # ── Advantage (baseline-subtracted); detached for the actor term ───
        advantages = (returns - values).detach()

        # ── Actor: re-derive the SAME masked categorical distribution the
        #    rollout was sampled from, to get differentiable log-probs ─────
        logits = self.actor_net(states)  # (T, num_actions)
        logits = logits.masked_fill(~action_masks, float("-inf"))
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)  # (T,)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns.detach())

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # ── Gradient step (single joint optimizer over both networks) ──────
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.actor_net.parameters()) + list(self.critic_net.parameters()),
                self.grad_clip_norm,
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self._update_count += 1

        return UpdateResult(
            loss=loss.item(),
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            entropy=entropy.item(),
            value_mean=values.detach().mean().item(),
            return_mean=returns.mean().item(),
        )

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def update_count(self) -> int:
        """Total number of gradient updates performed so far."""
        return self._update_count

    @property
    def current_lr(self) -> float:
        """Current learning rate from the optimizer's first param group."""
        return self.optimizer.param_groups[0]["lr"]

    def __repr__(self) -> str:
        return (
            f"A2C("
            f"gamma={self.gamma}, "
            f"value_coef={self.value_coef}, "
            f"entropy_coef={self.entropy_coef}, "
            f"grad_clip={self.grad_clip_norm}, "
            f"scheduler={type(self.scheduler).__name__ if self.scheduler else None}, "
            f"updates={self._update_count})"
        )
