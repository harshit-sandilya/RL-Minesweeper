"""
rl/algorithms/ppo.py

Proximal Policy Optimization (PPO, clipped variant) update logic.

What lives here
---------------
Exactly the PPO-clip update rule, applied to ONE minibatch at a time:

    1. Recompute the policy's log-prob and value estimate for the states
       in this minibatch, under the CURRENT (being-updated) parameters.
    2. Importance ratio: r = exp(new_log_prob - old_log_prob), where
       old_log_prob was captured at rollout-collection time, before any
       of this update epoch's gradient steps touched the policy.
    3. Clipped surrogate objective:
           L_actor = -E[ min(r * A, clip(r, 1-eps, 1+eps) * A) ]
       This is what keeps PPO's updates inside a trust region without
       needing the second-order machinery TRPO uses for the same purpose.
    4. Critic loss: MSE against the GAE-computed returns, optionally
       clipped the same way the original PPO paper's reference
       implementation clips it.
    5. Entropy bonus, same role as in A2C: PPO has no epsilon-schedule
       either, so this is the exploration pressure.
    6. loss = actor_loss + value_coef*critic_loss - entropy_coef*entropy
    7. One gradient step over BOTH actor and critic (single joint
       optimizer, same convention as rl.algorithms.a2c).

What does NOT live here
------------------------
- Rollout collection            → the agent plays rollout_episodes full
                                   episodes and hands them to RolloutBuffer
- GAE / advantage computation   → rl.common.rollout_buffer.RolloutBuffer
- Minibatching / shuffling      → rl.common.rollout_buffer.RolloutBuffer
- The multi-epoch outer loop    → the agent calls update() once per
                                   minibatch, ppo_epochs * num_minibatches
                                   times per rollout
- Config object                 → algorithm takes only the scalars it uses

Relationship to rl.algorithms.a2c
-----------------------------------
PPO here is A2C's clipped, multi-epoch, importance-corrected cousin: same
actor/critic network shapes, same joint optimizer convention, same
entropy-bonus role. The two things PPO adds that A2C doesn't have are
(a) the clipped ratio, which lets it safely reuse a rollout for several
gradient epochs instead of one, and (b) GAE(lambda) instead of raw Monte
Carlo returns, trading a small amount of bias for lower variance.

Reference
---------
Schulman et al. (2017) "Proximal Policy Optimization Algorithms",
arXiv:1707.06347. Advantage computation follows Schulman et al. (2016)
"High-Dimensional Continuous Control Using Generalized Advantage
Estimation", arXiv:1506.02438.
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
    Metrics produced by one PPO minibatch update.

    Attributes:
        loss:          Combined loss for this minibatch.
        actor_loss:    Clipped surrogate policy loss alone.
        critic_loss:   Value loss alone.
        entropy:       Mean policy entropy over the minibatch.
        approx_kl:     Approximate KL divergence between old and new
                        policy on this minibatch — the standard PPO
                        diagnostic for "did this epoch move the policy
                        too far"; large spikes indicate clip_epsilon or
                        ppo_epochs needs tuning.
        clip_fraction: Fraction of ratios that were actually clipped —
                        near 0 means the trust region isn't binding;
                        near 1 means updates are being aggressively
                        constrained.
    """

    loss: float
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


# ── Algorithm ──────────────────────────────────────────────────────────────────


class PPO:
    """
    Stateless (except for update counter) PPO-clip algorithm, operating on
    one minibatch per update() call. The agent's outer loop calls update()
    ppo_epochs * num_minibatches times per rollout, re-shuffling
    minibatches each epoch (see rl.common.rollout_buffer.RolloutBuffer).

    Args:
        actor_net:       Policy network; outputs per-cell logits (B, n²).
        critic_net:      Value network; outputs a scalar V(s) per board (B,).
        optimizer:       Single optimizer over BOTH networks' parameters.
        clip_epsilon:    PPO's trust-region clip range. Default 0.2.
        value_coef:      Weight on critic loss. Default 0.5.
        entropy_coef:    Weight on entropy bonus. Default 0.01.
        clip_value_loss: Also clip the value loss, as in the original
                         PPO paper's reference implementation. Default True.
        grad_clip_norm:  If set, clip gradient L2-norm before each step.
        scheduler:       Any torch LR scheduler, stepped once per update()
                         call. None = constant LR.

    Example:
        >>> algo = PPO(actor, critic, opt, clip_epsilon=0.2)
        >>> for epoch in range(ppo_epochs):
        ...     for minibatch in buf.minibatches(64, device):
        ...         result = algo.update(minibatch)
    """

    def __init__(
        self,
        actor_net: nn.Module,
        critic_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        clip_value_loss: bool = True,
        grad_clip_norm: Optional[float] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> None:
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.clip_value_loss = clip_value_loss
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler

        self._update_count: int = 0

    # ── Core update (one minibatch) ─────────────────────────────────────────────

    def update(self, batch: dict) -> UpdateResult:
        """
        Perform one gradient update from one minibatch.

        Batch contract (exactly what RolloutBuffer.minibatches() yields):
            "states":        (B, *state_shape) float32
            "actions":       (B,)              int64
            "action_masks":  (B, num_actions)   bool    — mask at rollout-collection time
            "old_log_probs": (B,)               float32 — under the rollout-time policy
            "advantages":    (B,)               float32 — GAE, already normalized
            "returns":       (B,)               float32 — GAE returns (advantages + old_values)
            "old_values":    (B,)               float32 — V(s) at rollout-collection time
                                                            (used only if clip_value_loss)
        """
        states = batch["states"]
        actions = batch["actions"]
        action_masks = batch["action_masks"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["old_values"]

        # ── Actor: current policy's log-prob/entropy for these (state, action) ──
        logits = self.actor_net(states)
        logits = logits.masked_fill(~action_masks, float("-inf"))
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # ── Clipped surrogate objective ─────────────────────────────────────
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate_1 = ratio * advantages
        surrogate_2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

        # ── Critic: optionally clipped the same way as the original PPO paper ──
        values = self.critic_net(states).squeeze(-1)
        if self.clip_value_loss:
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            critic_loss = torch.max(
                F.mse_loss(values, returns, reduction="none"),
                F.mse_loss(values_clipped, returns, reduction="none"),
            ).mean()
        else:
            critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # ── Gradient step ─────────────────────────────────────────────────
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

        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = (
                ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
            )

        return UpdateResult(
            loss=loss.item(),
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            entropy=entropy.item(),
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def __repr__(self) -> str:
        return (
            f"PPO("
            f"clip_epsilon={self.clip_epsilon}, "
            f"value_coef={self.value_coef}, "
            f"entropy_coef={self.entropy_coef}, "
            f"clip_value_loss={self.clip_value_loss}, "
            f"grad_clip={self.grad_clip_norm}, "
            f"updates={self._update_count})"
        )
