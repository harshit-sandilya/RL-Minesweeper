"""
rl/algorithms/dqn.py

Deep Q-Network (DQN) update logic — Mnih et al. (2015).

What lives here
---------------
Exactly the DQN update rule — nothing else.

    1. Compute Q(s, a; θ)  using the online network
    2. Compute max Q(s′, a′; θ⁻) using the target network
       — with optional action masking if the batch contains masks
    3. Form the TD target:  r + γ · max_a′ Q(s′, a′; θ⁻) · (1 − done)
    4. MSE loss between current Q and TD target
    5. Gradient step with optional gradient clipping
    6. LR scheduler step if a scheduler is attached

What does NOT live here
-----------------------
- Observation encoding  → the network handles that (CNN / MLP / anything)
- Action selection      → the agent handles ε-greedy and masking
- Target sync timing    → the agent calls sync_target() every N steps
- Logging               → update() returns an UpdateResult; caller logs
- Config object         → algorithm takes only the scalar values it uses

This separation means DQN.update() works identically for:
    2D board games  (spatial obs + action masking)
    CartPole        (flat obs, no masking)
    Atari           (stacked pixel obs, no masking)
    Any future env  — as long as the network maps obs → Q-values

Batch contract
--------------
update() expects a dict with these keys (from ReplayBuffer.sample()):

    Required:
        "states"       : (B, *state_shape)  float32 tensor
        "actions"      : (B,)               int64   tensor
        "rewards"      : (B,)               float32 tensor
        "next_states"  : (B, *state_shape)  float32 tensor
        "dones"        : (B,)               float32 tensor  (1.0 = terminal)

    Optional:
        "next_action_masks" : (B, num_actions) bool tensor
            If present, Q-values for invalid next-state actions are masked
            to -inf before taking the max. If absent, no masking is applied.

Reference
---------
Mnih et al. (2015) "Human-level control through deep reinforcement learning"
Nature 518, 529-533.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRScheduler(Protocol):
    def step(self) -> None: ...


# ── Return type ────────────────────────────────────────────────────────────────


@dataclass
class UpdateResult:
    """
    Metrics produced by a single DQN update step.

    The caller (agent / train script) decides which of these to log,
    at what frequency, and under what key names.

    Attributes:
        loss:   MSE TD loss for this minibatch.
        q_mean: Mean Q-value of the selected actions in this batch.
                Tracks whether Q-values are growing or collapsing.
    """

    loss: float
    q_mean: float
    current_q_mean: float
    target_q_mean: float
    target_q_max: float


# ── Algorithm ──────────────────────────────────────────────────────────────────


class DQN:
    """
    Stateless (except for update counter) DQN algorithm.

    Holds the networks, optimizer, and scheduler. The environment,
    replay buffer, target-sync timing, and logging are all handled
    externally by the agent.

    Args:
        online_net:     Q-network being trained (parameters θ).
        target_net:     Frozen copy of online_net (parameters θ⁻).
                        Must be initialised as an exact copy before
                        training — the agent does this via deepcopy.
                        Sync periodically by calling sync_target().
        optimizer:      Any torch.optim.Optimizer over online_net.
        gamma:          Discount factor γ.  Default 0.99.
        grad_clip_norm: If set, clip gradient L2-norm to this value
                        before each optimizer step.  None = no clipping.
                        10.0 is a reasonable starting value if training
                        is unstable.
        scheduler:      Any torch LR scheduler.  Stepped once per
                        update() call after the optimizer step.
                        None = constant LR.

    Example:
        >>> online = CNNQNetwork(input_shape=(1,8,8), num_actions=64)
        >>> target = copy.deepcopy(online)
        >>> opt    = torch.optim.Adam(online.parameters(), lr=1e-4)
        >>> algo   = DQN(online, target, opt, gamma=0.99)
        >>> result = algo.update(batch)
        >>> print(result.loss, result.q_mean)
    """

    def __init__(
        self,
        online_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        grad_clip_norm: Optional[float] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> None:
        self.online_net = online_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler

        self._update_count: int = 0

        # Target network is never trained directly — disable gradients
        for p in self.target_net.parameters():
            p.requires_grad_(False)

    # ── Core update ────────────────────────────────────────────────────────────

    def update(self, batch: dict) -> UpdateResult:
        """
        Perform one gradient update step on online_net.

        Args:
            batch: dict from ReplayBuffer.sample(). See module docstring
                   for the required and optional keys.

        Returns:
            UpdateResult(loss, q_mean) for this batch.
            The caller reads these and logs them; update() has no side
            effects beyond the gradient step and scheduler step.
        """
        states = batch["states"]  # (B, *state_shape)
        actions = batch["actions"]  # (B,)
        rewards = batch["rewards"]  # (B,)
        next_states = batch["next_states"]  # (B, *state_shape)
        dones = batch["dones"]  # (B,)

        # ── Current Q-values: Q(s, a; θ) ─────────────────────────────────
        # online_net produces Q-values for ALL actions; gather selects
        # only the action that was actually taken.
        all_q = self.online_net(states)  # (B, num_actions)
        current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # ── TD target: r + γ · max_a′ Q(s′, a′; θ⁻) · (1 − done) ───────
        with torch.no_grad():
            next_q = self.target_net(next_states)  # (B, num_actions)
            if "next_action_masks" in batch:
                next_q = next_q.masked_fill(~batch["next_action_masks"], -1e8)

            next_q_max = next_q.max(dim=1)[0]  # (B,)
            targets = rewards + self.gamma * next_q_max * (1.0 - dones)

        # ── Loss ──────────────────────────────────────────────────────────
        loss = F.smooth_l1_loss(current_q, targets)

        # ── Gradient step ─────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self._update_count += 1

        return UpdateResult(
            loss=loss.item(),
            q_mean=current_q.detach().mean().item(),
            current_q_mean=current_q.detach().mean().item(),
            target_q_mean=targets.detach().mean().item(),
            target_q_max=targets.detach().max().item(),
        )

    # ── Target network sync ────────────────────────────────────────────────────

    def sync_target(self) -> None:
        """
        Hard copy online_net weights → target_net.

        The agent calls this every target_update_freq steps.
        The algorithm owns the copy; the agent owns the timing.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def update_count(self) -> int:
        """Total number of gradient updates performed so far."""
        return self._update_count

    @property
    def current_lr(self) -> float:
        """Current learning rate from the optimizer's first param group.
        Works with or without a scheduler."""
        return self.optimizer.param_groups[0]["lr"]

    def __repr__(self) -> str:
        return (
            f"DQN("
            f"gamma={self.gamma}, "
            f"grad_clip={self.grad_clip_norm}, "
            f"scheduler={type(self.scheduler).__name__ if self.scheduler else None}, "
            f"updates={self._update_count})"
        )
