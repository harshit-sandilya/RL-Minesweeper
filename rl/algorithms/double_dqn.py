"""
rl/algorithms/double_dqn.py

Double Deep Q-Network (DDQN) update logic — van Hasselt et al. (2015).

Inherits from standard DQN but modifies the TD target calculation to decouple
action selection from action evaluation, reducing overestimation bias.

1. Select action a' for next_state using the online network.
2. Evaluate action a' using the target network.
3. Uses Huber loss (smooth_l1_loss) instead of MSE for better gradient stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.algorithms.dqn import DQN, UpdateResult


class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    Decouples action selection (online network) from evaluation (target network).
    """

    def update(self, batch: dict) -> UpdateResult:
        """
        Perform one Double DQN gradient update step.
        """
        states = batch["states"]  # (B, *state_shape)
        actions = batch["actions"]  # (B,)
        rewards = batch["rewards"]  # (B,)
        next_states = batch["next_states"]  # (B, *state_shape)
        dones = batch["dones"]  # (B,)

        # ── Current Q-values: Q(s, a; θ) ─────────────────────────────────
        all_q = self.online_net(states)  # (B, num_actions)
        current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Double DQN TD target ─────────────────────────────────────────
        with torch.no_grad():
            # 1. Action SELECTION: argmax_a' Q(s', a'; θ) using ONLINE network
            next_q_online = self.online_net(next_states)

            # Apply action masking to online network's output if available
            if "next_action_masks" in batch:
                next_q_online = next_q_online.masked_fill(
                    ~batch["next_action_masks"], -1e8
                )

            best_next_actions = next_q_online.argmax(dim=1)  # (B,)

            # 2. Action EVALUATION: Q(s', a'; θ⁻) using TARGET network
            next_q_target = self.target_net(next_states)

            # Evaluate the specific actions selected by the online network
            next_q_values = next_q_target.gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)  # (B,)

            # r + γ * Q_target(s', argmax Q_online) * (1 - done)
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        # ── Loss: Huber Loss (Smooth L1) ──────────────────────────────────
        # Less sensitive to outliers than MSE loss, highly recommended for DDQN
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

    def __repr__(self) -> str:
        return (
            f"DoubleDQN("
            f"gamma={self.gamma}, "
            f"grad_clip={self.grad_clip_norm}, "
            f"scheduler={type(self.scheduler).__name__ if self.scheduler else None}, "
            f"updates={self._update_count})"
        )
