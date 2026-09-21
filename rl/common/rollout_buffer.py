"""
rl/common/rollout_buffer.py

On-policy rollout storage for PPO.

NOT the replay buffer
----------------------
Deliberately separate from rl.common.replay_buffer.ReplayBuffer:
    - ReplayBuffer:  off-policy, persists across many updates, randomly
                     sampled, holds transitions from many different
                     (stale) policies at once.
    - RolloutBuffer: on-policy, built fresh every rollout, discarded
                     immediately after the epochs of updates that
                     consume it, every transition came from the CURRENT
                     policy at collection time.
Nothing in replay_buffer.py is touched or reused here.

What it stores
---------------
One rollout = a fixed number of FULL episodes (not a fixed step count).
Every episode runs to a true terminal state, so advantage computation
never needs to bootstrap across a truncated episode boundary -- only
within an episode, via GAE(lambda). This mirrors the same simplifying
choice rl.algorithms.a2c already makes (see its module docstring).

Usage
-----
    buf = RolloutBuffer()
    for _ in range(rollout_episodes):
        ... play one episode, collecting per-step arrays ...
        buf.add_episode(states, actions, old_log_probs, values, rewards, dones, masks)
    buf.compute_advantages(gamma, gae_lambda)
    for epoch in range(ppo_epochs):
        for minibatch in buf.minibatches(minibatch_size, device):
            algo.update(minibatch)
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import torch


class RolloutBuffer:
    """Accumulates full episodes, then exposes them as GAE-annotated minibatches."""

    def __init__(self) -> None:
        self._episodes: List[dict] = []
        self._flat: Optional[dict] = None  # populated by compute_advantages()

    def add_episode(
        self,
        states: np.ndarray,  # (T, *state_shape)
        actions: np.ndarray,  # (T,)
        old_log_probs: np.ndarray,  # (T,)
        values: np.ndarray,  # (T,) — V(s_t) at collection time
        rewards: np.ndarray,  # (T,)
        dones: np.ndarray,  # (T,) — last entry is always 1.0 (true terminal)
        action_masks: np.ndarray,  # (T, num_actions) bool
    ) -> None:
        assert dones[-1] == 1.0, "RolloutBuffer only accepts full (terminated) episodes"
        self._episodes.append(
            {
                "states": states,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "values": values,
                "rewards": rewards,
                "dones": dones,
                "action_masks": action_masks,
            }
        )
        self._flat = None  # invalidate cache

    def __len__(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return sum(len(ep["rewards"]) for ep in self._episodes)

    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        """
        GAE(lambda) advantage + return computation, per episode — never
        bootstrapping across episode boundaries, since every episode here
        ends at a true terminal state (bootstrap value there is always 0).
        """
        all_advantages, all_returns = [], []

        for ep in self._episodes:
            rewards = ep["rewards"]
            values = ep["values"]
            T = len(rewards)

            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                next_value = values[t + 1] if t + 1 < T else 0.0  # terminal -> 0
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages[t] = gae

            returns = advantages + values
            all_advantages.append(advantages)
            all_returns.append(returns)

        self._flat = {
            "states": np.concatenate([ep["states"] for ep in self._episodes], axis=0),
            "actions": np.concatenate([ep["actions"] for ep in self._episodes], axis=0),
            "old_log_probs": np.concatenate(
                [ep["old_log_probs"] for ep in self._episodes], axis=0
            ),
            "action_masks": np.concatenate(
                [ep["action_masks"] for ep in self._episodes], axis=0
            ),
            "advantages": np.concatenate(all_advantages, axis=0),
            "returns": np.concatenate(all_returns, axis=0),
            "old_values": np.concatenate(
                [ep["values"] for ep in self._episodes], axis=0
            ),
        }

        # Advantage normalization — standard PPO practice, keeps the
        # clipped objective's scale consistent across rollouts.
        adv = self._flat["advantages"]
        self._flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def minibatches(
        self, minibatch_size: int, device, shuffle: bool = True
    ) -> Iterator[dict]:
        """
        Yields dicts of tensors on `device`, shuffled and split into
        minibatches. Call compute_advantages() first.
        """
        if self._flat is None:
            raise RuntimeError("Call compute_advantages() before minibatches().")

        n = len(self._flat["actions"])
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, n, minibatch_size):
            b = idx[start : start + minibatch_size]
            yield {
                "states": torch.from_numpy(self._flat["states"][b]).float().to(device),
                "actions": torch.from_numpy(self._flat["actions"][b]).long().to(device),
                "old_log_probs": torch.from_numpy(self._flat["old_log_probs"][b])
                .float()
                .to(device),
                "action_masks": torch.from_numpy(self._flat["action_masks"][b]).to(
                    device
                ),
                "advantages": torch.from_numpy(self._flat["advantages"][b])
                .float()
                .to(device),
                "returns": torch.from_numpy(self._flat["returns"][b])
                .float()
                .to(device),
                "old_values": torch.from_numpy(self._flat["old_values"][b])
                .float()
                .to(device),
            }

    def clear(self) -> None:
        self._episodes = []
        self._flat = None

    def __repr__(self) -> str:
        return (
            f"RolloutBuffer(episodes={len(self)}, transitions={self.num_transitions})"
        )
