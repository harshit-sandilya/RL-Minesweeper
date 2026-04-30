"""
rl/common/replay_buffer.py

Problem-agnostic, shape-agnostic experience replay buffer.

Design decisions
----------------
state_shape : tuple
    Shape of a single observation, NOT the batch shape.
    Any valid NumPy array shape works:

        (64,)       flat vector (MLP input)
        (1, 8, 8)   single-channel 2D board (CNN, CHW)
        (4, 84, 84) stacked Atari frames (CNN, CHW)
        (18,)       CartPole / any flat continuous state

obs_dtype : np.dtype (default float32)
    Dtype of stored state arrays.  Set to np.uint8 for pixel observations
    (e.g. Atari) to cut memory usage by 4× compared to float32.
    The agent is responsible for casting to float before the network
    forward pass (trivial: tensor.float() / 255.0).

use_action_mask : bool (default False)
    If True, a boolean mask buffer of shape (capacity, action_dim) is
    allocated and push() expects next_action_mask to be provided.
    If False, no mask memory is allocated and mask args are ignored.
    Algorithms check for "next_action_masks" in the batch dict —
    it only appears when use_action_mask=True.

action_dim : int
    Number of discrete actions.  Used to size the actions and mask buffers.

Memory footprint (float32, no mask)
------------------------------------
    capacity=100_000, state_shape=(1,8,8):
        states:      100_000 × 64 × 4 =  25.6 MB  ×2 for next_states
        actions:     100_000 × 8       =   0.8 MB
        rewards:     100_000 × 4       =   0.4 MB
        dones:       100_000 × 4       =   0.4 MB
        total:                         ~  52.8 MB

    With use_action_mask=True, action_dim=64:
        masks:       100_000 × 64 × 1  =   6.4 MB  extra
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Uniform-random experience replay buffer.

    Stores transitions as NumPy arrays and returns PyTorch tensor batches.
    Fully agnostic to environment, network architecture, and observation
    shape — all shape/dtype decisions are made at construction time.

    Args:
        capacity:        Maximum number of transitions to store.
                         Oldest transitions are overwritten once full.
        state_shape:     Shape of a single observation, e.g. (64,) or (1,8,8).
        action_dim:      Number of discrete actions.
        obs_dtype:       NumPy dtype for state storage.  Use np.float32 for
                         normalised observations, np.uint8 for raw pixels.
        use_action_mask: If True, allocates a bool mask buffer and expects
                         next_action_mask in every push() call.

    Example — 2D board game (CNN, spatial obs, with masking):
        >>> buf = ReplayBuffer(
        ...     capacity=100_000,
        ...     state_shape=(1, 8, 8),
        ...     action_dim=64,
        ...     obs_dtype=np.float32,
        ...     use_action_mask=True,
        ... )
        >>> buf.push(obs.spatial, action, reward, next_obs.spatial,
        ...          done, next_obs.action_mask)

    Example — CartPole (no masking, flat state):
        >>> buf = ReplayBuffer(
        ...     capacity=10_000,
        ...     state_shape=(4,),
        ...     action_dim=2,
        ... )
        >>> buf.push(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        action_dim: int,
        obs_dtype: np.dtype = np.dtype(np.float32),
        use_action_mask: bool = False,
    ) -> None:
        assert capacity > 0, f"capacity must be > 0, got {capacity}"
        assert len(state_shape) >= 1, "state_shape must have at least one dimension"
        assert action_dim > 0, f"action_dim must be > 0, got {action_dim}"

        self._capacity = capacity
        self._state_shape = state_shape
        self._action_dim = action_dim
        self._obs_dtype = np.dtype(obs_dtype)
        self._use_action_mask = use_action_mask

        self._ptr = 0  # next write position
        self._size = 0  # current number of stored transitions

        # ── Preallocate all buffers ────────────────────────────────────────
        self.states = np.zeros((capacity, *state_shape), dtype=obs_dtype)
        self.next_states = np.zeros((capacity, *state_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Only allocate mask memory when explicitly requested
        self.next_action_masks: Optional[np.ndarray] = (
            np.zeros((capacity, action_dim), dtype=bool) if use_action_mask else None
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store a single transition.

        Args:
            state:            Observation array.  Shape must match state_shape.
            action:           Integer action index in [0, action_dim).
            reward:           Scalar reward signal.
            next_state:       Next observation.  Shape must match state_shape.
            done:             True if the episode ended after this transition.
            next_action_mask: Boolean mask of valid actions in next_state.
                              Required when use_action_mask=True, ignored otherwise.
                              Shape: (action_dim,) bool.
        """
        if self._use_action_mask and next_action_mask is None:
            raise ValueError(
                "ReplayBuffer was created with use_action_mask=True "
                "but push() was called without next_action_mask."
            )

        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        if self._use_action_mask and self.next_action_masks is not None:
            self.next_action_masks[idx] = next_action_mask

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    # ── Read ───────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, device: torch.device) -> dict:
        """
        Sample a random batch of transitions.

        Returns:
            dict with keys:
                "states"             : (B, *state_shape) tensor, obs_dtype cast to float32
                "actions"            : (B,)              int64 tensor
                "rewards"            : (B,)              float32 tensor
                "next_states"        : (B, *state_shape) float32 tensor
                "dones"              : (B,)              float32 tensor
                "next_action_masks"  : (B, action_dim)   bool tensor
                                       — ONLY present when use_action_mask=True

        Note: states and next_states are always returned as float32 regardless
        of obs_dtype.  For uint8 pixel buffers the cast is lossless; the agent
        or algorithm is responsible for further normalisation (e.g. / 255.0).
        """
        assert batch_size <= self._size, (
            f"Requested batch_size={batch_size} but buffer only has {self._size} transitions."
        )

        idxs = np.random.randint(0, self._size, size=batch_size)

        batch = {
            "states": torch.from_numpy(self.states[idxs].astype(np.float32)).to(device),
            "actions": torch.from_numpy(self.actions[idxs]).to(device),
            "rewards": torch.from_numpy(self.rewards[idxs]).to(device),
            "next_states": torch.from_numpy(
                self.next_states[idxs].astype(np.float32)
            ).to(device),
            "dones": torch.from_numpy(self.dones[idxs]).to(device),
        }

        if self._use_action_mask and self.next_action_masks is not None:
            batch["next_action_masks"] = torch.from_numpy(
                self.next_action_masks[idxs]
            ).to(device)

        return batch

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    @property
    def is_ready(self) -> bool:
        """Alias: True once at least one transition has been stored."""
        return self._size > 0

    def memory_mb(self) -> float:
        """Approximate memory footprint of all buffers in megabytes."""
        total_bytes = (
            2
            * self._capacity
            * int(np.prod(self._state_shape))
            * self._obs_dtype.itemsize
            + self._capacity * 8  # actions (int64)
            + self._capacity * 4  # rewards (float32)
            + self._capacity * 4  # dones   (float32)
        )
        if self._use_action_mask:
            total_bytes += self._capacity * self._action_dim * 1  # bool = 1 byte
        return total_bytes / (1024**2)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer("
            f"capacity={self._capacity:,}, "
            f"size={self._size:,}, "
            f"state_shape={self._state_shape}, "
            f"obs_dtype={self._obs_dtype}, "
            f"action_dim={self._action_dim}, "
            f"use_action_mask={self._use_action_mask}, "
            f"memory≈{self.memory_mb():.1f}MB)"
        )
