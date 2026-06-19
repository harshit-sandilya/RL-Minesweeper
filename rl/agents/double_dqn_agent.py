"""
rl/agents/double_dqn_agent.py

Double DQN agent for Minesweeper.
Inherits all environment interaction, logging, and checkpointing logic
from DQNAgent, but replaces the underlying algorithm with DoubleDQN.
"""

from __future__ import annotations

from rl.agents.dqn_agent import DQNAgent
from rl.algorithms.double_dqn import DoubleDQN
from rl.common.config import MinesweeperConfig


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent that trains on the Minesweeper environment.
    Uses DoubleDQN logic internally to reduce Q-value overestimation bias.
    """

    def __init__(self, config: MinesweeperConfig, env) -> None:
        super().__init__(config, env)

        self.algo = DoubleDQN(
            online_net=self.online_net,
            target_net=self.target_net,
            optimizer=self.optimizer,
            gamma=config.gamma,
            grad_clip_norm=10.0,
            scheduler=self.lr_scheduler,
        )

    def __repr__(self) -> str:
        return (
            f"DoubleDQNAgent("
            f"n={self.config.n}, "
            f"mines={self.config.mines}, "
            f"device={self.device}, "
            f"episode={self._episode}, "
            f"step={self._global_step})"
        )
