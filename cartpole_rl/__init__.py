"""CartPole RL package.

Provides utilities to configure, train, evaluate, and record rollouts for
Gymnasium's CartPole using Ray RLlib.
"""

from .config import build_dqn_config
from .train import train
from .rollout import collect_rollouts

__all__ = [
    "build_dqn_config",
    "train",
    "collect_rollouts",
]


