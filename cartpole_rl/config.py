from __future__ import annotations

from typing import List, Optional

from ray.rllib.algorithms.dqn import DQNConfig


def build_dqn_config(
    *,
    env_id: str = "CartPole-v1",
    learning_rate: float = 5e-4,
    fc_hiddens: Optional[List[int]] = None,
    activation: str = "tanh",
    num_env_runners: int = 4,
    num_envs_per_runner: int = 2,
    evaluation_episodes: int = 10,
    evaluation_interval: int = 1,
) -> DQNConfig:
    """Create a preconfigured DQNConfig for CartPole.

    Returns a configured RLlib DQNConfig. Call ``config.build_algo()`` to
    construct the Algorithm.
    """

    if fc_hiddens is None:
        fc_hiddens = [256, 256]

    config = DQNConfig()
    config.training(lr=learning_rate)
    config.environment(env=env_id)
    config.env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_runner,
    )
    config.evaluation(
        evaluation_config={"explore": False},
        evaluation_duration=evaluation_episodes,
        evaluation_interval=evaluation_interval,
        evaluation_duration_unit="episodes",
    )
    config.rl_module(
        model_config={
            "fc_hiddens": fc_hiddens,
            "fcnet_activation": activation,
        }
    )
    return config


