from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from ray.rllib.algorithms.algorithm import Algorithm


def collect_rollouts(
    *,
    agent: Algorithm,
    env_id: str = "CartPole-v1",
    num_episodes: int = 10,
    render_mode: Optional[str] = None,
    save_path: Optional[str] = None,
    include_iteration_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run evaluation episodes and return structured rollout data.

    Optionally saves the data to JSON if ``save_path`` is provided.
    """

    env = gym.make(env_id, render_mode=render_mode or "rgb_array")
    rl_module = agent.get_module()

    rollouts: List[Dict[str, Any]] = []
    for episode_index in range(num_episodes):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0
        steps: List[Dict[str, Any]] = []

        while not done:
            obs_batch = torch.from_numpy(np.asarray(observation)).unsqueeze(0)
            action = rl_module.forward_inference({"obs": obs_batch})["actions"].numpy()[
                0
            ]
            next_obs, reward, terminated, truncated, info = env.step(action=action)

            done = bool(terminated or truncated)
            total_reward += float(reward)

            step_record = {
                "observation": np.asarray(observation).tolist(),
                "action": np.asarray(action).tolist(),
                "reward": float(reward),
                "next_observation": np.asarray(next_obs).tolist(),
                "done": done,
            }
            if include_iteration_id is not None:
                step_record["training_iteration"] = int(include_iteration_id)
            steps.append(step_record)

            observation = next_obs

        rollouts.append(
            {
                "episode_id": episode_index + 1,
                "total_reward": total_reward,
                "steps": steps,
            }
        )

    env.close()

    if save_path:
        with open(save_path, "w") as fp:
            json.dump(rollouts, fp, indent=2)

    return rollouts


