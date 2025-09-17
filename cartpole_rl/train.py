from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import ray

from ray.rllib.algorithms.algorithm import Algorithm


def train(
    *,
    config_builder,
    iterations: int = 100,
    save_dir: Optional[str] = None,
    suppress_logs: bool = True,
) -> Dict[str, float]:
    """Train an RLlib Algorithm and optionally save it.

    Args:
        config_builder: Callable returning an RLlib config with ``build_algo``.
        iterations: Number of training iterations.
        save_dir: Directory to save the trained agent and metadata JSON.
        suppress_logs: If True, reduce Ray verbosity.

    Returns:
        Dict with summary metrics (e.g., last iteration mean reward).
    """

    if suppress_logs:
        logging.getLogger("ray").setLevel(logging.ERROR)

    # Initialize Ray lazily if not already running
    if not ray.is_initialized():
        ray.init(logging_level=logging.ERROR)

    config = config_builder()
    agent: Algorithm = config.build_algo()

    last_mean_eval_reward = float("nan")
    for _ in range(iterations):
        results = agent.train()
        try:
            last_mean_eval_reward = results["evaluation"]["env_runners"][
                "episode_return_mean"
            ]
        except Exception:
            # Not all iterations may include evaluation results
            pass

    metrics = {"mean_eval_return": last_mean_eval_reward}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        agent.save_to_path(save_dir)
        with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


