from __future__ import annotations

import argparse
import os

from cartpole_rl.config import build_dqn_config
from cartpole_rl.train import train
from cartpole_rl.rollout import collect_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and record CartPole rollouts")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="rollout_data.json")
    args = parser.parse_args()

    save_dir = os.getcwd()

    print("Training agent...")
    metrics = train(config_builder=build_dqn_config, iterations=args.iterations, save_dir=save_dir)
    print(f"Training complete. Mean eval return: {metrics['mean_eval_return']}")

    # Load agent from checkpoint directory (current dir used by train save)
    from ray.rllib.algorithms.algorithm import Algorithm

    agent = Algorithm.from_checkpoint(save_dir)
    print("Collecting rollouts...")
    _ = collect_rollouts(agent=agent, num_episodes=args.episodes, save_path=args.output)
    print(f"Rollout data saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
