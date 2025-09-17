from __future__ import annotations

import argparse
import os
from typing import Optional

from .config import build_dqn_config
from .train import train
from .rollout import collect_rollouts


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ivalue


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CartPole RL trainer and recorder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_p = subparsers.add_parser("train", help="Train DQN on CartPole")
    train_p.add_argument("--iterations", type=positive_int, default=100)
    train_p.add_argument("--save-dir", type=str, default=".")

    # Rollout command
    roll_p = subparsers.add_parser("rollout", help="Run evaluation rollouts and save JSON")
    roll_p.add_argument("--checkpoint-dir", type=str, required=True)
    roll_p.add_argument("--episodes", type=positive_int, default=10)
    roll_p.add_argument("--output", type=str, default="rollout_data.json")

    args = parser.parse_args(argv)

    if args.command == "train":
        metrics = train(config_builder=build_dqn_config, iterations=args.iterations, save_dir=args.save_dir)
        print(f"Training done. Mean eval return: {metrics['mean_eval_return']}")
        return

    if args.command == "rollout":
        # Load agent from saved checkpoint directory
        from ray.rllib.algorithms.algorithm import Algorithm

        agent = Algorithm.from_checkpoint(args.checkpoint_dir)
        data = collect_rollouts(agent=agent, num_episodes=args.episodes, save_path=args.output)
        print(f"Saved {len(data)} episodes to {os.path.abspath(args.output)}")
        return


if __name__ == "__main__":
    main()


