"""
main.py
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym

from utils import create_env, get_model_subdir

# Look page(https://gymnasium.farama.org/environments/toy_text/frozen_lake/) to understand default env.


def parse_args():
    parser = argparse.ArgumentParser(description="Q-Learning for discrete environments")
    parser.add_argument("--env", type=str, default="FrozenLake-v1",
                        help="Gym environment name (e.g., FrozenLake-v1, Taxi-v3)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during training/testing")


    # FrozenLake-specific parameters
    parser.add_argument("--desc", type=str, default=None,
                        help="Custom map structure (for FrozenLake)")
    parser.add_argument("--map_size", type=int, default=4,
                        help="Map size for FrozenLake (creates NxN grid)")
    parser.add_argument("--static_map", action="store_true", default=True,
                        help="Use default/custom map instead of generating random one")
    parser.add_argument("--is_slippery", action="store_true",
                        help="Whether the ice is slippery (for FrozenLake)")

    # Learning parameters
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate alpha")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor gamma")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial epsilon-greedy parameter")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Epsilon decay rate per episode")
    parser.add_argument("--min_epsilon", type=float, default=0.01,
                        help="Minimum epsilon value")

    # Mode setting
    parser.add_argument("--mode", choices=["train", "resume", "test"], default="train",
                        help="Running mode (train/resume/test)")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of training/testing episodes")
    parser.add_argument("--max_episode_steps", type=int, default=None,
                        help="Maximum steps per episode during agent operation")

    # Checkpoint arguments
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint to load (episode number or 'final')")
    parser.add_argument("--resume_episode", type=int, default=None,
                        help="Episode to start from when resuming (default: infer from checkpoint)")

    # Log and model path
    parser.add_argument("--logdir", type=str, default="./logs",
                        help="TensorBoard log directory")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Base directory to save/load models")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Episodes between model saves")

    return parser.parse_args()
    
def main():
    args = parse_args()

    env, env_config = create_env(args)

    # Get model subdirectory based on environment and configuration
    model_subdir = get_model_subdir(args)
    full_model_dir = os.path.join(args.model_dir, model_subdir)
    full_log_dir = os.path.join(args.logdir, model_subdir)

    # Create directories
    os.makedirs(full_model_dir, exist_ok=True)
    os.makedirs(full_log_dir, exist_ok=True)

    # Save environ,ent configuration for reference
    with open(os.path.join(full_model_dir, "env_config.json"), 'w') as f:
        json.dump(env_config, f, indent=4)

    # Display environment info
    print(f"[Info] Enviornment: {args.env}")
    print(f"[Info] State space: {env.observation_spaace}, Action space: {env.action_space}")
    print(f"[Info] Model directory: {full_model_dir}")

    # FrozenLake-specific info
    if args.env.startswith("FrozenLake"):
        print(f"[Info] Map size: {args.map_size}x{args.map_size}")
        print(f"[Info] Slippery: {args.is_slippery}")

        if env_config.get("map_type") == "random":
            print(f"[Info] Using random map")

        elif env_config.get("map_type") == "custom":
            print(f"[Info] Using custom map: {args.desc}")
        else:
            print(f"[Info] Using default map")

    if args.mode == "train":
        agent = 

    if args.env.startwith("FrozenLake"):
        print(f"[Info] Slippery: {args.is_slippery}")  