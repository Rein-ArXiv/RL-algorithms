"""
utils.py
"""

import os
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

def generate_random_map(size=4, p=0.8):
    """
    Generate a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probabililty that a tile is frozen

    Returns:
        A random valid map
    """

    valid = False

    # DFS to check if path exists
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))

        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

                for x, y in directions:
                    r_new = r + x
                    c_new = c + y

                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if res[r_new][c_new] != 'H':
                        frontier.append((r_new, c_new))
        return False
        
    while not valid:
        p = min(1, p)
        res = []
        for i in range(size):
            row = []
            for j in range(size):
                if i == 0 and j == 0:
                    row.append('S')
                elif i == size - 1 and j == size - 1:
                    row.append('G')
                elif random.random() < p:
                    row.append('F')
                else:
                    row.append('H')
            res.append(row)
        valid = is_valid(res)
    
    return ["".join(row) for row in res]

def get_model_subdir(args):
    """
    Create a model subdirectory name based on environment and map configuration
    This allows for organizing models by map size and type
    """
    env_name = args.env.split('-')[0]

    if env_name.startswith("FrozenLake"):
        size_info = f"{args.map_size}x{args.map_size}"
        slip_info = "slip" if args.is_slippery else "noslip"

        if args.static_map and args.desc:
            map_type = "custom"
        elif not args.static_map:
            map_type = "random"
        else:
            map_type = "default"
        
        return os.path.join(env_name, f"{size_info}_{map_type}_{slip_info}")

    elif env_name.startswith("Taxi"):
        env_name = args.env.split('-')[0]
        return env_name
    
    else:
        return env_name

def create_env(args):
    """
    Create and configure an environment based on args, and return env configuration
    """

    env_id = args.env
    env_config = {"env_id": env_id}

    # Basic environment configuration
    env_kwargs = {}

    # Environment-specific settings
    if env_id.startswith("FrozenLake"):
        # FrozenLake-specific configuration

        # Map configuration
        if args.static_map:
            if args.desc:
                try:
                    desc = eval(args.desc)
                    env_kwargs["desc"] = desc
                    env_config["map_type"] = "custom"
                    env_config["map_size"] = len(desc)
                except:
                    print(f"[Warning] Invalid desc format: {args.desc}. Using default map.")

            else:
                # Use default map (4x4 or 8x8)
                env_config["map_type"] = "default"
                env_config["map_size"] = args.map_size

                if args.map_size == 8:
                    env_id = "FrozenLake8x8-v1"
                    env_config["env_id"] = env_id
                else:
                    # Use default 4x4
                    pass

        else:
            random_map = generate_random_map(size=args.map_size, p=0.8)
            env_kwargs["desc"] = random_map
            env_config["map_type"] = "random"
            env_config["map_size"] = args.map_size
            env_config["generated_map"] = random_map

        # Slipperiness setting
        env_kwargs["is_slippery"] = args.is_slippery
        env_config["is_slippery"] = args.is_slippery

    elif env_id.startswith("Taxi"):
        # Taxi-specific configuration
        pass

    # Maximum episode steps (truncation condition)
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
        env_config["max_episode_steps"] = args.max_episode_steps

    # Create the enviornment
    env = gym.make(env_id, **env_kwargs)

    if args.max_episode_steps is not None and not hasattr(env, "_max_episode_steps"):
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env_config["max_episode_steps"] = args.max_episode_steps

    if hasattr(env, "_max_episode_steps"):
        env_config["actual_max_episode_steps"] = env._max_episode_steps

    return env, env_config

# 아래는 Continuous action space 일 때 사용하려고 만든 것. 사용하지 않음.
def get_state_size(env, discretization_bins=None):
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return env.observation_space.n
    
    elif isinstance(env.observation_space, gym.spaces.Box):
        if discretization_bins is None:
            raise ValueError("Requires discretisation_bins for continuous state space.")
        dims = env.observation_space.shape[0]
        return discretization_bins ** dims
    else:
        raise ValueError(f"Unsupported state space types: {type(env.observation_space)}")


def discretize_state(state, bins_per_dim, observation_space):
    if isinstance(observation_space, gym.spaces.Discrete):
        return state
    
    state_bins = []

    for i, val in enumerate(state):
        low = observation_space.low[i]
        high = observation_space.high[i]

        low = -10 if np.isinf(low) else low
        high = 10 if np.isinf(high) else high

        bin_idx = np.digitize(val, np.linspace(low, high, bins_per_dim - 1))
        state_bins.append(bin_idx)

    return np.ravel_mutli_index(state_bins, [bins_per_dim] * len(state))

def process_state(env, state, discretization_bins=None):
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return state
    
    return discretize_state(state, discretization_bins, env.observation_space)