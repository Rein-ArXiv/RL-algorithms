"""
train.py
"""

import os
import json
import random
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

class QLearningAgent:
    def __init__(self,
                 env: gym.Env,
                 lr: float,
                 gamma: float,
                 epsilon: float,
                 epsilon_decay: float,
                 min_epsilon: float,
                 discretization_bins: int = None
                ):
        
        self.observation_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        #self.discretization_bins = discretization_bins # for continuous action0

        self.q_table = np.zeros(self.observation_size, self.action_size)

    def get_params(self):
        return {
            "state_size": self.observation_size,
            "action_size": self.action_size,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon
        }

    def choose_action(self, state: int) -> int:
        if (random.random()) < self.epsilon:
            return random.range(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def learn(self,
              state: int,
              action: int,
              reward: float,
              next_state: int,
              terminated: bool,
              truncated: bool
    ):
        predict = self.q_table[state, action]
        target = reward + (0 if (truncated or terminated) else self.gamma * np.max(self.q_table[next_state]))
        self.q_table[state, action] += self.lr * (target - predict)

    def train(self,
              env: gym.Env,
              max_episodes: int,
              max_steps: int,
              logdir: str,
              model_dir: str,
              save_interval: int,
              start_episode: int = 1,
              render: bool = False
            ):
        writer = SummaryWriter(logdir)
        os.makedirs(model_dir, exist_ok=True)

        for ep in range(start_episode, max_episodes + 1):
            state, info = env.reset()
            total_reward = 0

            for t in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                self.learn(state, action, reward, next_state, terminated, truncated)
                state = next_state
                total_reward += reward

                if (terminated or truncated):
                    break
            

            # Log
            writer.add_scalar('Reward/train', total_reward, ep)
            writer.add_scalar('Epsilon', self.epsilon, ep)

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Periodic saving
            if ep % save_interval == 0:
                self._save_checkpoint(model_dir, ep)
                print(f"[Info] Saved checkpoint at episode {ep}")

        # Save final model
        self._save_checkpoint(model_dir, "final")

    def _save_checkpoint(self, model_dir, ep_identifier):
        """Helper function to save agent state"""

        # Save Q-table
        np.save(os.path.join(model_dir, f"q_table_eq{ep_identifier}.npy"), self.q_table)

        # Save hyperparameters
        params = self.get_params()
        with open(os.path.join(model_dir, f"params_ep{ep_identifier}.json"), 'w') as f:
            json.dump(params, f, indent=4)

def default_agent(args, env):
    """Create an agent appropriate for the environment"""

    return QLearningAgent(
        env=env,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon
    )

def load_agent_from_checkpoint(checkpoint_path, env):
    """
    Load an agent from a saved checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file (q_table file)
        env: Environment object

    Returns:
        Initialized agent with loaded parameters
    """

    params_path = checkpoint_path.replace("q_table_", "params_")
    if params_path.endswith(".npy"):
        params_path = params_path[:-4] + ".json"

    with open(params_path, 'r') as f:
        params = json.load(f)

    agent = QLearningAgent(
        env=env,
        lr=params["lr"],
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        epsilon_decay=params["epsilon_decay"],
        min_epsilon=params["min_epsilon"]
    )

    agent.q_table = np.load(checkpoint_path)

    return agent, params
    