import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Policy, self).__init__()

		self.layers = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, output_dim)
		)

		self.reset()
		self.train()

	def reset(self):
		self.log_probs = []
		self.rewards = []

	def forward(self, x):
		logit = self.layers(x)
		return logit

	def act(self, state):
		x = torch.from_numpy(state.astype(np.float32))
		logit = self.forward(x)
		prob = Categorical(logits=logit)
		action = prob.sample()
		log_prob = prob.log_prob(action)
		self.log_probs.append(log_prob)
		return action.item()

def train(policy, optimizer, gamma=0.99):
	T = len(policy.rewards)
	rets = np.empty(T, dtype=np.float32)
	future_ret = 0.0

	for t in reversed(range(T)):
		future_ret = policy.rewards[t] + gamma * future_ret
		rets[t] = future_ret
	rets = torch.tensor(rets)
	log_probs = torch.stack(policy.log_probs)
	loss = -log_probs * rets
	loss = torch.sum(loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

def main():
	env = gym.make('CartPole-v0', render_mode='rgb_array') # if mode='rgb_array', goes fast
	input_dim = env.observation_space.shape[0]
	output_dim = env.action_space.n
	policy = Policy(input_dim, output_dim)
	optimizer = optim.Adam(policy.parameters())

	for epi in range(300):
		state, info = env.reset()
		for t in range(200):
			action = policy.act(state)
			state, reward, terminated, truncated, _ = env.step(action)
			policy.rewards.append(reward)
			if (terminated or truncated):
				break
		loss = train(policy, optimizer)
		total_reward = sum(policy.rewards)
		solved = total_reward > 195.0
		policy.reset()
		print(f'Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}')

if __name__ == '__main__':
	main()
