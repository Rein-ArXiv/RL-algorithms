Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)


# Q-function
- Bellman equation
$$Q^\pi_\text{DQN}(s,a) \approx r + \gamma \max_{a'} Q^\pi(s', a')$$
$$Q^\pi_\text{tar:DQN}(s,a) = r + \gamma \max_{a'} Q^\pi(s', a')$$

