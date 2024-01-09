# Core ideas
1. 시간차 학습 (Temporal difference, TD)
2. Q 함수 (Q-function)
***
- Bellman equation (TD)

$$Q^\pi(s,a) = \mathbb{E}_{(s'|s,a), r\sim \mathcal{R}(s,a,s')}\left[r + \gamma \mathbb{E}_{a' \sim \pi(s')}[Q^\pi(s', a')]\right]$$
	- Current $Q$-value가 next $Q$-value를 통해 정의된다.
- Next state의 probability를 고려하지 않을 경우
$$Q^\pi(s, a) = r + \gamma \mathbb{E}_{a' \sim \pi(s')} \left[Q^\pi(s', a')\right]$$
***
> State $s$에서 시작하는 $N$개의 trajectory $\tau_i \ (i \in \{1, \dots, N \})$가 주어졌을 때 $Q^\pi_\text{target}(s, a)$에 대한 Monte-Carlo estimation
> $$Q^\pi_{\text{target:MC}} (s, a) = {1 \over N} \sum^N_{i=1} R(\tau_i)$$
> - 하나의 episode가 끝나야 학습하므로, training이 비효율적이다.
***

# SARSA Algorithm
***
$$\begin{align*}
1&: \quad \text{Initialize learning rate $\alpha$} \\
2&: \quad \text{Initialize $\epsilon$} \\ 
3&: \quad \text{Randomly initialize the network parameters $\theta$} \\
4&: \quad \mathbf{for\ } m = 1, \dots, MAX\_STEPS \ \mathbf{do} \\
5&: \qquad \text{Gather $N$ experiences $(s_i, a_i, r_i, s'_i, a'_i)$ using the current $\epsilon$-greedy policy}\\
6&: \qquad \mathbf{for\ } i = 1, \dots, N\ \mathbf{do} \\
7&: \qquad \quad \text{\# Calculate target $Q$-values for each example}\\
8&: \qquad \quad y_i = r_i + \delta_{s'_i} \gamma Q^{\pi_\theta}(s'_i, a'_i)\ \text{where $\delta_{s'_i}=0$ if $s'_i$ is terminal, 1 otherwise} \\
9&: \qquad \mathbf{end\ for} \\
10&: \qquad \text{\# Calculate the loss, for example using MSE} \\
11&: \qquad L(\theta) = {1 \over N} \sum_i(y_i - Q^{\pi_\theta}(s_i, a_i))^2 \\
12&: \qquad \text{\# Update the network's parameters} \\
13&: \qquad \theta = \theta - \alpha \nabla_\theta L(\theta) \\
14&: \qquad \text{Decay $\epsilon$} \\
15&: \quad \mathbf{end\ for}
\end{align*}$$
***
