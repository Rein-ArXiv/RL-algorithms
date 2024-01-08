[On-Line Q-Learning Using Connectionist Systems(Rummery, 1994)](file:///home/rein/Documents/Papers/On-Line%20Q-Learning%20Using%20Connectionist%20Systems.pdf)
# Core ideas
## 1. 시간차 학습 (Temporal difference, TD)
- SARSA에서 target value $Q_\text{target}$을 계산하는 방법
- State $s$에서 시작하는 $N$개의 trajectory $\tau_i \ (i \in \{1, \dots, N \})$가 주어졌을 때 $Q^\pi_\text{target}(s, a)$에 대한 Monte-Carlo estimation
$$ Q^\pi_{\text{target:MC}} (s, a) = {1 \over N} \sum^N_{i=1} R(\tau_i) $$
	- 하나의 episode가 끝나야 학습하므로, training에서 비효율적으로 진행된다.
- 
$$ Q^\pi(s,a) = \mathbb{E}_{s'|s,a), r\sim \mathcal{R}(s,a,s')}\left[r + \gamma \mathbb{E}_{a' \sim \pi(s')}[Q^\pi(s', a')]\right] $$
## 2. Q 함수 (Q-function)