# Policy Gradient
- Objective

$$\begin{align*}
J(\pi_\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \\
&= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum^T_{t=0} \gamma^t r_t\right]
\end{align*}$$

- Policy Gradient
$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum^T_{t=0} R_t(\tau)\nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

>	*Proof*.  함수 $f(x)$와 parameter로 표현되는 probability distribution $p(x|\theta)$, 함수의 expectation value $\mathbb{E}_{x \sim p(x|\theta)} [f(x)]$가 주어졌을 때, expectation value의 gradient는 다음과 같다.
>	$$
	\begin{align}
    \nabla_\theta \mathbb{E}_{x\sim p(x|\theta)} [f(x)]
	&= \nabla_\theta \int dx f(x)p(x|\theta) \\
	&= \int dx \nabla_\theta(f(x) p(x|\theta)) \\
	&= \int dx(f(x)\nabla_\theta p(x|\theta) + p(x|\theta) \nabla_\theta f(x)) \\
	&= \int dx f(x) \nabla_\theta p(x|\theta) \\
	&= \int dx f(x) p(x|\theta) {\nabla_\theta p(x|\theta) \over p(x|\theta)} \\
	&= \int dx f(x) p(x|\theta) \nabla_\theta \log p(x|\theta) \\
	&= \mathbb{E}_x [f(x) \nabla_\theta \log p(x|\theta)]
	\end{align}
$$
>	여기서 $x = \tau, \ f(x) = R(\tau), \ p(x|\theta) = p(\tau | \theta)$를 대입하면 다음의 식으로 표현된다.
>	$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)\nabla_\theta \log p(\tau|\theta)]
$$
>	또한 trajectory $\tau$에 대해 $a_t$는 $\pi_\theta(a_t|s_t)$에서, $s_{t+1}$은 $p(s_{t+1}|s_t,a_t)$에서 추출되고, 모든 확률이 서로 독립이므로 전체 trajectory의 확률은 다음과 같이 표현된다.
>	$$
	p(\tau | \theta) = \prod_{t \geq 0} p(s_{t+1} | s_t, a_t) \pi_\theta(a_t|s_t)
	$$
>	이제 양변에 로그를 취하면 된다.
>	$$
	\begin{align}
	\log p(\tau | \theta) &= \log \prod_{t \geq 0} p(s_{t+1} | s_t, a_t) \pi_\theta(a_t|s_t) \\
	\log p(\tau | \theta) &= \sum_{t \geq 0}\left(\log p(s_{t+1} | s_t, a_t) + \log \pi_\theta(a_t|s_t) \right) \\
	\nabla_\theta \log p(\tau | \theta) &= \nabla_\theta \sum_{t \geq 0}\left(\log p(s_{t+1}|s_t, a_t) + \log \pi_\theta(a_t|s_t) \right) \\
	\nabla_\theta \log p(\tau|\theta) &= \nabla_\theta \sum_{t \geq 0} \log \pi_\theta(a_t|s_t)
	\end{align}
	$$
>	이 결과를 $\nabla_\theta J(\pi_\theta)$에 대입하하고, 임의의 시각 $t$에서의 보상만을 고려하는 형태로 분산을 줄이면 다음의 식을 얻을 수 있다.
>	$$
	\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum^T_{t=0} R_t(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
	$$
>	이는 $R(\tau) = R_0(\tau) = \sum^T_{t'=0} \gamma^{t'}r_{t'} \rightarrow \sum^T_{t'=t} \gamma^{t'-t} r_{t'} = R_t(\tau)$에 의해 성립한다.

# REINFORCE Algorithm
***
$$
\begin{align*}
1&: \quad \text{Initialize learning rate } \alpha \\
2&: \quad \text{Initialize weight $\theta$ of a policy network $\pi_\theta$} \\ 
3&: \quad \mathbf{for\ } episode = 0, \dots, MAX\_EPISODE \ \mathbf{do} \\
4&: \qquad \text{Sample a trajectory }\tau = s_0, a_0, r_0, \dots, s_T, a_T, r_T \\
5&: \qquad \text{Set} \nabla_\theta J(\pi_\theta) = 0 \\
6&: \qquad \mathbf{for\ } t=0, \dots, T\ \mathbf{do} \\
7&: \qquad \quad R_t(\tau) = \Sigma^T_{t'=t} \gamma^{t'-t}r'_t \\
8&: \qquad \quad \nabla_\theta J(\pi_\theta) = \nabla_\theta J(\pi_\theta) + R_t(\tau) \nabla_\theta \log \pi_\theta (a_t|s_t) \\
9&: \qquad \mathbf{end\ for} \\
10&: \qquad \theta = \theta + \alpha \nabla_\theta J(\pi_\theta) \\
11&: \quad \mathbf{end\ for}
\end{align*}
$$
***
## Improved REINFORCE
- Reward의 baseline을 조절하여 분산 감소
	$$
	\nabla_\theta J(\pi_\theta)\approx \sum^T_{t=0} (R_t (\tau) - b(s_t)) \nabla_\theta \log \pi_\theta(a_t|s_t)
	$$
	- Baseline 예시
		1. Value function $V^\pi$
		2. Average return $b = {1 \over T} \sum ^T_{t=0} R_t(\tau)$
