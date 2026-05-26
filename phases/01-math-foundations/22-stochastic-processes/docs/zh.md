# Stochastic Processes

> Randomness with structure. The math behind random walks, Markov chains, and diffusion models.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (probability, Bayes)
**Time:** ~75 minutes

## Learning Objectives

- Simulate 1D and 2D random walks and verify the sqrt(n) scaling of displacement
- Build a Markov chain simulator and compute its stationary distribution via eigendecomposition
- Implement Metropolis-Hastings MCMC and Langevin dynamics for sampling from target distributions
- Connect the forward diffusion process to Brownian motion and explain how the reverse process generates data

## The Problem

Many AI systems involve randomness that evolves over time. Not static randomness -- structured, sequential randomness where each step depends on what came before.

Language models generate tokens one at a time. Each token depends on the previous context. The model outputs a probability distribution, samples from it, and moves on. That is a stochastic process.

Diffusion models add noise to an image step by step until it becomes pure static. Then they reverse the process, denoising step by step until a new image emerges. The forward process is a Markov chain. The reverse process is a learned Markov chain running backward.

Reinforcement learning agents take actions in an environment. Each action leads to a new state with some probability. The agent follows a random policy in a random world. The whole thing is a Markov decision process.

MCMC sampling -- the backbone of Bayesian inference -- constructs a Markov chain whose stationary distribution is the posterior you want to sample from.

All of these build on four foundational ideas:
1. Random walks -- the simplest stochastic process
2. Markov chains -- structured randomness with a transition matrix
3. Langevin dynamics -- gradient descent with noise
4. Metropolis-Hastings -- sampling from any distribution

## The Concept

### Random Walks

Start at position 0. At each step, flip a fair coin. Heads: move right (+1). Tails: move left (-1).

After n steps, your position is the sum of n random +/-1 values. The expected position is 0 (the walk is unbiased). But the expected distance from the origin grows as sqrt(n).

This is counterintuitive. The walk is fair -- no drift in either direction. But over time, it wanders further and further from where it started. The standard deviation after n steps is sqrt(n).

```
Step 0:  Position = 0
Step 1:  Position = +1 or -1
Step 2:  Position = +2, 0, or -2
...
Step 100: Expected distance from origin ~ 10 (sqrt(100))
Step 10000: Expected distance from origin ~ 100 (sqrt(10000))
```

**In 2D**, the walk moves up, down, left, or right with equal probability. The same sqrt(n) scaling applies to the distance from the origin. The path traces a fractal-like pattern.

**Why sqrt(n)?** Each step is +1 or -1 with equal probability. After n steps, the position S_n = X_1 + X_2 + ... + X_n where each X_i is +/-1. The variance of each step is 1, and the steps are independent, so Var(S_n) = n. Standard deviation = sqrt(n). By the central limit theorem, S_n / sqrt(n) converges to a standard normal distribution.

This sqrt(n) scaling shows up everywhere in ML. SGD noise scales as 1/sqrt(batch_size). Embedding dimensions scale as sqrt(d). The square root is the signature of independent random additions.

**Connection to Brownian motion.** Take a random walk with step size 1/sqrt(n) and n steps per unit time. As n goes to infinity, the walk converges to Brownian motion B(t) -- a continuous-time process where B(t) is normally distributed with mean 0 and variance t.

Brownian motion is the mathematical foundation of diffusion. It models the random jiggling of particles in a fluid, the fluctuations of stock prices, and -- crucially -- the noise process in diffusion models.

**Gambler's ruin.** A random walker starting at position k, with absorbing barriers at 0 and N. What is the probability of reaching N before 0? For a fair walk: P(reach N) = k/N. This is surprisingly simple and elegant. It connects to the theory of martingales -- the fair random walk is a martingale (expected future value = current value).

### Markov Chains

A Markov chain is a system that transitions between states according to fixed probabilities. The key property: the next state depends only on the current state, not on the history.

```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```

This is the Markov property. It means you can describe the entire dynamics with a transition matrix P:

```
P[i][j] = probability of going from state i to state j
```

Each row of P sums to 1 (you must go somewhere).

**Example -- Weather:**

```
States: Sunny (0), Rainy (1), Cloudy (2)

P = [[0.7, 0.1, 0.2],    (if sunny: 70% sunny, 10% rainy, 20% cloudy)
     [0.3, 0.4, 0.3],    (if rainy: 30% sunny, 40% rainy, 30% cloudy)
     [0.4, 0.2, 0.4]]    (if cloudy: 40% sunny, 20% rainy, 40% cloudy)
```

Start in any state. After many transitions, the distribution of states converges to the stationary distribution pi, where pi * P = pi. This is the left eigenvector of P with eigenvalue 1.

For the weather chain, the stationary distribution might be [0.53, 0.18, 0.29] -- over the long run, it is sunny 53% of the time regardless of the starting state.

```mermaid
graph LR
    S["Sunny"] -->|0.7| S
    S -->|0.1| R["Rainy"]
    S -->|0.2| C["Cloudy"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**Computing the stationary distribution.** There are two approaches:

1. **Power method**: multiply any initial distribution by P repeatedly. After enough iterations, it converges.
2. **Eigenvalue method**: find the left eigenvector of P with eigenvalue 1. This is the eigenvector of P^T with eigenvalue 1.

Both approaches require the chain to satisfy convergence conditions.

**Convergence conditions.** A Markov chain converges to a unique stationary distribution if it is:
- **Irreducible**: every state is reachable from every other state
- **Aperiodic**: the chain does not cycle with a fixed period

Most chains you encounter in ML satisfy both conditions.

**Absorbing states.** A state is absorbing if once you enter it, you never leave (P[i][i] = 1). Absorbing Markov chains model processes with terminal states -- a game that ends, a customer who churns, a token sequence that hits the end-of-text token.

**Mixing time.** How many steps until the chain is "close" to the stationary distribution? Formally, the number of steps until the total variation distance from stationarity drops below some threshold. Fast mixing = few steps needed. The spectral gap of P (1 minus the second-largest eigenvalue) controls the mixing time. Larger gap = faster mixing.

### Connection to Language Models

Token generation in a language model is approximately a Markov process. Given the current context, the model outputs a distribution over the next token. Temperature controls the sharpness:

```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```

- Temperature = 1.0: standard distribution
- Temperature < 1.0: sharper (more deterministic)
- Temperature > 1.0: flatter (more random)
- Temperature -> 0: argmax (greedy)

Top-k sampling truncates to the k highest-probability tokens. Top-p (nucleus) sampling truncates to the smallest set of tokens whose cumulative probability exceeds p. Both modify the Markov transition probabilities.

### Brownian Motion

The continuous-time limit of the random walk. Position B(t) has three properties:
1. B(0) = 0
2. B(t) - B(s) is normally distributed with mean 0 and variance t - s (for t > s)
3. Increments on non-overlapping intervals are independent

Brownian motion is continuous but nowhere differentiable -- it jiggles at every scale. The path has fractal dimension 2 in the plane.

In discrete simulation, you approximate Brownian motion by:

```
B(t + dt) = B(t) + sqrt(dt) * z,    where z ~ N(0, 1)
```

The sqrt(dt) scaling is important. It comes from the central limit theorem applied to random walks.

### Langevin Dynamics

Gradient descent finds the minimum of a function. Langevin dynamics finds the probability distribution proportional to exp(-U(x)/T), where U is an energy function and T is temperature.

```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```

Two forces act on the particle:
1. **Gradient force** (-dt * gradient(U)): pushes toward low energy (like gradient descent)
2. **Random force** (sqrt(2*T*dt) * z): pushes in random directions (exploration)

At temperature T = 0, this is pure gradient descent. At high temperature, it is nearly a random walk. At the right temperature, the particle explores the energy landscape and spends more time in low-energy regions.

**Connection to diffusion models.** The forward process of a diffusion model is:

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
```

This is a Markov chain that gradually mixes the data with noise. After enough steps, x_T is pure Gaussian noise.

The reverse process -- going from noise back to data -- is also a Markov chain, but its transition probabilities are learned by a neural network. The network learns to predict the noise that was added at each step, then subtracts it.

```mermaid
graph LR
    subgraph "Forward Process (add noise)"
        X0["x_0 (data)"] -->|"+ noise"| X1["x_1"]
        X1 -->|"+ noise"| X2["x_2"]
        X2 -->|"..."| XT["x_T (pure noise)"]
    end
    subgraph "Reverse Process (denoise)"
        XT2["x_T (noise)"] -->|"neural net"| XR2["x_{T-1}"]
        XR2 -->|"neural net"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (generated data)"]
    end
```

### MCMC: Markov Chain Monte Carlo

Sometimes you need to sample from a distribution p(x) that you can evaluate (up to a constant) but cannot sample from directly. Bayesian posteriors are the classic example -- you know the likelihood times the prior, but the normalizing constant is intractable.

**Metropolis-Hastings** constructs a Markov chain whose stationary distribution is p(x):

1. Start at some position x
2. Propose a new position x' from a proposal distribution Q(x'|x)
3. Compute acceptance ratio: a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. Accept x' with probability min(1, a). Otherwise stay at x.
5. Repeat.

If Q is symmetric (e.g., Q(x'|x) = Q(x|x') = N(x, sigma^2)), the ratio simplifies to a = p(x') / p(x). You only need the ratio of probabilities -- the normalizing constant cancels.

The chain is guaranteed to converge to p(x) under mild conditions. But convergence can be slow if the proposal is too small (random walk) or too large (high rejection). Tuning the proposal is the art of MCMC.

**Why it works.** The acceptance ratio ensures detailed balance: the probability of being at x and moving to x' equals the probability of being at x' and moving to x. Detailed balance implies that p(x) is the stationary distribution of the chain. So after enough steps, the samples come from p(x).

**Practical considerations:**
- **Burn-in**: discard the first N samples. The chain needs time to reach the stationary distribution from its starting point.
- **Thinning**: keep every k-th sample to reduce autocorrelation.
- **Multiple chains**: run several chains from different starting points. If they converge to the same distribution, you have evidence of convergence.
- **Acceptance rate**: for Gaussian proposals in d dimensions, the optimal acceptance rate is about 23% (Roberts & Rosenthal, 2001). Too high means the chain barely moves. Too low means it rejects everything.

### Stochastic Processes in AI

| Process | AI Application |
|---------|---------------|
| Random walk | Exploration in RL, Node2Vec embeddings |
| Markov chain | Text generation, MCMC sampling |
| Brownian motion | Diffusion models (forward process) |
| Langevin dynamics | Score-based generative models, SGLD |
| Markov decision process | Reinforcement learning |
| Metropolis-Hastings | Bayesian inference, posterior sampling |

## Build It

### Step 1: Random walk simulator

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # right
    dx[directions == 1] = -1  # left
    dy[directions == 2] = 1   # up
    dy[directions == 3] = -1  # down
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

The 1D walk stores cumulative sums. Each step is +1 or -1. After n steps, the position is the sum. The variance grows linearly with n, so the standard deviation grows as sqrt(n).

### Step 2: Markov chain

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

The stationary distribution is the left eigenvector of P with eigenvalue 1. We find it by computing eigenvectors of P^T (transposing turns left eigenvectors into right eigenvectors).

### Step 3: Langevin dynamics

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

The gradient pushes x toward low energy. The noise prevents it from getting stuck. At equilibrium, the distribution of samples is proportional to exp(-U(x)/temperature).

### Step 4: Metropolis-Hastings

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

The algorithm proposes a new point, checks if it has higher probability (or accepts with probability proportional to the ratio), and repeats. The acceptance rate should be around 23-50% for good mixing.

## Use It

In practice, you use established libraries for these algorithms. But understanding the mechanics matters for debugging and tuning.

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```

### numpy for transition matrices

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")
```

Multiply the initial distribution by P repeatedly. After enough iterations, it converges to the stationary distribution regardless of where you started. This is the power method for finding the dominant left eigenvector.

### Connections to real frameworks

- **PyTorch diffusion:** The `DDPMScheduler` in Hugging Face `diffusers` implements the forward and reverse Markov chains
- **NumPyro / PyMC:** Use MCMC (NUTS sampler, which improves on Metropolis-Hastings) for Bayesian inference
- **Gymnasium (RL):** The environment step function defines a Markov decision process

### Verifying Markov chain convergence

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```

The spectral gap tells you how fast the chain forgets its initial state. A gap of 0.2 means roughly 5 steps to mix. A gap of 0.01 means roughly 100 steps. Always check this before running long simulations -- a slowly mixing chain wastes compute.

## Ship It

This lesson produces:
- `outputs/prompt-stochastic-process-advisor.md` -- a prompt that helps identify which stochastic process framework applies to a given problem

## Connections

| Concept | Where it shows up |
|---------|------------------|
| Random walk | Node2Vec graph embeddings, exploration in RL |
| Markov chain | Token generation in LLMs, MCMC sampling |
| Brownian motion | Forward diffusion process in DDPM, SDE-based models |
| Langevin dynamics | Score-based generative models, stochastic gradient Langevin dynamics (SGLD) |
| Stationary distribution | MCMC convergence target, PageRank |
| Metropolis-Hastings | Bayesian posterior sampling, simulated annealing |
| Temperature | LLM sampling, Boltzmann exploration in RL, simulated annealing |
| Mixing time | Convergence speed of MCMC, spectral gap analysis |
| Absorbing state | End-of-sequence token, terminal states in RL |
| Detailed balance | Correctness guarantee for MCMC samplers |

Diffusion models deserve special attention. DDPM (Ho et al., 2020) defines a forward Markov chain:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

where beta_t is a noise schedule. After T steps, x_T is approximately N(0, I). The reverse process is parameterized by a neural network that predicts the noise:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

生成的每一步都是学习马尔可夫链中的一步。了解马尔可夫链意味着了解扩散模型如何以及为何生成数据。

SGLD（随机梯度朗之万动力学）将小批量梯度下降与朗之万噪声相结合。您可以使用随机估计并添加校准噪声，而不是计算完整梯度。随着学习率的下降，SGLD 从优化过渡到采样——您可以免费获得近似的贝叶斯后验样本。这是从神经网络获取不确定性估计的最简单方法之一。

所有这些联系的关键见解是：随机过程不仅仅是理论工具。它们是现代人工智能系统内部的计算机制。当你调整法学硕士的温度时，你正在调整马尔可夫链。当您训练扩散模型时，您正在学习反转类似布朗运动的过程。当您运行贝叶斯推理时，您正在构建一个收敛到后验的链。

## 练习

1. **模拟 10000 步的 1000 次随机游走。** 绘制最终位置的分布。验证其近似高斯分布，平均值为 0，标准差 sqrt(10000) = 100。

2. **使用马尔可夫链构建文本生成器。** 在小型语料库上进行训练：对于每个单词，计算到下一个单词的转换。构建转移矩阵。通过从链中采样生成新句子。

3. **使用 Metropolis-Hastings 实施模拟退火**。从高温开始（几乎接受一切）并逐渐冷却（仅接受改进）。用它来查找具有许多局部最小值的函数的最小值。

4. **比较不同温度下的 Langevin 动力学。** 来自双阱电势的样本 U(x) = (x^2 - 1)^2。在低温下，样品聚集在一个孔中。在高温下，它们会扩散到两者上。找到链在孔之间混合的临界温度。

5. **实施前向扩散过程。** 从一维信号（例如正弦波）开始。使用线性噪声计划逐步添加噪声超过 100 个步骤。显示信号如何退化为纯噪声。然后实现一个简单的降噪器来反转该过程（即使是一个简单的降噪器，只是减去估计的噪声）。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|随机游走 | “抛硬币运动”|每一步位置以随机增量变化的过程 |
|马尔可夫性质 | 《无记忆》|未来只取决于现状，而不取决于历史|
|转移矩阵| “概率表”| P[i][j] = 从状态 i 移动到状态 j 的概率 |
|固定式配送| “长期平均值”|分布 pi，其中 pi*P = pi -- 链的平衡 |
|布朗运动| “随机晃动”|随机游走的连续时间限制，B(t) ~ N(0, t) |
|朗之万动态| “带有噪声的梯度下降” |结合确定性梯度和随机扰动的更新规则 |
| MCMC | 《向着目标前进》|构建一条马尔可夫链，其平稳分布就是您想要的 |
|黑斯廷斯大都会 | “提议并接受/拒绝” |利用接受率保证收敛的MCMC算法 |
|温度| “随机性旋钮” |控制勘探和开发之间权衡的参数|
|扩散过程| “噪音进来，噪音出去” |前进：逐渐添加噪音。反向：逐渐去除。生成数据。 |

## 进一步阅读

- **Ho, Jain, Abbeel (2020)** --“去噪扩散概率模型”。 DDPM 论文掀起了扩散模型革命。前向和反向马尔可夫链的清晰推导。
- **Song & Ermon (2019)** --“通过估计数据分布梯度进行生成建模。”使用 Langevin 动力学进行采样的基于分数的方法。
- **Roberts & Rosenthal (2004)** --“一般状态空间马尔可夫链和 MCMC 算法。” MCMC 何时以及为何发挥作用背后的理论。
- **诺里斯 (1997)** -- “马尔可夫链”。标准教科书。涵盖收敛、平稳分布和命中时间。
- **Welling & Teh (2011)** --“通过随机梯度朗之万动力学进行贝叶斯学习。”将 SGD 与 Langevin 动力学相结合，实现可扩展的贝叶斯推理。
