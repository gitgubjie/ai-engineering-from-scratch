# Probability and Distributions

> Probability is the language AI uses to express uncertainty.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Learning Objectives

- Implement PMFs and PDFs from scratch for Bernoulli, categorical, Poisson, uniform, and normal distributions
- Compute expected value, variance, and use the Central Limit Theorem to explain why Gaussians dominate
- Build softmax and log-softmax functions with the numerical stability trick (subtract max logit)
- Calculate cross-entropy loss from logits and connect it to negative log-likelihood

## The Problem

A classifier outputs `[0.03, 0.91, 0.06]`. A language model picks the next word from 50,000 candidates. A diffusion model generates images by sampling from learned distributions. All of these are probability in action.

Every prediction a model makes is a probability distribution. Every loss function measures how far the predicted distribution is from the true one. Every training step adjusts parameters to make one distribution look more like another. Without probability, you cannot read a single ML paper, debug a single model, or understand why your training loss is NaN.

## The Concept

### Events, Sample Spaces, and Probability

The sample space S is the set of all possible outcomes. An event is a subset of the sample space. Probability maps events to numbers between 0 and 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Three axioms define all of probability:
1. P(A) >= 0 for any event A
2. P(S) = 1 (something always happens)
3. P(A or B) = P(A) + P(B) when A and B cannot both occur

Everything else (Bayes' theorem, expectations, distributions) follows from these three rules.

### Conditional Probability and Independence

P(A|B) is the probability of A given that B happened.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Two events are independent when knowing one tells you nothing about the other:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Coin flips are independent. Drawing cards without replacement is not.

### Probability Mass Functions vs Probability Density Functions

Discrete random variables have a probability mass function (PMF). Each outcome has a specific probability that you can read off directly.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Continuous random variables have a probability density function (PDF). The density at a single point is not a probability. Probability comes from integrating the density over an interval.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

This distinction matters in ML. Classification outputs are PMFs (discrete choices). VAE latent spaces use PDFs (continuous).

### Common Distributions

**Bernoulli:** one trial, two outcomes. Models binary classification.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** one trial, k outcomes. Models multi-class classification (softmax output).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** all outcomes equally likely. Used for random initialization.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** the bell curve. Parameterized by mean (mu) and variance (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** counts of rare events in a fixed interval. Models event rates.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Expected Value and Variance

Expected value is the weighted average outcome.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Variance measures spread around the mean.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

In ML, expected value appears as the loss function (average loss over the data distribution). Variance tells you about model stability. High variance in gradients means noisy training.

### Joint and Marginal Distributions

A joint distribution P(X, Y) describes two random variables together.

Joint PMF example (X = weather, Y = umbrella):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

The marginal distribution sums out the other variable:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

The row and column totals in the table above are the marginals.

### Why the Normal Distribution Shows Up Everywhere

The Central Limit Theorem: the sum (or average) of many independent random variables converges to a normal distribution, regardless of the original distribution.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

This is why:
- Measurement errors are approximately normal (many small independent sources)
- Weight initializations in neural networks use normal distributions
- Gradient noise in SGD is approximately normal (sum of many sample gradients)
- The normal distribution is the maximum entropy distribution for a given mean and variance

### Log Probabilities

Raw probabilities cause numerical problems. Multiplying many small probabilities together quickly underflows to zero.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilities fix this. Multiplications become additions.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Rules:
- log(a * b) = log(a) + log(b)
- log probabilities are always <= 0 (since 0 < P <= 1)
- More negative = less likely
- Cross-entropy loss is the negative log probability of the correct class

### Softmax as a Probability Distribution

Neural networks output raw scores (logits). Softmax converts them into a valid probability distribution.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

The softmax trick: subtract the max logit before exponentiating to prevent overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax combines softmax and log for numerical stability. PyTorch uses this internally for cross-entropy loss.

### Sampling

Sampling means drawing random values from a distribution. In ML:
- Dropout randomly samples which neurons to zero out
- Data augmentation samples random transformations
- Language models sample the next token from the predicted distribution
- Diffusion models sample noise and progressively denoise

Sampling from arbitrary distributions requires techniques like inverse transform sampling, rejection sampling, or the reparameterization trick (used in VAEs).

## Build It

### Step 1: Probability basics

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: PMF and PDF from scratch

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: Expected value and variance

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: Sampling from distributions

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax and log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: Central Limit Theorem demonstration

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: Visualization

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Full implementations with all visualizations are in `code/probability.py`.

## Use It

With NumPy and SciPy, everything above is one-liners:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

您从头开始构建了这些。现在你知道图书馆的电话正在做什么了。

## 练习

1. 对指数分布实施逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 进行比较来进行验证。

2. 为两个加载的骰子建立联合分配表。计算边际分布并检查骰子是否独立。

3. 计算 5 类分类器的交叉熵损失，当正确的类别为索引 3 时，该分类器输出 logits“[2.0, 0.5, -1.0, 3.0, 0.1]”。然后使用 PyTorch 的“nn.CrossEntropyLoss”验证您的答案。

4. 编写一个函数，获取对数概率列表并返回最可能的序列、总对数概率和等效的原始概率。使用包含 50 个单词的句子进行测试，其中每个单词的概率为 0.01。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|样本空间| “所有的可能性”|实验的每个可能结果的集合 S |
| PMF | “概率函数”|给出每个离散结果的精确概率的函数，总和为 1 |
| PDF | “概率曲线”|连续变量的密度函数。在一个区间内对其进行积分以获得概率 |
|条件概率| “给定某物的概率” | P(A\|B) = P(A 和 B) / P(B)。贝叶斯思维基础和贝叶斯定理|
|独立| “它们不会互相影响”| P(A 和 B) = P(A) * P(B)。了解一个事件并不能告诉你关于另一个事件的信息 |
|预期值| “平均”|所有结果的概率加权总和。损失函数是期望值 |
|方差| “如何分散”|与平均值的预期平方偏差。高方差 = 嘈杂、不稳定的估计 |
|正态分布| “钟形曲线”| f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于 CLT 无处不在 |
|中心极限定理| “平均值变得正常”|无论来源如何，许多独立样本的均值都会收敛于正态分布 |
|联合配送| “两个变量在一起” | P(X, Y) 描述 X 和 Y 结果的每种组合的概率 |
|边际分布 | “对其他变量求和” | P(X) = sum_y P(X, Y)。从联合 | 恢复一个变量的分布
|对数概率 | “概率的对数” |对数 P(x)。将乘积转化为和，防止长序列中的数值下溢 |
| Softmax| “将分数转化为概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实值 logits 映射到有效的概率分布 |
|交叉熵 | “损失函数”| -sum(p_true * log(p_predicted))。衡量两个分布的差异程度。越低越好 |
|洛吉特| “原始模型输出” | softmax 之前的非标准化分数。以物流函数命名 |
|取样| “绘制随机值” |根据概率分布生成值。模型如何生成输出 |

## 进一步阅读

- [3Blue1Brown：但是什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么平均值变得正常的直观证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 简明参考，涵盖此处的所有内容以及更多内容
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性很重要以及如何实现它
