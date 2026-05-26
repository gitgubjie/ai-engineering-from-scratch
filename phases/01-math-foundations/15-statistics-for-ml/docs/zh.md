# Statistics for Machine Learning

> Statistics is how you know if your model actually works or just got lucky.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Compute descriptive statistics, Pearson/Spearman correlation, and covariance matrices from scratch
- Perform hypothesis tests (t-test, chi-squared) and interpret p-values and confidence intervals correctly
- Use bootstrap resampling to construct confidence intervals for any metric without distributional assumptions
- Distinguish statistical significance from practical significance using effect size measures

## The Problem

You trained two models. Model A scores 0.87 on your test set. Model B scores 0.89. You deploy Model B. Three weeks later, production metrics are worse than before. What happened?

Model B did not actually outperform Model A. The 0.02 difference was noise. Your test set was too small, or the variance too high, or both. You shipped randomness dressed up as improvement.

This happens constantly. Kaggle leaderboard shakeups. Papers that fail to reproduce. A/B tests that declare winners based on a few hundred samples. The root cause is always the same: someone skipped the statistics.

Statistics gives you the tools to distinguish signal from noise. It tells you when a difference is real, how confident you should be, and how much data you need before you can trust a result. Every ML pipeline, every model comparison, every experiment needs statistics. Without it, you are guessing.

## The Concept

### Descriptive Statistics: Summarizing Your Data

Before you model anything, you need to know what your data looks like. Descriptive statistics compress a dataset into a few numbers that capture its shape.

**Measures of central tendency** answer "where is the middle?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

The mean is the balance point. The median is the halfway mark. When they diverge, your distribution is skewed. Income distributions have mean >> median (right skew from billionaires). Loss distributions during training often have mean << median (left skew from easy samples).

**Measures of spread** answer "how dispersed is the data?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentiles** divide sorted data into 100 equal parts. The 25th percentile (Q1) means 25% of values fall below this point. The 50th percentile is the median. The 75th percentile is Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

In ML, you care about percentiles for inference latency, prediction confidence distributions, and understanding error distributions. A model with low average error but terrible P99 error might be useless for safety-critical applications.

**Sample vs population statistics.** When computing variance from a sample, divide by (n-1) instead of n. This is Bessel's correction. It compensates for the fact that your sample mean is not the true population mean. With n in the denominator, you systematically underestimate the true variance. With (n-1), the estimate is unbiased.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

In practice: if n is large (thousands of samples), the difference is negligible. If n is small (dozens of samples), it matters.

### Correlation: How Variables Move Together

Correlation measures the strength and direction of a linear relationship between two variables.

**Pearson correlation coefficient** measures linear association:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson assumes the relationship is linear and both variables are roughly normally distributed. It is sensitive to outliers. A single extreme point can drag r from 0.1 to 0.9.

**Spearman rank correlation** measures monotonic association:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**When to use each:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**The golden rule:** correlation does not imply causation. Ice cream sales and drowning deaths are correlated because both increase in summer. Your model's accuracy and the number of parameters are correlated, but adding parameters does not automatically improve accuracy (see: overfitting).

### Covariance Matrix

The covariance between two variables measures how they vary together:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

For d features, the covariance matrix C is a d x d matrix where C[i][j] = Cov(feature_i, feature_j). The diagonal entries C[i][i] are the variances of each feature.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Connection to PCA.** PCA eigendecomposes the covariance matrix. The eigenvectors are the principal components (directions of maximum variance). The eigenvalues tell you how much variance each component captures. This is exactly what Lesson 10 covered, but now you see why the covariance matrix is the right thing to decompose: it encodes all pairwise linear relationships in your data.

**Connection to correlation.** The correlation matrix is the covariance matrix of standardized variables (each divided by its standard deviation). Correlation normalizes covariance so all values fall in [-1, 1].

### Hypothesis Testing

Hypothesis testing is a framework for making decisions under uncertainty. You start with a claim, collect data, and determine if the data is consistent with the claim.

**The setup:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**The p-value** is the probability of seeing data as extreme as what you observed, assuming H0 is true. It is NOT the probability that H0 is true. This is the single most common misunderstanding in statistics.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence intervals** give a range of plausible values for a parameter:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

The width of the confidence interval tells you about precision. Wide intervals mean high uncertainty. Narrow intervals mean your estimate is precise (but not necessarily accurate, if your data is biased).

### The t-test

The t-test compares means. There are several flavors.

**One-sample t-test:** is the population mean different from a hypothesized value?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (independent):** are two group means different?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test:** when measurements come in pairs (same model evaluated on same data splits):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

In ML, the paired t-test is common: you run both models on the same 10 cross-validation folds and compare their scores pairwise.

### Chi-squared Test

The chi-squared test checks if observed frequencies match expected frequencies. Useful for categorical data.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing for ML Models

A/B testing in ML is not the same as web A/B testing. Model comparison has specific challenges:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**The procedure:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

A result can be statistically significant but practically meaningless. With enough data, even a trivial difference becomes statistically significant.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** quantifies how big the difference is, independent of sample size:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Always report both the p-value and the effect size. The p-value tells you if the difference is real. The effect size tells you if it matters.

### Multiple Comparison Problem

When you test many hypotheses, some will be "significant" by chance. If you test 20 things at alpha = 0.05, you expect 1 false positive even when nothing is real.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction:** divide alpha by the number of tests.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

In ML, this matters when you compare a model across multiple metrics, test many hyperparameter configurations, or evaluate on multiple datasets.

### Bootstrap Methods

Bootstrapping estimates the sampling distribution of a statistic by resampling your data with replacement. No assumptions about the underlying distribution required.

**The algorithm:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval (percentile method):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Why bootstrap matters for ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap for model comparison:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

This is more robust than the paired t-test because it makes no distributional assumptions.

### Parametric vs Non-parametric Tests

**Parametric tests** assume a specific distribution (usually normal):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric tests** make no distributional assumptions:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**When to use non-parametric:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**When to use parametric:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

In ML experiments, you typically have small n (5 or 10 cross-validation folds), so non-parametric tests like Wilcoxon signed-rank are often more appropriate than t-tests.

### Central Limit Theorem: Practical Implications

The CLT says the distribution of sample means approaches a normal distribution as n grows, regardless of the underlying population distribution.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Why this matters for ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**What CLT does NOT do:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 论文中常见的统计错误

1. **在训练集上进行测试。** 保证过拟合。始终保留模型在训练期间从未见过的数据。

2. **没有置信区间。** 在没有不确定性的情况下报告单个准确度数字会使结果不可重复且无法验证。

3. **忽略多重比较。** 测试 50 种配置并报告最佳配置而不进行修正会增加误报率。

4. **令人困惑的统计意义和实际意义。** 对于 0.01% 的准确度改进，p 值为 0.001 没有意义。

5. **使用不平衡数据的准确度。** 在具有 99% 负类的数据集上，99% 的准确度意味着模型什么也没学到。使用精度、召回率、F1 或 AUC。

6. **挑选指标。** 仅报告模型获胜的指标。诚实的评估报告所有相关指标。

7. **在训练/测试拆分中泄漏信息。** 在拆分之前进行标准化，或使用未来的数据来预测过去。

8. **没有方差估计的小测试集。** 对 100 个样本进行评估并声称 2% 的改进是噪声，而不是信号。

9. **当数据不独立时假设独立性。** 来自同一患者的医学图像，来自同一文档的多个句子。组内的观察结果是相关的。

10. **P-hacking。** 尝试不同的测试、子集或排除标准，直到得到 p < 0.05。结果是搜索的产物。

## 构建它

您将实施：

1. **从头开始的描述性统计**（平均值、中位数、众数、标准差、百分位数、IQR）
2. **相关函数**（Pearson 和 Spearman，带有协方差矩阵）
3. **假设检验**（单样本t检验、双样本t检验、卡方检验）
4. **自举置信区间**（对于任何统计数据，无需假设）
5. **A/B测试模拟器**（生成数据、测试、检查I类和II类错误）
6. **统计与实际意义演示**（表明大 n 使一切都“有意义”）

一切从头开始，仅使用“数学”和“随机”。没有 numpy，没有 scipy。

## 关键术语

|术语 |定义|
|---|---|
|平均 |值的总和除以计数。对异常值敏感。 |
|中位数 |排序数据的中间值。对异常值具有鲁棒性。 |
|标准差|方差的平方根。措施以原始单位传播。 |
|百分位数 |低于给定百分比的数据的值。 |
| IQR |四分位数范围。 Q3 减去 Q1。中间的价差50%。 |
|皮尔逊相关 |测量两个变量之间的线性关联。范围 [-1, 1]。 |
|斯皮尔曼相关 |使用排名来衡量单调关联。 |
|协方差矩阵 |所有特征之间的成对协方差矩阵。 |
|零假设 |默认假设没有影响或没有差异。 |
| p 值 |在原假设成立的情况下，这种极端数据的概率。 |
|置信区间|给定置信水平下参数的合理值范围。 |
| t 检验 |测试均值是否存在显着差异。使用 t 分布。 |
|卡方检验|测试观察到的频率是否与预期频率不同。 |
|效应大小|差异的大小，与样本大小无关。科恩的 d 很常见。 |
|邦费罗尼校正 |将显着性阈值除以测试数量以控制误报。 |
|引导程序|通过替换重采样来估计采样分布。 |
| I 类错误 |误报。当 H0 为真时拒绝 H0。 |
| II 类错误 |假阴性。当 H0 为假时未能拒绝。 |
|统计功效|正确拒绝假 H0 的概率。功效 = 1 减去 II 类错误率。 |
|中心极限定理|随着样本量的增加，样本均值收敛于正态分布。 |
|参数测试|假设数据服从特定分布（通常是正态分布）。 |
|非参数检验|不做任何分布假设。适用于等级或标志。 |
