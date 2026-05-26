# Naive Bayes

> The "naive" assumption is wrong, and it works anyway. That's the beauty of it.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Learning Objectives

- Implement Multinomial Naive Bayes from scratch with Laplace smoothing for text classification
- Explain why the naive independence assumption is mathematically wrong but produces correct class rankings in practice
- Compare Multinomial, Bernoulli, and Gaussian Naive Bayes variants and select the right one for a given feature type
- Evaluate Naive Bayes against logistic regression on high-dimensional sparse data and explain the bias-variance tradeoff at work

## The Problem

You need to classify text. Emails into spam or not-spam. Customer reviews into positive or negative. Support tickets into categories. You have thousands of features (one per word) and limited training data.

Most classifiers choke here. Logistic regression needs enough samples to estimate thousands of weights reliably. Decision trees split on one word at a time and overfit wildly. KNN in 10,000 dimensions is meaningless because every point is equally far from every other point.

Naive Bayes handles this. It makes a mathematically wrong assumption (that every feature is independent of every other feature given the class), and it still outperforms "smarter" models on text classification, especially with small training sets. It trains in a single pass through the data. It scales to millions of features. It produces probability estimates (though often poorly calibrated due to the independence assumption).

Understanding why a wrong assumption leads to good predictions teaches you something fundamental about machine learning: the best model is not the most correct one, it is the one with the best bias-variance tradeoff for your data.

## The Concept

### Bayes' Theorem (Quick Review)

Bayes' theorem flips conditional probabilities:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

We want `P(class | features)` -- the probability that a document belongs to a class given the words in it. We can compute this from:
- `P(features | class)` -- the likelihood of seeing these words in documents of this class
- `P(class)` -- the prior probability of the class (how common is spam in general?)
- `P(features)` -- the evidence, same for all classes, so we can ignore it when comparing

The class with the highest `P(class | features)` wins.

### The Naive Independence Assumption

Computing `P(features | class)` exactly requires estimating the joint probability of all features together. With a vocabulary of 10,000 words, you would need to estimate a distribution over 2^10,000 possible combinations. Impossible.

The naive assumption: every feature is conditionally independent given the class.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Instead of one impossible joint distribution, you estimate n simple per-feature distributions. Each one needs only a count.

This assumption is obviously wrong. The words "machine" and "learning" are not independent in any document. But the classifier does not need correct probability estimates. It needs correct rankings -- which class has the highest probability. The independence assumption introduces systematic errors, but those errors affect all classes similarly, so the ranking stays correct.

### Why It Still Works

Three reasons:

1. **Ranking over calibration.** Classification only needs the top-ranked class to be correct. Even if P(spam) = 0.99999 when the true probability is 0.7, the classifier still picks spam correctly. We do not need correct probabilities. We need the correct winner.

2. **High bias, low variance.** The independence assumption is a strong prior. It constrains the model heavily, which prevents overfitting. With limited training data, a model that is slightly wrong but stable beats a model that is theoretically right but wildly unstable. This is the bias-variance tradeoff in action.

3. **Feature redundancy cancels out.** Correlated features provide redundant evidence. The classifier double-counts this evidence, but it double-counts it for the correct class too. If "machine" and "learning" always appear together, both provide evidence for the "tech" class. NB counts them twice, but it counts them twice for the right class.

A fourth, practical reason: Naive Bayes is extremely fast. Training is a single pass through the data counting frequencies. Prediction is a matrix multiplication. You can train on a million documents in seconds. This speed means you can iterate faster, try more feature sets, and run more experiments than with slower models.

### The Math Step by Step

Let us trace through a concrete example. Suppose we have two classes: spam and not-spam. Our vocabulary has three words: "free", "money", "meeting".

Training data:
- Spam emails mention "free" 80 times, "money" 60 times, "meeting" 10 times (150 total words)
- Not-spam emails mention "free" 5 times, "money" 10 times, "meeting" 100 times (115 total words)
- 40% of emails are spam, 60% are not-spam

With Laplace smoothing (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

New email contains: "free" (2 times), "money" (1 time), "meeting" (0 times).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam wins by a large margin. The word "free" appearing twice is strong evidence for spam. Note that "meeting" not appearing contributes zero to both log sums (0 * log(P)) -- in Multinomial NB, absent words have no effect. It is Bernoulli NB that explicitly models word absence.

### Three Variants

Naive Bayes comes in three flavors. Each models `P(feature | class)` differently.

#### Multinomial Naive Bayes

Models each feature as a count. Best for text data where features are word frequencies or TF-IDF values.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

The `alpha` is Laplace smoothing (explained below). This variant is the workhorse for text classification.

#### Gaussian Naive Bayes

Models each feature as a normal distribution. Best for continuous features.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Each class gets its own mean and variance per feature. This works well when features genuinely follow a bell curve within each class.

#### Bernoulli Naive Bayes

Models each feature as binary (present or absent). Best for short text or binary feature vectors.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Unlike Multinomial, Bernoulli explicitly penalizes the absence of a word. If "free" typically appears in spam but is absent from this email, Bernoulli counts that as evidence against spam.

### When to Use Each Variant

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Laplace Smoothing

What happens when a word appears in the test data but never appeared in the training data for a particular class?

Without smoothing: `P(word | class) = 0/N = 0`. One zero multiplied through the entire product makes `P(class | features) = 0`, regardless of all other evidence. A single unseen word destroys the entire prediction, no matter how much other evidence supports it.

Laplace smoothing adds a small count `alpha` (usually 1) to every feature count:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

With alpha=1, every word gets at least a tiny probability. The word "discombobulate" appearing in a test email no longer kills the spam probability. The smoothing has a Bayesian interpretation: it is equivalent to placing a uniform Dirichlet prior on the word distributions.

Higher alpha means stronger smoothing (more uniform distributions). Lower alpha means the model trusts the data more. Alpha is a hyperparameter you tune.

The effect of alpha:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### Log-Space Computation

Multiplying hundreds of probabilities (each less than 1) causes floating-point underflow. The product becomes zero in floating point even though the true value is a very small positive number.

The solution: work in log space. Instead of multiplying probabilities, add their logarithms:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

This turns the prediction into a dot product:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Matrix multiplication. That is why Naive Bayes prediction is so fast -- it is the same operation as a single-layer linear model.

### Naive Bayes vs Logistic Regression

Both are linear classifiers for text. The difference is in what they model.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models P(X\|Y)) | Discriminative (models P(Y\|X)) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Rule of thumb: start with Naive Bayes. If you have enough data and NB plateaus, switch to logistic regression.

### Classification Pipeline

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

In practice, we work in log space to avoid floating-point underflow. Instead of multiplying many small probabilities, we add their logarithms:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## Build It

The code in `code/naive_bayes.py` implements both MultinomialNB and GaussianNB from scratch.

### MultinomialNB

The from-scratch implementation:

1. **fit(X, y)**: For each class, count the frequency of each feature. Add Laplace smoothing. Compute log probabilities. Store class priors (log of class frequencies).

2. **predict_log_proba(X)**: For each sample, compute log P(class) + sum of log P(feature_i | class) for all classes. This is a matrix multiplication: X @ log_probs.T + log_priors.

3. **predict(X)**: Return the class with highest log probability.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

The key insight: after fitting, prediction is just matrix multiplication plus a bias. This is why Naive Bayes is so fast.

### GaussianNB

For continuous features, we estimate mean and variance per class per feature:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Prediction uses the Gaussian PDF per feature, multiplied across features (added in log space).

### Demo: Text Classification

The code generates synthetic bag-of-words data simulating two classes (tech articles vs sports articles). Each class has a different word frequency distribution. MultinomialNB classifies them using word counts.

The synthetic data works like this: we create 200 "words" (feature columns). Words 0-39 have high frequency in tech articles and low in sports. Words 80-119 have high frequency in sports and low in tech. Words 40-79 are medium frequency in both. This creates a realistic scenario where some words are strong class indicators and others are noise.

### Demo: Continuous Features

The code generates Iris-like data (3 classes, 4 features, Gaussian clusters). GaussianNB classifies using per-class mean and variance. Each class has a different center (mean vector) and different spread (variance), mimicking real-world data where measurements differ systematically between categories.

The code also demonstrates:
- **Smoothing comparison:** Training MultinomialNB with different alpha values to show the effect of smoothing strength on accuracy.
- **Training size experiment:** How NB accuracy improves as training data grows from 20 to 1600 samples. NB reaches decent accuracy even with very few samples -- this is its main advantage.
- **Confusion matrix:** Per-class precision, recall, and F1 score to show where NB makes mistakes.

### Prediction Speed

Naive Bayes prediction is a matrix multiplication. For n samples with d features and k classes:
- MultinomialNB: one matrix multiply (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k Gaussian PDF evaluations, each over d features = O(n * d * k)

Both are linear in every dimension. Compare this to KNN (which requires distance computation to all training points) or SVM with RBF kernel (which requires kernel evaluation against all support vectors). NB is faster by orders of magnitude at prediction time.

## Use It

With sklearn, both variants are one-liners:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

For text classification with sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

The code in `naive_bayes.py` compares from-scratch implementations against sklearn on the same data to verify correctness.

### TF-IDF with Naive Bayes

Raw word counts give every word equal weight per occurrence. But common words like "the" and "is" appear frequently in every class -- they carry no information. TF-IDF (Term Frequency - Inverse Document Frequency) downweights common words and upweights rare, discriminative words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF values are non-negative, so they work with MultinomialNB. The combination of TF-IDF + MultinomialNB is one of the strongest baselines for text classification. It frequently beats more complex models on datasets with fewer than 10,000 training samples.

### BernoulliNB for Short Text

For short text (tweets, SMS, chat messages), BernoulliNB can outperform MultinomialNB. Short texts have low word counts, so the frequency information that MultinomialNB relies on is noisy. BernoulliNB only cares about presence or absence, which is more reliable with short text.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

The `binary=True` flag in CountVectorizer converts all counts to 0/1. Without it, BernoulliNB still works but is seeing counts that it was not designed for.

### Calibrating NB Probabilities

NB probabilities are poorly calibrated. When NB says P(spam) = 0.95, the true probability might be 0.7. If you need reliable probability estimates (for example, to set a threshold or to combine with other models), use sklearn's CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这使用交叉验证在 NB 的原始分数之上拟合逻辑回归。得到的概率更接近真实的类别频率。

### 常见问题

1. **Negative feature values.** MultinomialNB requires non-negative features.如果您有负值（例如具有某些设置或标准化特征的 TF-IDF），请改用 GaussianNB，或将特征转变为正值。

2. **零方差特征。** GaussianNB 除以方差。如果某个特征对于某个类的方差为零（所有值相同），则概率计算将中断。该代码向所有方差添加了一个小的平滑项 (1e-9) 以防止出现这种情况。

3. **类别不平衡。** 如果 99% 的电子邮件不是垃圾邮件，则先前的 P（非垃圾邮件）= 0.99 非常强大，以至于压倒了可能性证据。您可以手动设置类先验或使用 sklearn 中的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它适用于计数）。 GaussianNB 也不需要缩放（它估计每个特征的统计数据）。与对特征尺度敏感的逻辑回归和 SVM 相比，这是一个优势。

## 发货

本课产生：
- `outputs/skill-naive-bayes-chooser.md` -- 用于选择正确 NB 变体的决策技能
- `code/naive_bayes.py` -- 从头开始​​的 MultinomialNB 和 GaussianNB，与 sklearn 比较

### 当朴素贝叶斯失败时

当独立性假设导致不正确的排名（不仅仅是不正确的概率）时，NB 失败。出现这种情况时：

1. **强大的特征交互。** 如果该类依赖于两个特征的组合而不是单独的一个（类似 XOR 的模式），NB 将完全错过它。每个特征本身并不能提供任何证据，并且 NB 无法将它们非线性地组合起来。

2. **具有相反证据的高度相关特征。** 如果特征 A 说“垃圾邮件”，特征 B 说“非垃圾邮件”，但 A 和 B 完全相关（它们在现实中总是一致），NB 将看到没有证据的地方存在冲突。

3. **非常大的训练集。** 有了足够的数据，像逻辑回归这样的判别模型就能学习到真正的决策边界，并且表现优于 NB。有助于处理小数据的独立性假设现在阻碍了模型的发展。

实际上，这些故障模式对于文本分类来说很少见。文本特征数量众多，个别特征较弱，并且独立性假设的错误往往会被抵消。对于几乎没有强相关特征的表格数据，首先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 在 alpha 值为 0.01、0.1、1.0、10.0 和 100.0 的文本数据上训练 MultinomialNB。绘制准确度与 alpha 的关系图。性能峰值在哪里？为什么过高的阿尔法会造成伤害？

2. **特征独立性测试。** 采用真实的文本数据集。选择两个明显相关的词（“机器”和“学习”）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 进行比较。独立性假设有多错误？会影响分类准确率吗？

3. **伯努利实现。** 使用 BernoulliNB 类扩展代码。将词袋转换为二进制（存在/不存在），并将文本数据的准确性与 MultinomialNB 进行比较。伯努利什么时候获胜？

4. **NB 与 Logistic 回归。** 在文本数据上训练两者。从 100 个训练样本开始，增加到 10,000 个。绘制两者的准确性与训练集大小的关系。逻辑回归在什么时候超越朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建完整的垃圾邮件分类器：标记原始电子邮件文本、构建词汇、创建词袋特征、训练 MultinomialNB、评估精确度和召回率（不仅仅是准确性 - 为什么？）。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|朴素贝叶斯 | “简单概率分类器”|应用贝叶斯定理的分类器，假设给定类别，特征是条件独立的 |
|有条件独立| “功能不会互相影响” | P(A, B \| C) = P(A \| C) * P(B \| C)——一旦你知道了 C | B，你就不会再知道 A 的任何新内容了。
|拉普拉斯平滑| “加一平滑”|向每个特征添加少量计数，以防止零概率主导预测 |
|之前 | “看到数据之前你相信什么”| P(class) -- 在观察任何特征之前每个类别的概率 |
|可能性| “数据的拟合程度如何” | P(features \| class) -- 如果类别已知，则观察到这些特征的概率 |
|后| “看到数据后你相信什么”| P(class \| features) -- 观察特征后类别的更新概率 |
|生成模型 | “模拟数据的生成方式” |学习 P(X \| Y) 和 P(Y) 的模型，然后使用贝叶斯定理得到 P(Y \| X) |
|判别模型| “对决策边界进行建模”|直接学习 P(Y \| X) 的模型，无需对 X 的生成方式进行建模 |
|对数概率 | “避免下溢” |使用 log P 而不是 P 来防止许多小数的乘积在浮点中变为零 |

## 进一步阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 所有三个变体都有数学细节
- [McCallum 和 Nigam，朴素贝叶斯文本分类事件模型的比较 (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项式与伯努利文本的经典比较
- [Rennie 等人，解决朴素贝叶斯文本分类器的不良假设 (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)——对文本 NB 的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativogenic.pdf) - 证明 NB 在数据较少的情况下比 LR 收敛得更快
