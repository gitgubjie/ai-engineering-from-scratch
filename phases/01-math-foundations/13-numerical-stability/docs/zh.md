# Numerical Stability

> Floating point is a leaky abstraction. It will bite you during training, and you will not see it coming.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Implement numerically stable softmax and log-sum-exp using the max-subtraction trick
- Identify overflow, underflow, and catastrophic cancellation in floating-point computations
- Verify analytical gradients against numerical gradients using centered finite differences
- Explain why bfloat16 is preferred over float16 for training and how loss scaling prevents gradient underflow

## The Problem

Your model trains for three hours, then the loss becomes NaN. You add a print statement. The logits are fine at step 9,000. At step 9,001 they are `inf`. By step 9,002 every gradient is `nan` and training is dead.

Or: your model trains to completion but accuracy is 2% worse than the paper claims. You check everything. Architecture matches. Hyperparameters match. Data matches. The problem is that the paper used float32 and you used float16 without the right scaling. Thirty-two bits of accumulated rounding error quietly ate your accuracy.

Or: you implement cross-entropy loss from scratch. It works on small logits. When logits exceed 100, it returns `inf`. The softmax overflowed because `exp(100)` is larger than float32 can represent. Every ML framework handles this with a two-line trick. You did not know the trick existed.

Numerical stability is not a theoretical concern. It is the difference between a training run that succeeds and one that silently fails. Every serious ML bug you will debug eventually comes down to floating point.

## The Concept

### IEEE 754: How Computers Store Real Numbers

Computers store real numbers as floating point values following the IEEE 754 standard. A float has three parts: a sign bit, an exponent, and a mantissa (significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

The mantissa determines precision (how many significant digits). The exponent determines range (how large or small a number can be).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 gives you about 7 decimal digits of precision. That means it can tell apart 1.0000001 and 1.0000002, but not 1.00000001 and 1.00000002. After 7 digits, everything is rounding noise.

float16 gives you about 3 digits. The largest number it can represent is 65,504. That is disturbingly small for ML where logits, gradients, and activations routinely exceed this.

bfloat16 is Google's answer to float16's range problem. It has the same 8-bit exponent as float32 (same range, up to 3.4e38) but only 7 mantissa bits (less precision than float16). For training neural networks, range matters more than precision, so bfloat16 usually wins.

### Why 0.1 + 0.2 != 0.3

The number 0.1 cannot be represented exactly in binary floating point. In base 2, it is a repeating fraction:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 truncates this to 23 bits of mantissa. The stored value is approximately 0.100000001490116. Similarly, 0.2 is stored as approximately 0.200000002980232. Their sum is 0.300000004470348, not 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

This matters for ML because:

1. Loss comparisons like `if loss < threshold` can give wrong answers
2. Accumulating many small values (gradient updates over thousands of steps) drifts from the true sum
3. Checksums and reproducibility tests fail if you compare floats with `==`

The fix: never compare floats with `==`. Use `abs(a - b) < epsilon` or `math.isclose()`.

### Catastrophic Cancellation

When you subtract two nearly equal floating point numbers, the significant digits cancel and you are left with rounding noise promoted to leading digits.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

That is a 19% relative error from a single subtraction. In ML, this happens whenever you:

- Compute variance of data with a large mean: `E[x^2] - E[x]^2` when E[x] is large
- Subtract nearly equal log-probabilities
- Compute finite-difference gradients with too-small epsilon

The fix: rearrange formulas to avoid subtracting large, nearly equal numbers. For variance, use the Welford algorithm or center the data first. For log-probabilities, work in log-space throughout.

### Overflow and Underflow

Overflow happens when a result is too large to represent. Underflow happens when it is too small (closer to zero than the smallest representable positive number).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

The `exp()` function is the primary source of overflow in ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

The `log()` function hits the other direction:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

In ML, `exp()` appears in softmax, sigmoid, and probability computations. `log()` appears in cross-entropy, log-likelihoods, and KL divergence. The combination `log(exp(x))` is a minefield without the right tricks.

### The Log-Sum-Exp Trick

Computing `log(sum(exp(x_i)))` directly is numerically dangerous. If any `x_i` is large, `exp(x_i)` overflows. If all `x_i` are very negative, every `exp(x_i)` underflows to zero and `log(0)` is `-inf`.

The trick: subtract the maximum value before exponentiating.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Why this works: after subtracting `max(x)`, the largest exponent is `exp(0) = 1`. No overflow is possible. At least one term in the sum is 1, so the sum is at least 1, and `log(1) = 0`. No underflow to `-inf` is possible.

Proof:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Set `c = max(x)` and overflow is eliminated.

This trick appears everywhere in ML:
- Softmax normalization
- Cross-entropy loss computation
- Log-probability summation in sequence models
- Mixture of Gaussians
- Variational inference

### Why Softmax Needs the Max-Subtraction Trick

Softmax converts logits to probabilities:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Without the trick, logits of [100, 101, 102] cause overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

With the trick, subtract max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

The probabilities are identical. The computation is safe. This is not an optimization. It is a requirement for correctness.

### NaN and Inf: Detection and Prevention

`nan` (Not a Number) and `inf` (infinity) propagate virally through computation. One `nan` in a gradient update makes the weight `nan`, which makes every subsequent output `nan`. Training is dead within one step.

How `inf` appears:
- `exp()` of a large positive number
- Division by zero: `1.0 / 0.0`
- `float32` overflow in accumulations

How `nan` appears:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` of a negative number
- `log()` of a negative number
- Any arithmetic involving an existing `nan`

Detection:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Prevention strategies:

1. Clamp inputs to `exp()`: `exp(clamp(x, -80, 80))`
2. Add epsilon to denominators: `x / (y + 1e-8)`
3. Add epsilon inside `log()`: `log(x + 1e-8)`
4. Use stable implementations (log-sum-exp, stable softmax)
5. Gradient clipping to prevent weight explosion
6. Check for `nan`/`inf` after every forward pass during debugging

### Numerical Gradient Checking

Analytical gradients (from backpropagation) can have bugs. Numerical gradient checking verifies them by computing gradients with finite differences.

The centered difference formula:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

This is O(h^2) accurate, much better than the forward difference `(f(x+h) - f(x)) / h` which is only O(h).

Choosing h: too large and the approximation is wrong. Too small and catastrophic cancellation destroys the answer. `h = 1e-5` to `1e-7` is typical.

The check: compute the relative difference between analytical and numerical gradients.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Rules of thumb:
- relative_error < 1e-7: perfect, gradient is correct
- relative_error < 1e-5: acceptable, probably correct
- relative_error > 1e-3: something is wrong
- relative_error > 1: gradient is completely wrong

Always check gradients when implementing a new layer or loss function. PyTorch provides `torch.autograd.gradcheck()` for this.

### Mixed Precision Training

Modern GPUs have specialized hardware (Tensor Cores) that compute float16 matrix multiplications 2-8x faster than float32. Mixed precision training exploits this:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

The problem with pure float16 training: gradients are often very small (1e-8 or smaller). Float16 underflows anything below ~6e-8 to zero. Your model stops learning because all gradient updates are zero.

The fix is loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling adjusts the scale factor automatically. Start with a large value (65536). If gradients overflow to `inf`, halve it. If N steps pass without overflow, double it.

### bfloat16 vs float16: Why bfloat16 Wins for Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 has more precision (10 mantissa bits vs 7) but limited range (max ~65,504). bfloat16 has less precision but the same range as float32 (max ~3.4e38).

For training neural networks:

- Activations and logits regularly exceed 65,504 during training spikes. float16 overflows; bfloat16 handles it.
- Loss scaling is required with float16 but usually unnecessary with bfloat16 because its range covers the gradient magnitude spectrum.
- bfloat16 is a simple truncation of float32: drop the bottom 16 bits of the mantissa. Conversion is trivial and lossless in the exponent.

float16 is preferred for inference where values are bounded and precision matters more. bfloat16 is preferred for training where range matters more. This is why TPUs and modern NVIDIA GPUs (A100, H100) have native bfloat16 support.

### Gradient Clipping

Exploding gradients happen when gradients grow exponentially through many layers (common in RNNs, deep networks, and transformers). A single large gradient can corrupt all weights in one step.

Two types of clipping:

**Clip by value:** clamp each gradient element independently.

```
grad = clamp(grad, -max_val, max_val)
```

Simple but can change the direction of the gradient vector.

**Clip by norm:** scale the entire gradient vector so its norm does not exceed a threshold.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Preserves the direction of the gradient. This is what `torch.nn.utils.clip_grad_norm_()` does. It is the standard choice.

Typical values: `max_norm=1.0` for transformers, `max_norm=0.5` for RL, `max_norm=5.0` for simpler networks.

Gradient clipping is not a hack. It is a safety mechanism. Without it, a single outlier batch can produce a gradient large enough to ruin weeks of training.

### Normalization Layers as Numerical Stabilizers

Batch normalization, layer normalization, and RMS normalization are usually presented as regularizers that help training converge. They are also numerical stabilizers.

Without normalization, activations can grow or shrink exponentially through layers:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization recenters and rescales activations at every layer:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

The `epsilon` (typically 1e-5) prevents division by zero when all activations are identical. The learned parameters `gamma` and `beta` let the network restore any scale it needs.

This keeps values in a numerically safe range throughout the network, preventing both overflow in the forward pass and gradient explosion in the backward pass.

### Common ML Numerical Bugs

**Bug: Loss is NaN after a few epochs.**
Cause: logits grew too large, softmax overflowed. Or learning rate is too high and weights diverged.
Fix: use stable softmax (max subtraction), reduce learning rate, add gradient clipping.

**Bug: Loss is stuck at log(num_classes).**
Cause: model outputs are near-uniform probabilities. Often means gradients are vanishing or the model is not learning at all.
Fix: check that data labels are correct, verify the loss function, check for dead ReLUs.

**Bug: Validation accuracy is lower than expected by 1-3%.**
Cause: mixed precision without proper loss scaling. Gradient underflow silently zeroes out small updates.
Fix: enable dynamic loss scaling, or switch to bfloat16.

**Bug: Gradient norms are 0.0 for some layers.**
Cause: dead ReLU neurons (all inputs negative), or float16 underflow.
Fix: use LeakyReLU or GELU, use gradient scaling, check weight initialization.

**Bug: Model works on one GPU but gives different results on another.**
Cause: non-deterministic floating point accumulation order. GPU parallel reductions sum in different orders on different hardware, and floating point addition is not associative.
Fix: accept small differences (1e-6), or set `torch.use_deterministic_algorithms(True)` and accept the speed penalty.

**Bug: `exp()` returns `inf` in loss computation.**
Cause: raw logits passed to `exp()` without the max-subtraction trick.
Fix: use `torch.nn.functional.log_softmax()` which implements log-sum-exp internally.

**Bug: Training diverges after switching from float32 to float16.**
Cause: float16 cannot represent gradient magnitudes below 6e-8 or activations above 65,504.
Fix: use mixed precision with loss scaling (AMP), or use bfloat16 instead.

## Build It

### Step 1: Demonstrate floating point precision limits

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: Implement naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: Implement stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: Implement stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mixed precision simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

有关演示所有边缘情况的完整实现，请参阅“code/numerical.py”。

## 发货

本课产生：
- `code/numeric.py` 具有稳定的 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md` 用于诊断训练中的 NaN/Inf 和数值问题

这些稳定的实现在构建训练循环时的第 3 阶段和实施注意力机制时的第 4 阶段中重新出现。

## 练习

1. **灾难性取消。** 使用 float32 中的朴素公式“E[x^2] - E[x]^2”计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 的在线算法进行计算。将误差与真实方差 (0.6667) 进行比较。

2. **精确搜索。** 在 Python 中找到满足“1.0 + x == 1.0”的最小正 float32 值“x”。这是机器 epsilon。验证它与“numpy.finfo(numpy.float32).eps”匹配。

3. **Log-sum-exp 边缘情况。** 使用以下条件测试您的“logsumexp_stable”函数：(a) 所有值相等，(b) 一个值远大于其余值，(c) 所有值都非常负 (-1000)。验证它在幼稚版本失败的情况下给出正确的结果。

4. **对神经网络层进行梯度检查。** 实现单个线性层“y = Wx + b”及其分析后向传递。使用“numeric_gradient”验证 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 使用 float16 模拟训练：在 [1e-9, 1e-3] 范围内创建随机梯度，转换为 float16，并测量哪些分数变为零。然后应用损失缩放（乘以 1024），转换为 float16，缩小，并再次测量零分数。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
| IEEE 754 | IEEE 754 “浮动标准”|定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
|机器epsilon | “精度极限”|给定浮点格式中 1.0 + e != 1.0 的最小值 e。对于float32来说，大约是1.19e-7。 |
|灾难性取消 | “减法的精度损失”|当减去几乎相等的浮点数时，有效数字会被抵消，并且舍入噪声在结果中占主导地位。 |
|溢出| “数字太大”|结果超出最大可表示值并变为 inf。 exp(89) 溢出 float32。 |
|下溢| “数量太小”|结果比最小可表示正数更接近零，并且变为 0.0。 exp(-104) 下溢 float32。 |
|对数和表达式技巧 | “先减去最大值”|通过分解 exp(max(x)) 来计算 log(sum(exp(x))) 以防止上溢和下溢。用于 softmax、交叉熵和对数概率数学。 |
|稳定的softmax | “不会爆炸的Softmax”|在求幂之前减去 max(logits)。数值结果相同，不可能溢出。 |
|梯度检查 | “验证你的反向传播” |将反向传播的分析梯度与有限差分的数值梯度进行比较，以捕获实现错误。 |
|混合精度 | “float16 向前，float32 向后”|对于速度关键的操作使用较低精度的浮点数，对于数字敏感的操作使用高精度的浮点数。典型的加速比是 2-3 倍。 |
|损失缩放| “防止梯度下溢”|在反向传播之前将损失乘以一个大常数，以便梯度保持在 float16 的可表示范围内，然后在权重更新之前除以相同的常数。 |
| bfloat16 | 《脑浮点》| Google 的 16 位格式，具有 8 个指数位（范围与 float32 相同）和 7 个尾数位（精度低于 float16）。首选用于培训。 |
|渐变裁剪| “限制梯度范数” |缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
|南| “不是一个数字” |来自未定义操作的特殊浮点值（0/0、inf-inf、sqrt(-1)）。传播到所有后续算术。 |
|信息 | “无限”|溢出或除以零的特殊浮点值。可以组合产生 NaN (inf - inf, inf * 0)。 |
|数值梯度| “暴力衍生”|通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。验证缓慢但可靠。 |

## 进一步阅读

- [每个计算机科学家应该了解浮点算术 (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威的参考资料，密集但完整
- [混合精度训练 (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— NVIDIA 论文，介绍了 float16 训练的损失缩放
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) -- 为什么 Google 为 TPU 选择这种格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点和舍入误差的算法
