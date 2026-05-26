# Complex Numbers for AI

> The square root of -1 is not imaginary. It is the key to rotations, frequencies, and half of signal processing.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04 (linear algebra, calculus)
**Time:** ~60 minutes

## Learning Objectives

- Perform complex arithmetic (add, multiply, divide, conjugate) in both rectangular and polar form
- Apply Euler's formula to convert between complex exponentials and trigonometric functions
- Implement the Discrete Fourier Transform using complex roots of unity
- Explain how complex rotations underlie RoPE and sinusoidal positional encodings in transformers

## The Problem

You open a paper on Fourier transforms and there is `i` everywhere. You look at transformer positional encodings and see `sin` and `cos` at different frequencies -- the real and imaginary parts of complex exponentials. You read about quantum computing and find everything expressed in complex vector spaces.

Complex numbers seem abstract. A number system built on the square root of -1 feels like a mathematical trick. But it is not a trick. It is the natural language of rotations and oscillations. Every time something spins, vibrates, or oscillates, complex numbers are the right tool.

Without understanding complex numbers, you cannot understand the Discrete Fourier Transform. You cannot understand FFT. You cannot understand how RoPE (Rotary Position Embedding) works in modern language models. You cannot understand why sinusoidal positional encodings in the original Transformer paper use the frequencies they do.

This lesson builds complex arithmetic from scratch, connects it to geometry, and shows you exactly where complex numbers appear in machine learning.

## The Concept

### What is a complex number?

A complex number has two parts: a real part and an imaginary part.

```
z = a + bi

where:
  a is the real part
  b is the imaginary part
  i is the imaginary unit, defined by i^2 = -1
```

That is it. You extend the number line into a plane. The real numbers sit on one axis. The imaginary numbers sit on the other. Every complex number is a point in this plane.

### Complex arithmetic

**Addition.** Add the real parts together, add the imaginary parts together.

```
(a + bi) + (c + di) = (a + c) + (b + d)i

Example: (3 + 2i) + (1 + 4i) = 4 + 6i
```

**Multiplication.** Use the distributive law and remember that i^2 = -1.

```
(a + bi)(c + di) = ac + adi + bci + bdi^2
                 = ac + adi + bci - bd
                 = (ac - bd) + (ad + bc)i

Example: (3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2
                            = 3 + 14i - 8
                            = -5 + 14i
```

**Conjugate.** Flip the sign of the imaginary part.

```
conjugate of (a + bi) = a - bi
```

The product of a complex number and its conjugate is always real:

```
(a + bi)(a - bi) = a^2 + b^2
```

**Division.** Multiply numerator and denominator by the conjugate of the denominator.

```
(a + bi) / (c + di) = (a + bi)(c - di) / (c^2 + d^2)
```

This eliminates the imaginary part from the denominator, giving you a clean complex number.

### The complex plane

The complex plane maps every complex number to a 2D point. The horizontal axis is the real axis, the vertical axis is the imaginary axis.

```
z = 3 + 2i  corresponds to the point (3, 2)
z = -1 + 0i corresponds to the point (-1, 0) on the real axis
z = 0 + 4i  corresponds to the point (0, 4) on the imaginary axis
```

A complex number is simultaneously a point and a vector from the origin. This dual interpretation is what makes complex numbers useful for geometry.

### Polar form

Any point in the plane can be described by its distance from the origin and its angle from the positive real axis.

```
z = r * (cos(theta) + i*sin(theta))

where:
  r = |z| = sqrt(a^2 + b^2)     (magnitude, or modulus)
  theta = atan2(b, a)             (phase, or argument)
```

Rectangular form (a + bi) is good for addition. Polar form (r, theta) is good for multiplication.

**Multiplication in polar form.** Multiply the magnitudes, add the angles.

```
z1 = r1 * e^(i*theta1)
z2 = r2 * e^(i*theta2)

z1 * z2 = (r1 * r2) * e^(i*(theta1 + theta2))
```

This is why complex numbers are perfect for rotations. Multiplying by a complex number with magnitude 1 is a pure rotation.

### Euler's formula

The bridge between complex exponentials and trigonometry:

```
e^(i*theta) = cos(theta) + i*sin(theta)
```

This is the most important formula in this lesson. When theta = pi:

```
e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i = -1

Therefore: e^(i*pi) + 1 = 0
```

Five fundamental constants (e, i, pi, 1, 0) linked in one equation.

### Why Euler's formula matters for ML

Euler's formula says that `e^(i*theta)` traces the unit circle as theta varies. At theta = 0, you are at (1, 0). At theta = pi/2, you are at (0, 1). At theta = pi, you are at (-1, 0). At theta = 3*pi/2, you are at (0, -1). A full rotation is theta = 2*pi.

This means complex exponentials ARE rotations. And rotations are everywhere in signal processing and ML.

### Connection to 2D rotations

Multiplying the complex number (x + yi) by e^(i*theta) rotates the point (x, y) by angle theta around the origin.

```
Rotation via complex multiplication:
  (x + yi) * (cos(theta) + i*sin(theta))
  = (x*cos(theta) - y*sin(theta)) + (x*sin(theta) + y*cos(theta))i

Rotation via matrix multiplication:
  [cos(theta)  -sin(theta)] [x]   [x*cos(theta) - y*sin(theta)]
  [sin(theta)   cos(theta)] [y] = [x*sin(theta) + y*cos(theta)]
```

They produce identical results. Complex multiplication IS 2D rotation. The rotation matrix is just complex multiplication written in matrix notation.

```mermaid
graph TD
    subgraph "Complex Multiplication = 2D Rotation"
        A["z = x + yi<br/>Point (x, y)"] -->|"multiply by e^(i*theta)"| B["z' = z * e^(i*theta)<br/>Point rotated by theta"]
    end
    subgraph "Equivalent Matrix Form"
        C["vector [x, y]"] -->|"multiply by rotation matrix"| D["[x cos theta - y sin theta,<br/> x sin theta + y cos theta]"]
    end
    B -.->|"same result"| D
```

### Phasors and rotating signals

A complex exponential e^(i*omega*t) is a point rotating around the unit circle at angular frequency omega. As t increases, the point traces the circle.

The real part of this rotating point is cos(omega*t). The imaginary part is sin(omega*t). A sinusoidal signal is the shadow of a rotating complex number.

```
e^(i*omega*t) = cos(omega*t) + i*sin(omega*t)

Real part:      cos(omega*t)    -- a cosine wave
Imaginary part: sin(omega*t)    -- a sine wave
```

This is the phasor representation. Instead of tracking a wiggly sine wave, you track a smoothly rotating arrow. Phase shifts become angle offsets. Amplitude changes become magnitude changes. Addition of signals becomes vector addition.

### Roots of unity

The N-th roots of unity are N points equally spaced on the unit circle:

```
w_k = e^(2*pi*i*k/N)    for k = 0, 1, 2, ..., N-1
```

For N = 4, the roots are: 1, i, -1, -i (the four compass points).
For N = 8, you get the four compass points plus the four diagonals.

Roots of unity are the foundation of the Discrete Fourier Transform. The DFT decomposes a signal into components at these N equally-spaced frequencies.

### Connection to the DFT

The Discrete Fourier Transform of a signal x[0], x[1], ..., x[N-1] is:

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)
```

Each X[k] measures how much the signal correlates with the k-th root of unity -- a complex sinusoid at frequency k. The DFT breaks a signal into N rotating phasors and tells you the amplitude and phase of each one.

### Why i is not imaginary

The word "imaginary" is a historical accident. Descartes used it dismissively. But i is no more imaginary than negative numbers were when people first rejected them. Negative numbers answer "what do you subtract 5 from 3 to get?" The imaginary unit answers "what do you square to get -1?"

More usefully: i is a 90-degree rotation operator. Multiply a real number by i once, you rotate 90 degrees to the imaginary axis. Multiply by i again (i^2), you rotate another 90 degrees -- now you are pointing in the negative real direction. That is why i^2 = -1. It is not mysterious. It is a half-turn built from two quarter-turns.

This is why complex numbers are everywhere in engineering. Anything that rotates -- electromagnetic waves, quantum states, signal oscillations, positional encodings -- is naturally described by complex numbers.

### Complex exponentials vs trigonometric functions

Before Euler's formula, engineers wrote signals as A*cos(omega*t + phi) -- amplitude A, frequency omega, phase phi. This works but makes arithmetic painful. Adding two cosines with different phases requires trigonometric identities.

With complex exponentials, the same signal is A*e^(i*(omega*t + phi)). Adding two signals is just adding two complex numbers. Multiplying (modulating) is just multiplying magnitudes and adding angles. Phase shifts become angle additions. Frequency shifts become multiplications by phasors.

The entire field of signal processing switched to complex exponential notation because the math is cleaner. The "real signal" is always just the real part of the complex representation. The imaginary part is carried along as bookkeeping, making all the algebra work out naturally.

### Connection to transformers

**Sinusoidal positional encodings** (original Transformer paper):

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

The sin and cos pairs are the real and imaginary parts of complex exponentials at different frequencies. Each frequency provides a different "resolution" for encoding position. Low frequencies change slowly (coarse position). High frequencies change quickly (fine position). Together they give each position a unique frequency fingerprint.

**RoPE (Rotary Position Embedding)** takes this further. It explicitly multiplies query and key vectors by complex rotation matrices. The relative position between two tokens becomes a rotation angle. Attention is computed using these rotated vectors, making the model sensitive to relative position through complex multiplication.

| Operation | Algebraic Form | Geometric Meaning |
|-----------|---------------|-------------------|
| Addition | (a+c) + (b+d)i | Vector addition in the plane |
| Multiplication | (ac-bd) + (ad+bc)i | Rotate and scale |
| Conjugate | a - bi | Reflect over real axis |
| Magnitude | sqrt(a^2 + b^2) | Distance from origin |
| Phase | atan2(b, a) | Angle from positive real axis |
| Division | multiply by conjugate | Reverse rotation and rescale |
| Power | r^n * e^(i*n*theta) | Rotate n times, scale by r^n |

```mermaid
graph LR
    subgraph "Unit Circle"
        direction TB
        U1["e^(i*0) = 1"] -.-> U2["e^(i*pi/2) = i"]
        U2 -.-> U3["e^(i*pi) = -1"]
        U3 -.-> U4["e^(i*3pi/2) = -i"]
        U4 -.-> U1
    end
    subgraph "Applications"
        A1["Euler's formula:<br/>e^(i*theta) = cos + i*sin"]
        A2["DFT uses roots of unity:<br/>e^(2*pi*i*k/N)"]
        A3["RoPE uses rotation:<br/>q * e^(i*m*theta)"]
    end
    U1 --> A1
    U1 --> A2
    U1 --> A3
```

## Build It

### Step 1: Complex class

Build a Complex number class that supports arithmetic, magnitude, phase, and conversion between rectangular and polar forms.

```python
import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)
```

### Step 2: Polar conversion and Euler's formula

```python
def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))
```

Verify: `euler(theta).magnitude()` should always be 1.0. `euler(0)` should give (1, 0). `euler(pi)` should give (-1, 0).

### Step 3: Rotation

Rotating a point (x, y) by angle theta is one complex multiplication:

```python
point = Complex(3, 4)
rotated = point * euler(math.pi / 4)
```

The magnitude stays the same. Only the angle changes.

### Step 4: DFT from complex arithmetic

```python
def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result
```

This is the O(N^2) DFT. Each output X[k] is the sum of the signal samples multiplied by roots of unity.

### Step 5: Inverse DFT

The inverse DFT reconstructs the original signal from its spectrum. The only changes from the forward DFT: flip the sign in the exponent and divide by N.

```python
def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

This gives you perfect reconstruction. Apply DFT, then IDFT, and you get back the original signal to machine precision. No information is lost.

### Step 6: Roots of unity

```python
def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]
```

Verify two properties:
- Every root has magnitude exactly 1.
- The sum of all N roots is zero (they cancel out by symmetry).

These properties are what make the DFT invertible. The roots of unity form an orthogonal basis for the frequency domain.

## Use It

Python has built-in complex number support. The literal `j` represents the imaginary unit.

```python
z = 3 + 2j
w = 1 + 4j

print(z + w)
print(z * w)
print(abs(z))

import cmath
print(cmath.phase(z))
print(cmath.exp(1j * cmath.pi))
```

For arrays, numpy handles complex numbers natively:

```python
import numpy as np

z = np.array([1+2j, 3+4j, 5+6j])
print(np.abs(z))
print(np.angle(z))
print(np.conj(z))
print(np.real(z))
print(np.imag(z))

signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(128, d=1/128)
```

## 发货

运行“code/complex_numbers.py”以生成“outputs/skill-complex-arithmetic.md”。

## 练习

1. **手动进行复杂算术。** 计算 (2 + 3i) * (4 - i) 并使用代码进行验证。然后计算 (5 + 2i) / (1 - 3i)。在复平面上绘制两个结果，并检查乘法是否旋转并缩放了第一个数字。

2. **旋转顺序。** 从点 (1, 0) 开始。乘以 e^(i*pi/6) 十二次。验证 12 次乘法后是否返回 (1, 0)。打印每一步的坐标并确认它们描绘的是规则的 12 边形。

3. **已知信号的 DFT。** 创建一个信号，该信号是在 32 个点采样的 sin(2*pi*3*t) 和 0.5*sin(2*pi*7*t) 之和。运行 DFT。验证幅度谱在频率 3 和 7 处有峰值，7 处的峰值是 3 处峰值高度的一半。

4. **统一根可视化。** 计算八次统一根。验证它们的总和为零。验证将任何根乘以原根 e^(2*pi*i/8) 都会得到下一个根。

5. **旋转矩阵等价。** 对于 10 个随机角度和 10 个随机点，验证复数乘法与 2x2 旋转矩阵的矩阵向量乘法给出的结果相同。打印最大数值差。

## 关键术语

|术语 |这意味着什么 |
|------|----------------|
|复数 |数字 a + bi，其中 a 是实部，b 是虚部，i^2 = -1 |
|虚数单位|数字 i，定义为 i^2 = -1。在哲学意义上不是虚构的——它是一个旋转运算符 |
|复平面 | x 轴为实轴、y 轴为虚轴的 2D 平面。也称为阿尔甘飞机 |
|幅度（模数）|距原点的距离：sqrt(a^2 + b^2)。写为\|z\| |
|阶段（参数）|与正实轴的角度：atan2(b, a)。写为 arg(z) |
|共轭|穿过实轴的镜像： a + bi 的共轭是 a - bi |
|极性形式|将 z 表示为 r * e^(i*theta) 而不是 a + bi。让乘法变得简单 |
|欧拉公式 | e^(i*theta) = cos(theta) + i*sin(theta)。将指数与三角函数联系起来 |
|相量|表示正弦信号的旋转复数 e^(i*omega*t) |
|团结之根| N 个复数 e^(2*pi*i*k/N)（k = 0 到 N-1）。单位圆上N个等距点|
|密度泛函理论|离散傅里叶变换。使用单位根将信号分解为复杂的正弦分量 |
|绳索 |旋转位置嵌入。使用复数乘法来编码变压器注意力中的相对位置 |

## 进一步阅读

- [欧拉公式的视觉介绍](https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/) - 无需繁重的符号即可建立几何直觉
- [Su et al.: RoFormer (2021)](https://arxiv.org/abs/2104.09864) - 介绍使用复杂旋转的旋转位置嵌入的论文
- [Vaswani 等人：Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 具有正弦位置编码的原始 Transformer 论文
- [3Blue1Brown：介绍群论的欧拉公式](https://www.youtube.com/watch?v=mvmuCPvRoWQ) - 为什么 e^(i*pi) = -1 的直观解释
- [Needham: Visual Complex Analysis](https://global.oup.com/academic/product/visual-complex-analysis-9780198534464) - 复数的最佳视觉处理，充满几何洞察力
- [Strang：线性代数导论，第 1 章10](https://math.mit.edu/~gs/linearalgebra/) - 线性代数和特征值背景下的复数
