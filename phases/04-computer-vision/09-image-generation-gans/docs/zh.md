# Image Generation — GANs

> A GAN is two neural networks in a fixed game. One draws, one critiques. They get better together until the drawings fool the critic.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (Optimizers), Phase 3 Lesson 07 (Regularization)
**Time:** ~75 minutes

## Learning Objectives

- Explain the minimax game between generator and discriminator and why the equilibrium corresponds to p_model = p_data
- Implement a DCGAN in PyTorch and get it to generate coherent 32x32 synthetic images in under 60 lines
- Stabilise GAN training with the three standard tricks: non-saturating loss, spectral norm, TTUR (two-timescale update rule)
- Read training curves that distinguish healthy convergence from mode collapse, oscillation, and discriminator-wins-completely

## The Problem

Classification teaches a network to map images to labels. Generation inverts the problem: sample new images that look like they came from the same distribution. There is no "correct" output you can diff against; there is only a distribution you want to mimic.

The standard loss functions (MSE, cross-entropy) cannot measure "did this sample come from the real distribution." Minimising per-pixel error produces blurry averages, not realistic samples. The breakthrough was to learn the loss: train a second network whose job is to tell real from fake, and use its judgement to push the generator.

GANs (Goodfellow et al., 2014) defined that framework. By 2018 StyleGAN was producing 1024x1024 faces indistinguishable from photographs. Diffusion models have since taken the throne on quality and controllability, but every trick that makes diffusion practical — normalisation choices, latent spaces, feature losses — was first understood on GANs.

## The Concept

### The two networks

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

The **generator** G takes a vector of noise `z` and outputs an image. The **discriminator** D takes an image and outputs a single scalar: the probability that the image is real.

### The game

G wants D to be wrong. D wants to be right. Formally:

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Read right to left: D is maximising accuracy on real (`log D(real)`) and fake (`log (1 - D(fake))`) images. G is minimising D's accuracy on fakes — it wants `D(G(z))` to be high.

Goodfellow proved that this minimax has a global equilibrium where `p_G = p_data`, D outputs 0.5 everywhere, and the Jensen-Shannon divergence between generated and real distributions is zero. The hard part is getting there.

### Non-saturating loss

The form above is numerically unstable. Early in training, `D(G(z))` is near zero for every fake, so `log(1 - D(G(z)))` has vanishing gradients with respect to G. The fix: flip G's loss.

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating
```

Now when `D(G(z))` is near zero, G's loss is large and its gradient is informative. Every modern GAN trains with this variant.

### DCGAN architecture rules

Radford, Metz, Chintala (2015) distilled years of failed experiments into five rules that make GAN training stable:

1. Replace pooling with strided convs (both nets).
2. Use batch norm in both generator and discriminator, except output of G and input of D.
3. Remove fully connected layers on deeper architectures.
4. G uses ReLU on all layers except output (tanh for output in [-1, 1]).
5. D uses LeakyReLU (negative_slope=0.2) on all layers.

Every modern conv-based GAN (StyleGAN, BigGAN, GigaGAN) still starts from these rules and replaces pieces one at a time.

### Failure modes and their signatures

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**: G finds one image that fools D and produces only that. Fix: add minibatch discrimination, spectral norm, or label-conditioning.
- **Discriminator wins**: D becomes too strong too fast, G's gradients vanish. Fix: smaller D, lower D learning rate, or apply label smoothing on the real labels.
- **Oscillation**: the two nets trade wins without ever approaching equilibrium. Fix: TTUR (D learns faster than G by a factor of 2-4), or switch to Wasserstein loss.

### Evaluation

GANs have no ground truth, so how do you know they are working?

- **Sample inspection** — just look at 64 samples at the end of every epoch. Non-negotiable.
- **FID (Fréchet Inception Distance)** — distance between Inception-v3 feature distributions of real and generated sets. Lower is better. Community standard.
- **Inception Score** — older, more brittle; prefer FID.
- **Precision/Recall for generative models** — measures quality (precision) and coverage (recall) separately. More informative than FID alone.

For a small synthetic-data run, sample inspection is enough.

## Build It

### Step 1: Generator

A small DCGAN generator that takes 64-dim noise and produces a 32x32 image.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

Four transposed convs, each with `kernel_size=4, stride=2, padding=1` so they cleanly double spatial size. Output activations in [-1, 1] via tanh.

### Step 2: Discriminator

Mirror of the generator. LeakyReLU, strided convs, ends with a scalar logit.

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

The last conv reduces a `4x4` feature map to `1x1`. Output is a single scalar per image; apply sigmoid only during loss computation.

### Step 3: Training step

Alternate: update D once, then G once, every batch.

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

`G(z).detach()` in the D step is critical: we do not want gradients flowing into G during its update. Forgetting that is the classic beginner bug.

### Step 4: Full training loop on synthetic shapes

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))` is the DCGAN default — the low beta1 keeps the momentum term from stabilising the adversarial game too much.

### Step 5: Sampling

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

Always switch to eval mode before sampling. For DCGAN this matters because batch norm running stats are used instead of the batch's stats.

### Step 6: Spectral normalisation

A drop-in replacement for BN in the discriminator that guarantees the network is 1-Lipschitz. Fixes most "D wins too hard" failures.

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

将“Discriminator”替换为“build_sn_discriminator()”，您通常不需要 TTUR 技巧。频谱范数是您可以应用的最简单的单一鲁棒性升级。

## 使用它

对于严肃的生成，请使用预训练的权重或切换到扩散。两个标准库：

- `torch_fidelity` 无需编写自定义评估代码即可计算生成器上的 FID / IS。
- “pytorch-gan-zoo”（旧版）和“StudioGAN”测试了 DCGAN、WGAN-GP、SN-GAN、StyleGAN 和 BigGAN 的实现。

到 2026 年，GAN 仍然是以下方面的最佳选择：实时图像生成（延迟 <10 毫秒）、风格迁移、精确控制的图像到图像转换（Pix2Pix、CycleGAN）。扩散在照片真实感和文本条件方面获胜。

## 发货

本课产生：

- `outputs/prompt-gan-training-triage.md` — 读取训练曲线描述并选择故障模式（模式崩溃、D-wins、振荡）以及单个建议修复的提示。
- `outputs/skill-dcgan-scaffold.md` — 一项从 `z_dim`、目标 `image_size` 和 `num_channels` 编写 DCGAN 支架的技能，包括训练循环和样本保存程序。

## 练习

1. **（简单）** 在合成圆数据集上训练上面的 DCGAN，并在每个 epoch 结束时保存 16 个样本的网格。到哪个时期生成的圆变得明显是圆形的？
2. **（中）** 用谱范数替换鉴别器的批量范数。并排训练两个版本。哪一个收敛得更快？哪一个在三颗种子中的方差较低？
3. **（难）** 实现一个条件 DCGAN：将类标签输入 G 和 D（将 one-hot 与 G 中的噪声连接，在 D 中连接一个类嵌入通道）。对第 7 课中的合成“圆形与方形”数据集进行训练，并表明类调节通过使用特定标签进行采样来发挥作用。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|发电机 (G) | “抽奖网”|将噪声映射到图像；训练来愚弄鉴别器|
|鉴别器 (D) | “批评家”|二元分类器；训练以区分真实图像和生成图像 |
|极小极大 | “游戏” |对抗性损失的 min 超过 G，max 超过 D；平衡是 p_G = p_data |
|非饱和损耗| “数字上理智的版本”| G 的损失是 -log(D(G(z))) 而不是 log(1 - D(G(z))) 以避免在训练早期梯度消失 |
|模式崩溃| “发电机制造一件事”| G 只产生数据分布的一小部分；使用 SN、小批量区分或较大批量进行修复 |
| TTUR | “两种学习率”| D 的学习速度比 G 快，通常是 G 的 2-4 倍；稳定训练|
|谱范数| “1-Lipschitz 层”|限制每层 Lipschitz 常数的权重归一化；阻止 D 变得任意陡峭 |
| FID | “Fréchet 起始距离”|真实集和生成集的 Inception-v3 特征分布之间的距离；标准评估指标|

## 进一步阅读

- [生成对抗网络 (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) — 这篇论文开始了这一切
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434) — 使 GAN 可训练的架构规则
- [GAN 的光谱标准化 (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) — 最有用的稳定技巧
- [StyleGAN3（Karras 等人，2021）](https://arxiv.org/abs/2106.12423) — SOTA GAN；读起来就像一张收录了过去十年所有技巧的精选专辑
