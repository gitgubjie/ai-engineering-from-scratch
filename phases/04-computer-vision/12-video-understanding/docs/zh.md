# Video Understanding — Temporal Modeling

> A video is a sequence of images plus the physics that connects them. Every video model either treats time as an extra axis (3D conv), a sequence to attend over (transformer), or a feature to extract once and pool (2D+pool).

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**Time:** ~45 minutes

## Learning Objectives

- Distinguish the three main video-modelling approaches (2D+pool, 3D conv, spatio-temporal transformer) and predict their cost and accuracy trade-offs
- Implement frame sampling, temporal pooling, and a 2D+pool baseline classifier in PyTorch
- Explain why I3D's "inflated" 3D kernels transfer well from ImageNet weights and what a factorised (2+1)D conv does differently
- Read the standard action-recognition datasets and metrics: Kinetics-400/600, UCF101, Something-Something V2; top-1 accuracy at the clip and video level

## The Problem

A 30-second video at 30 fps is 900 images. Naively, video classification is image classification run 900 times followed by some kind of aggregation. That works when the action is visible in almost every frame (sports, cooking, exercise videos) and fails badly when the action is defined by motion itself: "pushing something from left to right" looks like two still objects in every single frame.

The core question for every video architecture is: when does temporal structure get modelled, and how? The answer drives everything else — compute cost, pretraining strategy, whether you can reuse ImageNet weights, what datasets the model trains on.

This lesson is deliberately shorter than the static-image lessons. The core image machinery is already in place, and video understanding is mostly about the temporal story: sampling, modelling, and aggregating.

## The Concept

### The three architectural families

```mermaid
flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + pool

Take a 2D CNN (ResNet, EfficientNet, ViT). Run it independently on every sampled frame. Average (or max-pool, or attention-pool) the per-frame embeddings. Feed the pooled vector to a classifier.

Pros:
- ImageNet pretraining transfers directly.
- Simplest to implement.
- Cheap: T frames * single-image inference cost.

Cons:
- Cannot model motion. Action = aggregate of appearances.
- Temporal pooling is order-invariant; "open door" and "close door" look the same.

When to use: appearance-heavy tasks, transfer learning on small video datasets, initial baselines.

### 3D convolutions

Replace 2D (H, W) kernels with 3D (T, H, W) kernels. The network convolves over both space and time. Early family: C3D, I3D, SlowFast.

I3D trick: take a pretrained 2D ImageNet model, "inflate" each 2D kernel by copying it along a new time axis. A 3x3 2D conv becomes a 3x3x3 3D conv. This gives the 3D model strong pretrained weights instead of training from scratch.

Pros:
- Directly models motion.
- I3D inflation gives free transfer learning.

Cons:
- T/8 more FLOPs than the 2D counterpart (for temporal kernel of 3 stacked 3 times).
- Temporal kernels are small; long-range motion needs a pyramid or dual-stream approach.

When to use: action recognition where motion is the signal (Something-Something V2, Kinetics with motion-heavy classes).

### Spatio-temporal transformers

Tokenise the video into a grid of space-time patches and attend across all of them. TimeSformer, ViViT, Video Swin, VideoMAE.

Attention patterns that matter:
- **Joint** — one big attention over (t, h, w). Quadratic in `T*H*W`; expensive.
- **Divided** — two attentions per block: one over time, one over space. Linear-ish scaling.
- **Factorised** — time attention alternates with space attention across blocks.

Pros:
- SOTA accuracy on every major benchmark.
- Transfers from image transformers (ViT) via patch inflation.
- Supports long-context video via sparse attention.

Cons:
- Compute-hungry.
- Requires careful attention pattern choice or runtime balloons.

When to use: large datasets, high-fidelity video understanding, multi-modal video+text tasks.

### Frame sampling

A 10-second clip at 30 fps is 300 frames; feeding all 300 to any model is wasteful. Standard strategies:

- **Uniform sampling** — pick T frames evenly across the clip. Default for 2D+pool.
- **Dense sampling** — random contiguous T-frame window. Common for 3D convs because motion requires neighbouring frames.
- **Multi-clip** — sample multiple T-frame windows from the same video, classify each, average predictions at test time.

T is usually 8, 16, 32, or 64. Higher T = more temporal signal at more compute.

### Evaluation

Two levels:
- **Clip-level accuracy** — model sees one T-frame clip, reports top-k.
- **Video-level accuracy** — average clip-level predictions across multiple clips per video; higher and more stable.

Always report both. A model that scores 78% clip / 82% video is relying heavily on test-time averaging; one that scores 80% / 81% is more robust per-clip.

### Datasets you will meet

- **Kinetics-400 / 600 / 700** — the general-purpose action dataset. 400k clips; YouTube URLs (many now dead).
- **Something-Something V2** — motion-defined actions ("moving X from left to right"). Cannot be solved by 2D+pool.
- **UCF-101**, **HMDB-51** — older, smaller, still reported.
- **AVA** — action *localisation* in space and time; harder than classification.

## Build It

### Step 1: Frame sampler

Uniform and dense samplers that work on a list of frames (or a video tensor).

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

Both return `T` indices that you use to slice the video tensor.

### Step 2: A 2D+pool baseline

Run a 2D ResNet-18 over every frame, average-pool features, classify.

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

Eleven million parameters, ImageNet pretrained, runs per-frame, averages, classifies. This baseline is often within 5-10 points of proper 3D models on appearance-heavy tasks — sometimes better, because it reuses a stronger ImageNet backbone.

### Step 3: An I3D-style inflated 3D conv

Turn a single 2D conv into a 3D conv by repeating weights along a new time axis.

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

The division by `time_kernel` keeps the activation magnitudes roughly constant — important for not breaking batch-norm statistics on the first pass.

### Step 4: Factorised (2+1)D conv

Split a 3D conv into a 2D (spatial) and a 1D (temporal) conv. Same receptive field, fewer parameters, better accuracy on some benchmarks.

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

完整的 R(2+1)D 网络与 ResNet-18 相同，其中每个 3x3 卷积都替换为“Conv2Plus1D”。

## 使用它

两个库涵盖制作视频作品：

- `torchvision.models.video` — R(2+1)D、MViT、Swin3D，具有预训练的动力学权重。与图像模型相同的 API。
- `pytorchvideo` (元) — 模型动物园、Kinetics / SSv2 / AVA 的数据加载器、标准转换。

对于视觉语言视频模型（视频字幕、视频 QA），请使用“transformers”（“VideoMAE”、“VideoLLaMA”、“InternVideo”）。

## 发货

本课产生：

- `outputs/prompt-video-architecture-picker.md` — 根据外观与运动、数据集大小和计算预算选择 2D+pool / I3D / (2+1)D / Transformer 的提示。
- `outputs/skill-frame-sampler-auditor.md` — 一项检查视频管道采样器并标记常见错误的技能：离一索引、当 `num_frames < T` 时采样不均匀、缺乏保留宽高比的裁剪等。

## 练习

1. **（简单）** 计算 T=8 的 FramePool 与 T=8 的 I3D 样式 3D ResNet 的 FLOP（大约）。证明为什么 2D+pool 便宜 3-5 倍。
2. **（中）** 生成合成视频数据集：沿随机方向移动的随机球，由运动方向标记（“从左到右”、“从右到左”、“对角线向上”）。在其上训练 FramePool。证明它达到了近乎偶然的精度，证明仅靠外观不足以完成运动任务。
3. **（困难）** 通过用“Conv2Plus1D”替换 ResNet-18 中的每个 Conv2d 来构建 R(2+1)D-18。从 ImageNet 预训练的 ResNet-18 中增加第一个卷积的权重。使用练习 2 中的运动数据集进行训练并击败 FramePool。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
| 2D + 台球 | “每帧分类器” |在每个采样帧上运行 2D CNN，跨时间平均池特征，分类 |
| 3D 卷积 | “时空内核”|卷积 (T, H, W) 的内核；可以本地模拟运动|
|通货膨胀 | “将 2D 权重提升到 3D”|通过沿新时间轴重复 2D 卷积的权重来初始化 3D 卷积权重，然后除以 kernel_T 以保留激活比例 |
| (2+1)D | “因式分解转化” |将3D分割成2D空间+1D时间；参数更少，| 之间额外的非线性
|注意力分散| “时间然后空间” |每层有两个注意力的 Transformer 块：一个在同一帧的令牌上，一个在同一位置的令牌上 |
|剪辑| “T型窗”| T 帧的采样子序列；视频模型消耗的单位 |
|剪辑与视频准确性 | “两个评估设置”|剪辑 = 每个视频一个样本，视频 = 多个采样剪辑的平均值 |
|动力学| “视频的 ImageNet”| 400-700 个动作类、30 万多个 YouTube 剪辑、标准视频预训练语料库 |

## 进一步阅读

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) — 介绍通货膨胀和动力学数据集
- [R(2+1)D：仔细观察时空卷积 (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) — 因式分解卷积，仍然是一个强大的基线
- 【TimeSformer：时空注意力就是你所需要的吗？ (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 第一个强大的视频转换器
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) — 视频的屏蔽自动编码器预训练；当前占主导地位的预训练配方
