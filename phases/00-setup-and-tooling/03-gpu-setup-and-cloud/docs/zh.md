# GPU Setup & Cloud

> Training on CPU is fine for learning. Training for real needs a GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Learning Objectives

- Verify local GPU availability using `nvidia-smi` and PyTorch's CUDA API
- Configure Google Colab with a T4 GPU for free cloud-based experiments
- Benchmark matrix multiplication on CPU vs GPU and measure the speedup
- Estimate the largest model that fits in your VRAM using the fp16 rule of thumb

## The Problem

Most lessons in phases 1-3 run fine on CPU. But once you start training CNNs, transformers, or LLMs (phases 4+), you need GPU acceleration. A training run that takes 8 hours on CPU takes 10 minutes on GPU.

You have three options: local GPU, cloud GPU, or Google Colab (free).

## The Concept

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## Build It

### Option 1: Local NVIDIA GPU

Check if you have one:

```bash
nvidia-smi
```

Install PyTorch with CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Run `!nvidia-smi` to verify

Upload notebooks from this course directly to Colab.

### Option 3: Cloud GPU

For Lambda Labs, RunPod, or Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### No GPU? No problem.

Most lessons work on CPU. The ones that need GPU will say so and include Colab links.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试并比较 CPU 与 GPU 时间
2.如果没有GPU，可以在Google Colab上运行一下并进行比较
3. 检查您有多少 GPU 内存并估计您可以容纳的最大模型（经验法则：对于 fp16，每个参数 2 个字节）

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
| CUDA | 《GPU 编程》| NVIDIA 的并行计算平台可让您在 GPU 上运行代码 |
|显存 | “GPU内存” | GPU 上的视频 RAM，与系统 RAM 分开。限制模型尺寸。 |
| FP16 | “半精度”| 16 位浮点，使用 fp32 一半的内存，精度损失最小 |
|张量核心 | “快速矩阵硬件”|用于矩阵乘法的专用 GPU 内核，比常规内核快 4-8 倍 |
