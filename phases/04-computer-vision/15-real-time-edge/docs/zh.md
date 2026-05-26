# Real-Time Vision — Edge Deployment

> Edge inference is the discipline of getting a 90-accuracy model to run at 30 fps on a device with 2 GB of RAM. Every percentage point of accuracy is traded against milliseconds of latency.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 04 (Image Classification), Phase 10 Lesson 11 (Quantization)
**Time:** ~75 minutes

## Learning Objectives

- Measure inference latency, peak memory, and throughput for any PyTorch model, and read the FLOPs / params / latency trade-off
- Quantise a vision model to INT8 using PyTorch's post-training quantisation and verify accuracy loss < 1%
- Export to ONNX and compile with ONNX Runtime or TensorRT; name the three most common export failures and their fixes
- Explain when to pick MobileNetV3, EfficientNet-Lite, ConvNeXt-Tiny, or MobileViT for an edge constraint

## The Problem

A training-time vision model is a floating-point monster. 100M parameters, 10 GFLOPs per forward pass, 2 GB of VRAM. None of that fits on a phone, a car's infotainment unit, an industrial camera, or a drone. Shipping a vision system means fitting the same predictions into a budget that is 100x smaller.

Three knobs do most of the work: model choice (a smaller architecture with the same recipe), quantisation (INT8 instead of FP32), and the inference runtime (ONNX Runtime, TensorRT, Core ML, TFLite). Getting them right is the difference between a demo that runs on a workstation and a product that ships on a $30 camera module.

This lesson sets up the measurement discipline first (you cannot optimise what you cannot measure), then walks the three knobs. The goal is not to learn every edge runtime but to know what levers exist and how to verify each one does what you think.

## The Concept

### The three budgets

```mermaid
flowchart LR
    M["Model"] --> LAT["Latency<br/>ms per image"]
    M --> MEM["Memory<br/>peak MB"]
    M --> PWR["Power<br/>mJ per inference"]

    LAT --> SHIP["Ship / no-ship<br/>decision"]
    MEM --> SHIP
    PWR --> SHIP

    style LAT fill:#fecaca,stroke:#dc2626
    style MEM fill:#fef3c7,stroke:#d97706
    style PWR fill:#dbeafe,stroke:#2563eb
```

- **Latency**: p50, p95, p99. Averaging only p50 hides tail behaviour that matters for real-time systems.
- **Peak memory**: the maximum the device ever sees, not the steady-state average. Matters because OOMs are fatal on embedded targets.
- **Power / energy**: millijoules per inference on a battery-powered device. Often proxied by CPU/GPU utilisation * time.

A table of (model, latency, memory, accuracy) is what an edge decision is made from. Every cell is measured on the target device, not the workstation.

### Measurement discipline

Three rules that every edge profile should follow:

1. **Warm up** the model with 5-10 dummy forward passes before measuring. Cold caches and JIT compilation produce unrepresentative first numbers.
2. **Synchronise** GPU workloads with `torch.cuda.synchronize()` before and after the timed block. Without this you measure kernel dispatch, not kernel execution.
3. **Fix input sizes** to the production resolution. Latency on 224x224 is not latency on 512x512.

### FLOPs as a proxy

FLOPs (floating-point operations per inference) is a cheap, device-independent proxy for latency. Useful for architecture comparison, misleading as absolute wall-clock. A model with 10% more FLOPs can be 2x faster in practice because it uses hardware-friendly ops (depthwise convs compile well, large 7x7 convs do not).

Rule: use FLOPs for architecture search, use on-device latency for deployment decisions.

### Quantisation in one paragraph

Replace FP32 weights and activations with INT8. Model size drops 4x, memory bandwidth drops 4x, compute drops 2-4x on hardware that has INT8 kernels (every modern mobile SoC, every NVIDIA GPU with Tensor Cores). Accuracy loss on vision tasks is typically 0.1-1 percentage points with post-training static quantisation.

Types:

- **Dynamic** — quantise weights to INT8, activations computed in FP. Easy, small speedup.
- **Static (post-training)** — quantise weights + calibrate activation ranges on a small calibration set. Much faster than dynamic.
- **Quantisation-aware training (QAT)** — simulate quantisation during training so the model learns around it. Best accuracy, needs labelled data.

For vision, post-training static quantisation gives 95% of the benefit with 5% of the effort. Use QAT only when accuracy loss from PTQ is unacceptable.

### Pruning and distillation

- **Pruning** — remove unimportant weights (magnitude-based) or channels (structured). Works well on overparameterised models; less useful on already-compact architectures.
- **Distillation** — train a small student to mimic a large teacher's logits. Often recovers most of the accuracy lost by shrinking the model. Standard for production edge models.

### The inference runtimes

- **PyTorch eager** — slow, not for deployment. Use for development only.
- **TorchScript** — legacy. Superseded by `torch.compile` and ONNX export.
- **ONNX Runtime** — the neutral runtime. CPU, CUDA, CoreML, TensorRT, OpenVINO all have ONNX providers. Start here.
- **TensorRT** — NVIDIA's compiler. Best latency on NVIDIA GPUs (workstation and Jetson). Integrates with ONNX Runtime or standalone.
- **Core ML** — Apple's runtime for iOS/macOS. Needs `.mlmodel` or `.mlpackage`.
- **TFLite** — Google's runtime for Android/ARM. Needs `.tflite`.
- **OpenVINO** — Intel's runtime for CPU/VPU. Needs `.xml` + `.bin`.

In practice: export PyTorch -> ONNX -> pick the runtime for the target. ONNX is the lingua franca.

### Edge architecture picker

| Budget | Model | Why |
|--------|-------|-----|
| < 3M params | MobileNetV3-Small | Compiles everywhere, good baseline |
| 3-10M | EfficientNet-Lite-B0 | Best accuracy per param on TFLite |
| 10-20M | ConvNeXt-Tiny | Best accuracy-per-param, CPU-friendly |
| 20-30M | MobileViT-S or EfficientViT | Transformer with ImageNet accuracy |
| 30-80M | Swin-V2-Tiny | If stack supports window attention |

Quantise all of these to INT8 unless you have a specific reason not to.

## Build It

### Step 1: Measure latency correctly

```python
import time
import torch

def measure_latency(model, input_shape, device="cpu", warmup=10, iters=50):
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": sum(times) / len(times),
    }
```

Warm up, synchronise, use `time.perf_counter()`. Report percentiles, not just mean.

### Step 2: Parameter and FLOP counts

```python
def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def flops_estimate(model, input_shape):
    """
    Rough FLOP count for a conv/linear-only model. For production use `fvcore` or `ptflops`.
    """
    total = 0
    def conv_hook(m, inp, out):
        nonlocal total
        c_out, c_in, kh, kw = m.weight.shape
        h, w = out.shape[-2:]
        total += 2 * c_in * c_out * kh * kw * h * w
    def linear_hook(m, inp, out):
        nonlocal total
        total += 2 * m.in_features * m.out_features
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))
    for h in hooks:
        h.remove()
    return total
```

For real projects use `fvcore.nn.FlopCountAnalysis` or `ptflops`; they handle every module type correctly.

### Step 3: Post-training static quantisation

```python
def quantise_ptq(model, calibration_loader, backend="x86"):
    import torch.ao.quantization as tq
    model = model.eval().cpu()
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model
```

Three steps: configure, prepare (insert observers), calibrate with real data, convert (fuse + quantise). Requires the model to be fused (`Conv -> BN -> ReLU` -> `ConvBnReLU`), which `torch.ao.quantization.fuse_modules` handles.

### Step 4: Export to ONNX

```python
def export_onnx(model, sample_input, path="model.onnx"):
    model = model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path
```

`opset_version=17` is the safe default in 2026. `dynamic_axes` lets you run the ONNX model with arbitrary batch size.

### Step 5: Benchmark and compare regimes

```python
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def compare_regimes():
    model = mobilenet_v3_small(weights=None, num_classes=10)
    params = parameter_count(model)
    flops = flops_estimate(model, (1, 3, 224, 224))
    lat_fp32 = measure_latency(model, (1, 3, 224, 224), device="cpu")
    print(f"FP32 MobileNetV3-Small: {params:,} params  {flops/1e9:.2f} GFLOPs  "
          f"p50={lat_fp32['p50_ms']:.2f}ms  p95={lat_fp32['p95_ms']:.2f}ms")
```

对“resnet50”、“efficientnet_v2_s”和“convnext_tiny”运行相同的函数，您将获得部署决策所需的比较表。

## 使用它

生产堆栈集中在以下三个路径之一：

- **Web/无服务器**：PyTorch -> ONNX -> ONNX 运行时（CPU 或 CUDA 提供程序）。最简单，对大多数人来说足够好。
- **NVIDIA 边缘（Jetson、GPU 服务器）**：PyTorch -> ONNX -> TensorRT。最佳的延迟，最大的工程努力。
- **移动**：PyTorch -> ONNX -> Core ML (iOS) 或 TFLite (Android)。导出前进行量化。

对于测量，macOS 上的“torch-tb-profiler”、“nvprof”/“nsys”和 Instruments 提供了逐层细分。 `benchmark_app` (OpenVINO) 和 `trtexec` (TensorRT) 提供独立的 CLI 编号。

## 发货

本课产生：

- `outputs/prompt-edge-deployment-planner.md` — 在给定目标设备和延迟 SLA 的情况下选择主干、量化策略和运行时的提示。
- `outputs/skill-latency-profiler.md` — 一种编写完整的延迟基准脚本的技能，包括预热、同步、百分位数和内存跟踪。

## 练习

1. **（简单）** 在 CPU 上以 224x224 测量“resnet18”、“mobilenet_v3_small”、“efficientnet_v2_s”和“convnext_tiny”的 p50 延迟。报告该表并确定哪种架构具有最佳的每毫秒精度。
2. **（中）** 对“mobilenet_v3_small”应用训练后静态量化。报告 CIFAR-10 或类似子集上的 FP32 与 INT8 延迟和准确性损失。
3. **（困难）** 将 `convnext_tiny` 导出到 ONNX，通过 `onnxruntime` 和 `CPUExecutionProvider` 运行它，并将延迟与 PyTorch eager 基线进行比较。确定 ONNX 运行时速度更快的第一层并解释原因。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|延迟 | “多快”|从输入到输出的时间； p50/p95/p99 百分位数，不是平均值 |
|失败 | “模型尺寸”|每次前向传递的浮点运算；计算成本的粗略代理|
| INT8 量化 | “8 位” |将 FP32 权重/激活替换为 8 位整数；体积缩小约 4 倍，速度提高 2-4 倍 |
| PTQ | “训练后量化” |量化经过训练的模型，无需重新训练；简单，通常就够了 |
|快讯 | “量化感知训练” |在训练期间模拟量化；最佳准确度，需要标记数据 |
| ONNX | “中性格式” |每个主流推理运行时都支持的模型交换格式 |
|张量RT | “NVIDIA 编译器”|将 ONNX 编译为 NVIDIA GPU 的优化引擎 |
|蒸馏| “老师->学生” |训练一个小模型来模仿大模型的 logits；恢复大部分失去的准确性|

## 进一步阅读

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) — 高效架构的复合扩展
- [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244) — 具有 h-swish 和 squeeze-excite 的移动优先架构
- [TensorRT 优化实用指南 (NVIDIA)](https://developer.nvidia.com/blog/acceleating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users/) — 如何实际获取论文中的吞吐量数字
- [ONNX 运行时文档](https://onnxruntime.ai/docs/) — 量化、图形优化、提供商选择
