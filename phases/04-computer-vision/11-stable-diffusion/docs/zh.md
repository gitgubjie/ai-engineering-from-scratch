# Stable Diffusion — Architecture & Fine-Tuning

> Stable Diffusion is a DDPM that runs in the latent space of a pretrained VAE, conditioned on text via cross-attention, sampled with a fast deterministic ODE solver, and steered by classifier-free guidance.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 minutes

## Learning Objectives

- Trace the five pieces of a Stable Diffusion pipeline: VAE, text encoder, U-Net, scheduler, safety checker — and what each of them actually does
- Explain latent diffusion and why training in a 4x64x64 latent space (instead of a 3x512x512 image) reduces compute by 48x without quality loss
- Use `diffusers` to generate images, run image-to-image, inpainting, and ControlNet-guided generation
- Fine-tune Stable Diffusion with LoRA on a small custom dataset and load the LoRA adapter at inference

## The Problem

Training a DDPM directly on 512x512 RGB images is expensive. Every training step backprops through a U-Net that sees 3x512x512 = 786,432 input values, and sampling takes 50+ forward passes through that same U-Net. At the quality level of Stable Diffusion 1.5 (released 2022), pixel-space diffusion would need roughly 256 GPU-months of training and 10-30 seconds per image on a consumer GPU.

The trick that made open-weight text-to-image practical was **latent diffusion** (Rombach et al., CVPR 2022). Train a VAE that maps a 3x512x512 image to a 4x64x64 latent tensor and back, then do the diffusion in that latent space. Compute drops by `(3*512*512)/(4*64*64) = 48x`. Sampling drops from tens of seconds to under two seconds on the same GPU.

Almost every modern image-generation model — SDXL, SD3, FLUX, HunyuanDiT, Wan-Video — is a latent diffusion model with variations on the autoencoder, the denoiser (U-Net or DiT), and the text conditioning. Learn Stable Diffusion and you have learnt the template.

## The Concept

### The pipeline

```mermaid
flowchart LR
    TXT["Text prompt"] --> TE["Text encoder<br/>(CLIP-L or T5)"]
    TE --> CT["Text<br/>embedding"]

    NOISE["Noise<br/>4x64x64"] --> UNET["UNet<br/>(denoiser with<br/>cross-attention<br/>to text)"]
    CT --> UNET

    UNET --> SCHED["Scheduler<br/>(DPM-Solver++,<br/>Euler)"]
    SCHED --> LATENT["Clean latent<br/>4x64x64"]
    LATENT --> VAE["VAE decoder"]
    VAE --> IMG["512x512<br/>RGB image"]

    style TE fill:#dbeafe,stroke:#2563eb
    style UNET fill:#fef3c7,stroke:#d97706
    style SCHED fill:#fecaca,stroke:#dc2626
    style IMG fill:#dcfce7,stroke:#16a34a
```

- **VAE** — frozen autoencoder. Encoder turns image into latents (used for img2img and training). Decoder turns latents back into an image.
- **Text encoder** — CLIP text encoder (SD 1.x/2.x), CLIP-L + CLIP-G (SDXL), or T5-XXL (SD3/FLUX). Produces a sequence of token embeddings.
- **U-Net** — the denoiser. Has cross-attention layers that attend from latents to the text embedding at every resolution level.
- **Scheduler** — the sampling algorithm (DDIM, Euler, DPM-Solver++). Picks sigmas, blends predicted noise back into the latent.
- **Safety checker** — optional NSFW / illegal-content filter on the output image.

### Classifier-free guidance (CFG)

Plain text conditioning learns `epsilon_theta(x_t, t, c)` for every prompt `c`. CFG trains the same network with `c` dropped 10% of the time (replaced by an empty embedding), giving a single model that predicts both the conditional and the unconditional noise. At inference:

```
eps = eps_uncond + w * (eps_cond - eps_uncond)
```

`w` is the guidance scale. `w=0` is unconditional, `w=1` is plain conditional, `w>1` pushes the output toward being "more conditioned on the prompt" at the cost of diversity. SD default is `w=7.5`.

CFG is the reason text-to-image works at production quality. Without it, prompts bias the output weakly; with it, prompts dominate.

### Latent space geometry

The VAE's 4-channel latent is not just a compressed image. It is a manifold where arithmetic roughly corresponds to semantic edits (prompt engineering + interpolation both live here), and where the diffusion U-Net has been trained to spend its entire modelling budget. Decoding a random 4x64x64 latent does not produce a random-looking image — it produces garbage, because only a specific submanifold of latents decodes to valid images.

Two consequences:

1. **Img2img** = encode image to latent, add partial noise, run the denoiser, decode. Image structure survives because encoding is near-invertible; content changes based on the prompt.
2. **Inpainting** = same as img2img but the denoiser only updates masked regions; unmasked regions are kept at the encoded latent.

### The U-Net architecture

The SD U-Net is a big version of the TinyUNet from Lesson 10 with three additions:

- **Transformer blocks** at every spatial resolution, containing self-attention + cross-attention to the text embedding.
- **Time embedding** via MLP on sinusoidal encoding.
- **Skip connections** between encoder and decoder at matching resolutions.

Total parameters in SD 1.5: ~860M. SDXL: ~2.6B. FLUX: ~12B. The jump in params is mostly in attention layers.

### LoRA fine-tuning

Full fine-tuning of Stable Diffusion needs 20+ GB of VRAM and updates 860M parameters. LoRA (Low-Rank Adaptation) keeps the base model frozen and injects small rank-decomposition matrices into the attention layers. A LoRA adapter for SD is typically 10-50 MB, trains in 10-60 minutes on a single consumer GPU, and loads at inference time as a drop-in modification.

```
Original: W_q : (d_in, d_out)   frozen
LoRA:     W_q + alpha * (A @ B)   where A : (d_in, r), B : (r, d_out)

r is typically 4-32.
```

LoRA is how almost every community fine-tune is distributed. CivitAI and Hugging Face host millions of them.

### Schedulers you will see

- **DDIM** — deterministic, ~50 steps, simple.
- **Euler ancestral** — stochastic, 30-50 steps, slightly more creative samples.
- **DPM-Solver++ 2M Karras** — deterministic, 20-30 steps, production default.
- **LCM / TCD / Turbo** — consistency models and distilled variants; 1-4 steps at the cost of some quality.

Swapping schedulers is a one-line change in `diffusers` and sometimes fixes sample issues without any retraining.

## Build It

This lesson uses `diffusers` end-to-end rather than rebuilding Stable Diffusion from scratch. The pieces you would need to rebuild (VAE, text encoder, U-Net, scheduler) are topics of their own lessons; here the goal is fluency with the production API.

### Step 1: Text-to-image

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="a dog riding a skateboard in tokyo, studio ghibli style",
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("dog.png")
```

`float16` halves VRAM with no visible quality loss. `num_inference_steps=25` with the default DPM-Solver++ matches `num_inference_steps=50` with DDIM.

### Step 2: Swap the scheduler

```python
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
```

Scheduler state is decoupled from U-Net weights. You can train on DDPM and sample with any scheduler.

### Step 3: Image-to-image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init_image = Image.open("dog.png").convert("RGB").resize((512, 512))
out = img2img(
    prompt="a dog riding a skateboard, oil painting",
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
).images[0]
```

`strength` is how much noise to add before denoising (0.0 = unchanged, 1.0 = full regeneration). 0.5-0.7 is the standard range for style transfer.

### Step 4: Inpainting

```python
from diffusers import StableDiffusionInpaintPipeline

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("dog.png").convert("RGB").resize((512, 512))
mask = Image.open("dog_mask.png").convert("L").resize((512, 512))

out = inpaint(
    prompt="a cat",
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
).images[0]
```

White pixels in the mask are the area to regenerate. Black pixels are preserved.

### Step 5: LoRA loading

```python
pipe.load_lora_weights("sayakpaul/sd-lora-ghibli")
pipe.fuse_lora(lora_scale=0.8)

image = pipe(prompt="a village square in ghibli style").images[0]
```

`lora_scale` controls strength; 0.0 = no effect, 1.0 = full effect. `fuse_lora` bakes the adapter into the weights in place for speed, but prevents swapping. Call `pipe.unfuse_lora()` before loading a different adapter.

### Step 6: LoRA training (sketch)

Real LoRA training lives in `peft` or `diffusers.training`. The outline:

```python
# Pseudocode
for step, batch in enumerate(dataloader):
    images, prompts = batch
    latents = vae.encode(images).latent_dist.sample() * 0.18215

    t = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    text_emb = text_encoder(tokenizer(prompts))

    pred_noise = unet(noisy_latents, t, text_emb)  # LoRA weights injected here

    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()
```

只有LoRA矩阵接收梯度；基础 U-Net、VAE 和文本编码器被冻结。批量大小为 1 且梯度检查点适合 8 GB 的 VRAM。

## 使用它

在生产中，您实际做出的决定：

- **模型系列**：SD 1.5 用于开源社区微调，SDXL 用于更高的保真度，SD3 / FLUX 用于最先进的技术和严格的许可要求。
- **调度器**：DPM-Solver++ 2M Karras 用于 20-30 个步骤，LCM-LoRA 当延迟低于 1 秒时。
- **精度**：4080/4090 上为“float16”，A100 及更新版本上为“bfloat16”，当 VRAM 紧张时为“int8”（通过“bitsandbytes”或“compel”）。
- **调节**：纯文本作品；为了获得更强的控制，请在基础管道的顶部添加 ControlNet（canny、深度、姿势）。

对于批量生成，“AUTO1111”/“ComfyUI”是社区工具；对于生产 API，使用 TensorRT 编译的“diffusers”+“accelerate”或“optimum-nvidia”。

## 发货

本课产生：

- `outputs/prompt-sd-pipeline-planner.md` — 在给定延迟预算、保真度目标和许可约束的情况下选择 SD 1.5 / SDXL / SD3 / FLUX 加上调度程序和精度的提示。
- `outputs/skill-lora-training-setup.md` — 一项为自定义数据集编写完整 LoRA 训练配置的技能，包括标题、排名、批量大小和学习率。

## 练习

1. **（简单）** 在 `[1, 3, 5, 7.5, 10, 15]` 中生成与 `guidance_scale` 相同的提示。描述图像如何变化。文物出现的指导价值是多少？
2. **（中）** 拍摄任何真实照片，通过“StableDiffusionImg2ImgPipeline”以“[0.2, 0.4, 0.6, 0.8, 1.0]”中的“strength”运行它。哪种力量可以在改变风格的同时保留构图？为什么1.0完全忽略输入？
3. **（难）** 在单个主题（宠物、徽标、角色）的 10-20 张图像上训练 LoRA，并生成包含该主题的新颖场景。报告 LoRA 排名和训练步骤，这些步骤可以在不过度拟合输入图像的情况下实现最佳身份保留。

## 关键术语

|术语 |人们怎么说 |它实际上意味着什么 |
|------|----------------|----------------------|
|潜在扩散| “潜伏中扩散” |在 VAE 潜在空间 (4x64x64) 而不是像素空间 (3x512x512) 中运行整个 DDPM； 48 倍计算节省 |
| VAE 比例因子 | “0.18215”|将 VAE 的原始潜在变量重新调整为大致单位方差的常量；在每个 SD 管道中进行硬编码 |
|无分类器指导 | “CFG”|混合有条件和无条件噪声预测；最具影响力的推理旋钮|
|调度程序| “采样器” |将噪声+模型预测转化为去噪潜在轨迹的算法 |
|洛拉 | “低级适配器”|小型排名分解矩阵，可在不触及基本权重的情况下微调注意力层 |
|交叉注意力 | “文本-图像注意力” |从潜在标记到文本标记的注意力；在每个 U-Net 级别注入提示信息 |
|控制网| “结构调节” |一个单独训练的适配器，通过额外的输入（精明、深度、姿势、分割）来控制 SD |
| DPM-求解器++ | “默认调度程序” |二阶确定性 ODE 求解器； 2026 年以低步数 (20-30) 获得最佳质量 |

## 进一步阅读

- [具有潜在扩散的高分辨率图像合成（Rombach 等人，2022）](https://arxiv.org/abs/2112.10752) — 稳定扩散论文；包括证明设计合理性的每一个消融
- [无分类器扩散指导 (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) — CFG 论文
- [LoRA：大型语言模型的低阶适应（Hu et al., 2021）](https://arxiv.org/abs/2106.09685) — LoRA 是 NLP 优先；它几乎没有任何变化地转移到 SD
- [diffusers 文档](https://huggingface.co/docs/diffusers) — 每个 SD / SDXL / SD3 / FLUX 管道的参考
