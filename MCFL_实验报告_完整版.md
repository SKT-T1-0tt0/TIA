# MCFL 多模态协同特征融合实验报告（完整版）

---

## 文档说明

本报告整合了 **MCFL 方法设计**、**视频生成流程**、**多数据集泛化实验**、**Online Baseline Imitation 设计**、**Gate 标定策略** 及 **更新后评估结果**，形成一份统一的技术与实验文档。

**方法层面**（对应代码 `tacm/modules/mcfl.py`、`diffusion/condition_builder.py`）：MCFL 为两阶段注意力融合——Joint Self-Attention 对 text/image/audio 三模态 token 对称交互，再以 Text-Centric Cross-Attention（text 为 query、image+audio 为 key/value）得到融合表示；条件注入采用 **Temporal-only + Gated Residual**（spatial 保持 baseline，temporal 使用 `c_fused = c_text + gate·delta`）。**鲁棒音频管线**：数据侧（`tacm/data.py`）支持 peak/rms 归一化与可选 soft_clip（推荐 `peak`+`none`），训练与采样侧仅在送入 BEATs 前做**单次** response（tanh/compand），保证 response alignment 与 train/sample 一致。**跨分布鲁棒训练**（`diffusion/tacm_train_temp_util.py`）：random gain、random response strength、modality dropout；**自适应条件融合**：norm-based adaptive gating（BEATs 范数→置信度三角形）、可选 z-score 标定、gate 时间 EMA、guardrails（`gate_lambda_max`、`norm_clip_clamp` 按 clip p5–p95 限幅）、以及可选的 **audio-visual similarity 调整 gate**（`condition_builder` 中 audio–image cosine sim 映射为 av_conf，`gate *= (1−β)+β·av_conf`）。

评估数据基于 `evaluation_report_three_groups.txt`（生成时间：2026-03-16，**50 个视频**），采用 Gate 统一标定 [7.2, 10.0]、单次压缩（数据侧 peak + soft_clip=none，BEATs 前单次 response；**推荐 compand**，训练脚本默认 tanh、采样脚本默认 compand）、c_temp EMA（α=0.9）、以及可选的 AV 置信度 Gate（脚本默认关闭）等配置。

---

## 一、方法概述

### 1.1 MCFL 定义

**MCFL（Multi-modal Collaborative Feature Layer）** 是一种用于视频生成任务的多模态协同特征融合层。

与传统的条件拼接（concatenation）不同，MCFL 通过 Self-Attention 与 Cross-Attention 的联合建模，使文本、图像和音频三种模态在同一表示空间中进行显式交互与协同学习。

> **核心思想**：多模态条件不应被视为彼此独立的约束，而应在注意力机制中协同决定最终的生成语义与运动模式。

### 1.2 Baseline 的局限性

原始 TIA2V Baseline 采用简单拼接：

```python
c_ti = concat(c_text, c_image)   # spatial condition
c_at = concat(c_audio, c_text)   # temporal condition
```

存在三个核心问题：

| 问题 | 说明 |
|------|------|
| **音频利用不足** | `c_at` 中 audio 仅 8 token，text 有 77 token，token 数量悬殊，模型更易依赖 text/image，音频对运动的驱动有限 |
| **缺乏模态间交互** | 不同模态仅在 UNet 中被动叠加，无法显式建模语义关系 |
| **时序条件不稳定** | 引入 MCFL 后，audio 与 text 的拼接在 temporal attention 中易放大帧间跳变，引发 flicker |

### 1.3 MCFL 设计目标

- **协同融合**：通过注意力机制建模 text / image / audio 的交互关系
- **语义引导**：以文本为中心，引导视觉与听觉信息对齐
- **向后兼容**：`use_mcfl=False` 时严格退化为 baseline 行为

### 1.4 MCFL 集成策略（当前设计）

#### 条件注入

| 设计 | 说明 |
|------|------|
| **Temporal-only** | `c_ti` 保持 baseline（text + image），仅 `c_at` 使用 MCFL 融合结果 |
| **Spatial 不变** | 避免破坏空间结构，保护 FFC 等指标 |
| **Gated Residual** | `c_fused = c_text + gate·delta`，`delta` 由 MCFL 产生，`gate` 为置信度自适应 |

#### Gate 设计

| 设计 | 说明 |
|------|------|
| **Gate 统一标定** | `norm_low=7.2`，`norm_high=10.0`，覆盖三数据集 BEATs pooled norm p5–p95 |
| **Gate 时间 EMA** | `gate_ema=0.9`，对 per-frame gate 做时间平滑，减轻 flicker |
| **AV 置信度 Gate** | `use_av_conf=True`：`gate *= (1-β) + β·av_conf`，`av_conf` 由 audio-image cosine sim 映射，音画对齐弱时削弱 gate，提升跨域稳定性 |
| **护栏** | `gate_lambda_max=0.2` 硬上限；`norm_clip_clamp=True` 按 clip 内 p5–p95 限幅，避免单帧 spike |

#### 音频与鲁棒性

| 设计 | 说明 |
|------|------|
| **单次压缩** | `audio_norm_mode=peak`，`audio_soft_clip=none`，`audio_response=compand` |
| **c_temp EMA** | 对 BEATs 输出沿时间维 EMA（α=0.9），减轻帧间 jitter |
| **L2 归一化** | 送入 MCFL 前对 image/audio 做 L2 归一化，减弱幅值差异 |
| **梯度半截断** | image / audio 输入 MCFL 时 detach，仅 text 传梯度 |

#### 推荐参数（与采样命令一致）

- `mcfl_gate_lambda=0.1`；AV 置信度 Gate 可选：`mcfl_gate_use_av_conf=True` 时 `mcfl_gate_av_beta=0.5`，`mcfl_gate_av_sim_low=0.0`，`mcfl_gate_av_sim_high=0.3`（训练/采样脚本中**默认均为 False**，可按需开启）
- 音频：`audio_norm_mode=peak`，`audio_soft_clip=none`；**单次 response 推荐 `audio_response=compand`**（`scripts/train_temp.py` 默认为 `tanh`，若需与报告一致请传 `--audio_response compand`；`scripts/sample_motion_optim.py` 默认已为 `compand`）

### 1.5 鲁棒音频管线与自适应条件融合（总览）

本节将 MCFL 中与**鲁棒音频管线**、**跨分布鲁棒训练**、**自适应条件融合**相关的设计做统一归纳，便于查阅与扩展。

#### 1.5.1 鲁棒音频管线

| 设计 | 说明 |
|------|------|
| **Peak / RMS normalization** | 音频输入采用 peak 或 RMS 归一化（当前推荐 `audio_norm_mode=peak`），统一幅值尺度，避免不同音源幅度差异导致 BEATs 输出分布漂移 |
| **单次压缩** | 仅做一次非线性压缩：`audio_soft_clip=none`、`audio_response=compand`，避免双重 tanh 等多次压缩造成强瞬态失真与跨分布崩坏 |
| **Response alignment** | 训练与推理阶段使用相同的 compand/响应曲线与归一化方式，保证 BEATs 输入分布一致，即 response 对齐 |
| **Train / Sample 一致** | 推理管线（归一化方式、compand、c_temp EMA α、gate 参数等）与训练时完全一致，防止分布偏移导致 gate 或条件失配 |

#### 1.5.2 跨分布鲁棒训练

| 设计 | 说明 |
|------|------|
| **Random gain** | 训练时对音频施加随机增益，增强模型对音量变化的鲁棒性，利于跨数据集与跨设备泛化 |
| **Random response strength** | 对响应曲线强度或压缩程度做随机化，模拟不同录制/编码带来的响应差异，提升跨分布稳定性 |
| **Modality dropout** | 训练时以一定概率将 **audio 条件置零**（`c_temp` 置零），避免模型过度依赖音频，提高在缺失或弱音频信号下的稳健性（当前实现仅对 audio，见 `tacm_train_temp_util.py`） |

#### 1.5.3 自适应条件融合

| 设计 | 说明 |
|------|------|
| **Norm-based adaptive gating** | 以 BEATs 输出（或 pooled norm）的统计量为依据，将 gate 映射到 [norm_low, norm_high] 区间，条件强度随特征范数自适应，避免过强/过弱注入 |
| **Z-score / calibration** | 可选：用各数据集 BEATs norm 的均值与方差做 z-score 标定，或通过统一 calibration 将不同数据集的 norm 映射到同一尺度，再驱动 gate |
| **Temporal smoothing** | 对 per-frame gate 做时间维 EMA（如 `gate_ema=0.9`），平滑帧间跳变，减轻因 gate 抖动带来的 flicker 与不稳定 |
| **Guardrails** | 硬性上限与限幅：如 `gate_lambda_max=0.2` 限制 gate 乘数；`norm_clip_clamp=True` 按 clip 内 p5–p95 对 norm 限幅，避免单帧 spike 拉高 gate |
| **Agreement-aware gating** | Gate 不仅依赖 norm，还考虑多模态“一致性”：当各模态语义一致时适当加强融合，不一致时减弱，避免冲突条件主导生成 |
| **用 audio-visual similarity 调整 gate** | 即 AV 置信度 Gate：用 audio–image cosine similarity 得到 `av_conf`，`gate_final = gate_norm × ((1−β) + β·av_conf)`。音画对齐强时保持或略升 gate，对齐弱时降低 gate，用 **audio-visual similarity** 直接调制 gate，提升跨域与跨数据集的稳定性 |

上述各项在本文档中的具体参数与实现位置见 §1.4、§2.2、§7。

---

## 二、视频生成流程（MCFL）

当前 MCFL 推理流程（`sample_motion_optim.py`）如下。

### 2.1 数据流概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输入                                               │
│  文本 (raw_text)  首帧图像 (video[:,0])  音频 (audio, 16 段 clip)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           编码                                                │
│  CLIP Text      → c_t [B, 77, 768]  文本语义                                 │
│  CLIP Image     → image_cat [B, 1, 768]  首帧视觉特征                        │
│  音频: peak 归一 + 单次 compand  → BEATs → c_temp [B*16, 8, 768]             │
│  c_temp 时间 EMA (α=0.9)  减轻帧间 jitter                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           条件构建 (build_conditions, use_mcfl=True)         │
│  c_ti = concat(c_t, image_cat)  [B, 78, 768]  spatial（保持 baseline）       │
│  MCFL(text, image, audio) → c_fused  梯度半截断 + L2 归一化                   │
│  gate = conf_norm × av_conf  （[7.2,10.0] + EMA + AV 置信度）                │
│  c_at = concat(c_temp, c_fused_expanded)  [B*16, 8+8, 768]  temporal         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           扩散采样                                            │
│  init: 首帧保留 + 后续帧置零  → x_0 初始化                                    │
│  p_sample_loop / ddim: x_t → UNet(x_t, t, c_ti, c_at) → x_{t-1}              │
│  UNet: c_ti → spatial cross-attn，c_at → temporal cross-attn                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出                                                │
│  sample [B, 3, 16, H, W] → clamp(-0.5, 0.5) + 0.5 → 保存为 mp4               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键环节说明

| 环节 | 说明 |
|------|------|
| **音频预处理** | 数据侧 `audio_norm_mode=peak`、`audio_soft_clip=none`；BEATs 前单次 `audio_response=compand` |
| **c_temp EMA** | 对 BEATs 输出沿时间维 EMA（α=0.9），减轻帧间 jitter |
| **c_ti** | 保持 baseline：`concat(c_t, image_cat)`，不经过 MCFL |
| **c_at（MCFL）** | MCFL 融合 → c_fused，gate 由 norm 标定 + AV 置信度调制，c_at = concat(c_temp, c_fused_expanded) |
| **UNet** | `context=c_ti`（spatial），`context_temp=c_at`（temporal） |
| **init_image** | 首帧来自真实视频，其余帧置零，引导扩散逐步生成后续帧 |

### 2.3 训练与推理一致性

训练与推理需保持一致：

- gate：`norm_low=7.2`，`norm_high=10.0`，`gate_ema=0.9`；AV 置信度可选，若启用则 `use_av_conf=True`，`av_beta=0.5`，`av_sim_low/high`（训练/采样脚本中默认 `use_av_conf=False`）
- 音频管线：数据侧 `audio_norm_mode=peak`、`audio_soft_clip=none`（由 `tacm/data.py` 与 DataLoader 参数决定）；送入 BEATs 前单次 response 需与训练一致，**推荐 `audio_response=compand`**（训练时需显式传 `--audio_response compand`，因 `train_temp.py` 默认为 `tanh`）
- c_temp EMA：α=0.9

---

## 三、实验配置与评估设置

### 3.1 三组数据集

| 数据集 | Real 目录 | Baseline 目录 | MCFL 目录 |
|--------|-----------|---------------|-----------|
| post_URMP | results/3_tacm_/real | results/3_tacm_/fake1_30fps | results/6_tacm_/fake1_30fps |
| post_landscape | results/7_tacm_/real | results/7_tacm_/fake1_30fps | results/8_tacm_/fake1_30fps |
| post_audioset_drums | results/9_tacm_/real | results/9_tacm_/fake1_30fps | results/10_tacm_/fake1_30fps |

### 3.2 评估指标

| 指标 | 方向 | 说明 |
|------|------|------|
| FVD | ↓ 越低越好 | Fréchet Video Distance，视频分布相似度 |
| FID | ↓ 越低越好 | Fréchet Inception Distance，视觉保真度 |
| FFC | ↓ 越低越好 | First Frame Consistency，首帧一致性 |
| CLIP | ↑ 越高越好 | 文本-视频语义对齐 |
| AV_ALIGN | ↑ 越高越好 | 音视频对齐 |
| TC_FLICKER | ↓ 越低越好 | 时间一致性 / 闪烁（flicker） |

### 3.3 关键配置（更新后）

- **Gate 标定**：`mcfl_gate_norm_low=7.2`，`mcfl_gate_norm_high=10.0`（统一覆盖三数据集 p5–p95）
- **Gate 时间平滑**：`mcfl_gate_ema=0.9`
- **单次压缩**：`audio_norm_mode=peak`，`audio_soft_clip=none`，`audio_response=compand`
- **AV 置信度 Gate（可选）**：`mcfl_gate_use_av_conf=True`，`mcfl_gate_av_beta=0.5`，`mcfl_gate_av_sim_low=0.0`，`mcfl_gate_av_sim_high=0.3`

---

## 四、实验结果（更新后，2026-03-16，50 个视频）

多数据集上 Baseline（无 MCFL）与 MCFL 的指标对比如下。Δ = MCFL − Baseline，**粗体** 表示 MCFL 优于 Baseline。数据来源：`evaluation_report_three_groups.txt`（50 个视频）。

### 4.1 FVD（越低越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 28.60 ± 0.77 | 30.61 ± 1.35 | +2.01 |
| post_landscape | 58.01 ± 3.88 | 60.65 ± 2.15 | +2.64 |
| post_audioset_drums | 97.60 ± 3.34 | 74.48 ± 2.63 | **−23.12** |

### 4.2 FID（越低越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 465.61 ± 8.92 | 562.13 ± 10.61 | +96.51 |
| post_landscape | 604.90 ± 6.21 | 568.16 ± 14.94 | **−36.74** |
| post_audioset_drums | 1784.72 ± 6.71 | 744.92 ± 9.62 | **−1039.80** |

### 4.3 FFC（越低越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 0.1755 ± 0.0003 | 0.1752 ± 0.0001 | **−0.0003** |
| post_landscape | 0.1954 ± 0.0026 | 0.2054 ± 0.0029 | +0.0100 |
| post_audioset_drums | 0.2255 ± 0.0013 | 0.2898 ± 0.0006 | +0.0643 |

### 4.4 CLIP（越高越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 0.2910 ± 0.0152 | 0.2862 ± 0.0162 | −0.0049 |
| post_landscape | 0.1922 ± 0.0101 | 0.1943 ± 0.0101 | **+0.0021** |
| post_audioset_drums | 0.2197 ± 0.0086 | 0.2219 ± 0.0111 | **+0.0022** |

### 4.5 AV_ALIGN（越高越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 0.5063 ± 0.1887 | 0.4937 ± 0.1986 | −0.0126 |
| post_landscape | 0.4824 ± 0.2330 | 0.4242 ± 0.1615 | −0.0582 |
| post_audioset_drums | 0.3639 ± 0.2107 | 0.4021 ± 0.2214 | **+0.0382** |

### 4.6 TC_FLICKER（越低越好；优先参考 median）

| 数据集 | Baseline (flicker mean) | MCFL (flicker mean) | Δ (mean) | Baseline (median) | MCFL (median) | Δ (median) |
|--------|-------------------------|---------------------|----------|-------------------|---------------|------------|
| post_URMP | 52.22 ± 90.94 | 59.63 ± 96.39 | +7.41 | 23.15 | 27.12 | +3.98 |
| post_landscape | 307.18 ± 372.54 | 291.69 ± 502.37 | **−15.49** | 152.63 | 163.90 | +11.27 |
| post_audioset_drums | 163.43 ± 137.49 | 64.07 ± 41.52 | **−99.36** | 131.17 | 52.78 | **−78.39** |

> **说明**：本次评估基于 **50 个视频**，样本量增大后指标更稳定。TC_FLICKER 仍建议**优先参考 median**，以减轻异常值影响。

---

## 五、结果分析

### 5.1 三数据集整体表现（50 个视频）

| 数据集 | 主要结论 |
|--------|----------|
| **post_URMP** | 50 视频评估下 MCFL 在 FFC（−0.0003）上略优；FVD、FID、CLIP、AV_ALIGN 略逊于 Baseline（Δ +2.01、+96.51、−0.0049、−0.0126）；TC_FLICKER median +3.98。URMP 上 real 与 baseline 同源（均为 3_tacm_），MCFL 与 baseline 差异主要来自融合策略，可结合后续 gate/AV 调参进一步优化 |
| **post_landscape** | MCFL 在 FID（−36.74）、CLIP（+0.0021）、TC_FLICKER mean（−15.49）上优于 Baseline；FVD、FFC、AV_ALIGN 略差；TC median +11.27。Landscape 方差较大，mean 与 median 趋势不完全一致，宜综合 FID/CLIP 与 flicker mean 判断 |
| **post_audioset_drums** | MCFL 在 FVD（−23.12）、FID（−1039.80）、CLIP（+0.0022）、AV_ALIGN（+0.0382）、TC_FLICKER（mean −99.36、median −78.39）上**全面优于** Baseline；仅 FFC 上升（+0.0643）。drums 上 **MCFL 跨分布鲁棒性表现最佳**，与 Gate 标定与单次 compand 等改进一致 |

### 5.2 drums 从“崩坏”到“改善”的转变

**早期现象**（Gate 标定与单次压缩改进前）：

- drums 上 MCFL 出现 **FID +912、FVD +23、AV_ALIGN −0.15** 的典型“跨分布崩坏”
- 原因：gate 区间 [5, 30] 与 drums BEATs pooled norm（8.3–8.9）脱节；双重非线性压缩（tanh+tanh）使强瞬态信号失真；条件时间不平滑

**当前表现**（Gate 统一标定 + 单次 compand + c_temp EMA，50 视频评估）：

- drums 上 MCFL **FID 从 1785 降至 745**（Δ −1039.80），**FVD 从 97.6 降至 74.5**（Δ −23.12）
- AV_ALIGN（+0.0382）、CLIP（+0.0022）提升，TC_FLICKER mean/median 大幅改善（−99.36 / −78.39）
- 说明：**Gate [7.2, 10.0]、单次 compand、gate/embedding 时间平滑** 等改动有效解决了 drums 跨分布问题；样本量增至 50 后结论更稳健

### 5.3 TC_FLICKER 解读

- **URMP**：MCFL flicker median +3.98，适度上升；mean 受高方差影响（52.22 vs 59.63），与更强 audio 驱动带来的时序敏感一致
- **landscape**：mean 下降（−15.49），median 上升（+11.27），与 landscape 本身方差大、存在异常样本有关；宜结合 FID/CLIP 综合评估
- **drums**：mean/median 均大幅下降（−99.36 / −78.39），多数视频时间一致性明显更好，验证改进有效

---

## 六、Online Baseline Imitation

### 6.1 设计动机

MCFL 强化 audio 驱动后，temporal attention 对条件变化的敏感性增加，容易放大帧间跳变，引发 flicker。**Online Baseline Imitation** 直接约束 MCFL 分支的 temporal attention 在时间上的**变化模式**（Δ-attn）接近 baseline，从而在保留多模态协同的同时抑制 flicker。

### 6.2 核心思路

- **位置**：UNet → TemporalTransformer → CrossAttention（`context_temp = c_at`），不是 MCFL 内部
- **目标**：让 MCFL 的 temporal attention 的帧间差分 Δ-attn 接近 baseline
- **约束**：`L_attn = ||Δattn_mcfl − Δattn_base||²`

**Δ 计算**（沿 query/time 维）：

```python
# attn: [B*H, F, M]，F=query/time 帧数
Δ = attn[:, 1:, :] - attn[:, :-1, :]   # 正确：沿 dim 1
# ❌ 错误：attn[:, :, 1:] - attn[:, :, :-1] 是 context 维
```

### 6.3 实现要点

| 要点 | 说明 |
|------|------|
| **同一 noise** | MCFL 与 baseline 共用同一 `noise`，保证 `x_t` 一致，可比较同一步的 attention |
| **baseline 无梯度** | baseline 前向使用 `with th.no_grad()` |
| **MCFL attn 不 detach** | MCFL 分支的 attention 保留计算图，用于 L_attn 回传 |
| **attn_cache** | CrossAttention 通过 `attn_cache` 收集最后一个 temporal block 的 attn，避免扫描 modules |
| **双前向** | 先 MCFL 前向得到 attn_mcfl；再构建 c_at_baseline，baseline 前向得到 attn_base |

### 6.4 Loss 与 λ 调度

```
loss = loss_diffusion + λ_attn * L_attn
L_attn = ((Δattn_mcfl - Δattn_base.detach()) ** 2).mean()
```

**λ_attn 调度**（逐步衰减，与 `tacm_train_temp_util.py` 一致）：

| 步数 s | λ_attn |
|--------|--------|
| s < 3000 | 0.1 |
| 3000 ≤ s < 6000 | 0.03 |
| s ≥ 6000 | 0.005 |

> 代码中另有权重较小的 `loss_attn_energy`（β=0.01）约束 Δ 幅度。λ_attn 不宜 >0.2，不宜常数跑全程，避免限制 MCFL 自由学习。

### 6.5 开关与模式

| 模式 | mcfl_conservative | use_baseline_imitation |
|------|-------------------|------------------------|
| 保守版 MCFL | True | False |
| Online Baseline Imitation | **False** | **True** |

启用 Online Baseline Imitation 时需设置 `mcfl_conservative=False`，使 MCFL 不被冻结，在 imitation 约束下自由学习。

### 6.6 数据流示意

```
                    use_baseline_imitation?
                                │
                ┌───────────────┴───────────────┐
                │ False                         │ True
                ▼                               ▼
         compute_losses()                compute_losses(noise=固定)
         loss = loss_diffusion           attn_mcfl = _get_last_temporal_attn()
                                                │
                                                ▼
                                         构建 c_at_baseline (无 MCFL)
                                                │
                                                ▼
                                         with no_grad: baseline 前向
                                         attn_base = _get_last_temporal_attn()
                                                │
                                                ▼
                                         L_attn = ||Δattn_mcfl - Δattn_base||²
                                         loss = loss_diffusion + λ_attn * L_attn
```

### 6.7 涉及代码

- `diffusion/attention.py`：CrossAttention 增加 `return_attn`、`attn_cache`
- `diffusion/attention_dual.py`：DualTemporalTransformer 传递 `return_attn`
- `diffusion/tacm_unet_temp_dual.py`：UNet、TimestepEmbedSequential 传递 `return_attn`
- `diffusion/tacm_train_temp_util.py`：双前向、L_attn 计算、λ_attn 调度
- `scripts/train_temp.py`：`--use_baseline_imitation` 参数

---

## 七、Gate 标定与改动汇总

### 7.1 问题与改进对照

| 问题 | 改进 |
|------|------|
| Gate 区间 [5, 30] 与 drums norm 脱节 | 统一标定 [7.2, 10.0]，覆盖三数据集 p5–p95 |
| 双重非线性压缩（tanh+tanh） | 单次压缩：`audio_soft_clip=none` + `audio_response=compand` |
| 条件时间跳变 | c_temp EMA（α=0.9）+ gate 时间 EMA（0.9） |
| 跨集幅度差异 | Random Gain、Modality Dropout、L2 归一化（`mcfl_norm_modality=True`） |

### 7.2 三数据集 BEATs pooled norm 统计

| 数据集 | mean | std | p5 | p95 |
|--------|------|-----|-----|-----|
| post_audioset_drums | 8.57 | 0.16 | 8.30 | 8.82 |
| post_URMP | 8.08 | 0.69 | 7.44 | 9.79 |
| post_landscape | 8.58 | 0.23 | 8.19 | 8.90 |

统一 gate 区间 [7.2, 10.0] 可同时覆盖三数据集，实现“该开时开、异常时关”。

### 7.3 AV 置信度 Gate（新增）

当 `mcfl_gate_use_av_conf=True` 时，在 norm-based gate 基础上再乘 audio-image cosine similarity 映射：

```
av_conf = clamp((sim - av_sim_low) / (av_sim_high - av_sim_low), 0, 1)
gate_final = gate_norm * ((1 - beta) + beta * av_conf)
```

| 参数 | 推荐默认 | 说明 |
|------|----------|------|
| mcfl_gate_use_av_conf | True | 启用 AV 置信度调制 |
| mcfl_gate_av_beta | 0.5 | 混合系数，0=不影响，1=纯乘 av_conf |
| mcfl_gate_av_sim_low | 0.0 | cosine sim 映射下界 |
| mcfl_gate_av_sim_high | 0.3 | cosine sim 映射上界 |

作用：音画对齐弱时适当削弱 gate，减少条件噪声，提升跨域稳定性。

### 7.4 推荐训练/推理配置

| 类别 | 参数 | 推荐值 | 说明 |
|------|------|--------|------|
| 音频 | audio_norm_mode | peak | 数据侧，DataLoader 通过 args 传入 |
| 音频 | audio_soft_clip | none | 数据侧，与单次 response 配合避免双重压缩 |
| 音频 | audio_response | compand | 训练脚本默认 **tanh**，推荐传 `--audio_response compand`；采样脚本默认 compand |
| Gate | mcfl_gate_norm_low | 7.2 | 与 condition_builder 一致 |
| Gate | mcfl_gate_norm_high | 10.0 | 同上 |
| Gate | mcfl_gate_ema | 0.9 | 同上 |
| Gate | mcfl_gate_lambda | 0.1 | 同上 |
| Gate（可选） | mcfl_gate_use_av_conf | True（推荐） | 训练/采样脚本中**默认 False**，需显式开启 |
| Gate（可选） | mcfl_gate_av_beta | 0.5 | 启用 AV 时使用 |

---

## 八、代码改动与工具

### 8.1 涉及模块

| 模块 | 改动 |
|------|------|
| `tacm/data.py` | TAVDataset：audio_norm_mode、audio_soft_clip、audio_rms_target（归一化在数据加载时完成；response 在训练/采样循环中做） |
| `diffusion/tacm_train_temp_util.py` | Random Gain、单次 response（tanh/compand）、Modality Dropout（仅 audio）、c_temp EMA、build_conditions 及全部 gate/AV 参数传递 |
| `diffusion/condition_builder.py` | Gate 标定（norm/z-score）、gate 时间 EMA、AV 置信度 Gate、L2 归一化（mcfl_norm_modality）、delta 归一化与 scale |
| `scripts/sample_motion_optim.py` | 与训练一致的 gate、音频管线（单次 response 默认 compand）、AV Gate 参数（默认 False） |
| `scripts/train_temp.py` | 新增 gate、AV、音频相关参数；**默认 audio_response='tanh'**，推荐传 `--audio_response compand`；mcfl_gate_use_av_conf 默认 False |

### 8.2 统计与标定工具

- **`scripts/audio_beats_stats.py`**：统计原始音频与 BEATs pooled/per-frame norm，给出 gate 建议
- **`scripts/unify_gate_calibration.py`**：合并多数据集 stats，输出统一 low/high 与 z-score 参数
- **`scripts/MCFL_GATE_CALIBRATION.md`**：Gate 标定策略、验证矩阵、防炸护栏说明

---

## 九、小结

### 9.1 实验结论

1. **三数据集（50 视频评估，2026-03-16）**：在 Gate 统一标定、单次压缩、时间平滑等改进后，**post_audioset_drums** 上 MCFL 全面优于 Baseline（FVD/FID/AV_ALIGN/TC_FLICKER 等）；**post_landscape** 上 FID、CLIP、TC mean 改善；**post_URMP** 上 FFC 略优，其余指标与 Baseline 接近或略逊，可结合 real=3_tacm_ 配置与 gate/AV 调参进一步优化。
2. **drums**：从早期“跨分布崩坏”转为 FID/FVD/AV_ALIGN/TC_FLICKER 多指标显著改善（50 视频下 FID Δ −1039.80、TC median −78.39），证明 gate 标定与单次 compand 等改动有效。
3. **TC_FLICKER**：URMP 上 median 适度上升（+3.98）；landscape 上 mean 改善、median 略升；drums 上 mean/median 均大幅改善。评估时优先参考 median 以减轻异常值影响。

### 9.2 推荐实践

- **鲁棒音频管线**：训练与推理使用同一套 gate、音频、AV 参数，保证 train/sample 一致；单次压缩方案：`peak + soft_clip=none + response=compand`，配合 response alignment
- **跨分布鲁棒**：训练时可启用 random gain、random response strength、modality dropout（见 §1.5.2）
- **自适应条件融合**：norm-based gating + temporal smoothing + guardrails；可选启用 **audio-visual similarity 调整 gate**（AV 置信度 Gate，`mcfl_gate_use_av_conf=True`）提升跨域稳定性
- 新数据集上线前，用 `audio_beats_stats.py` 检查 pooled norm 分布，必要时调整 gate 或启用 z-score 标定（§1.5.3）

### 9.3 文档与参考

- **方法设计**：`MCFL_报告.md`
- **Gate 标定**：`scripts/MCFL_GATE_CALIBRATION.md`
- **设计修改**：`MCFL_DESIGN_MODIFICATIONS_REPORT.md`
- **评估报告**：`evaluation_report_three_groups.txt`

---

## 十、未来工作：Learned Gate 微调

在保留现有 **norm gate + agreement gate** 的前提下，可在其基础上叠加一层 **learned refinement**，形成「手工 gate 保底 + 学习 gate 微调」的最小可落地方案。

**核心形式**：\( g = g_{\text{hand}} \cdot \sigma(\text{MLP}(x)) \)。其中 \( g_{\text{hand}} \) 为当前已有的 \( \lambda_{\max} \cdot c_{\text{norm}} \cdot ((1-\beta)+\beta c_{\text{av}}) \)；\( \sigma(\text{MLP}(x)) \in (0,1) \) 为可学习因子，仅做微调，不改动原有设计，失败风险低。

**建议先做 clip-level**（每 clip 一个因子）：输入特征取 4 维——pooled audio 范数、pooled image 范数、audio–image cosine similarity、per-frame audio norm 的时间标准差；MLP 输出经 sigmoid 得到 learned factor，与 \( g_{\text{hand}} \) 相乘。特征建议 detach，并加轻量 gate 正则（如对 learned factor 的均值做 L1/L2）避免学成全开。

**验证顺序**：先小规模短训，观察 learned factor 的均值/方差是否在合理范围（如 0.3–0.8）；再在 **landscape** 上对比 `use_learned=False` 与 `True`，看 FID/FVD、AV_ALIGN、TC_FLICKER median 是否更稳；最后在 **drums / URMP** 上确认不损失现有收益。叙事上可概括为：*robust hand-crafted prior + learned adaptive correction*。

---

*报告整合时间：2026-03-16（实验结果更新为 50 视频数据）*
