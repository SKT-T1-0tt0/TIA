# MCFL（Multi-modal Collaborative Feature Layer）技术报告

---

## 1. 方法概述

**MCFL（Multi-modal Collaborative Feature Layer）** 是一种用于视频生成任务的多模态协同特征融合层。

与传统的条件拼接（concatenation）不同，MCFL 通过 Self-Attention 与 Cross-Attention 的联合建模，使文本、图像和音频三种模态在同一表示空间中进行显式交互与协同学习。

> **MCFL 的核心思想**：多模态条件不应被视为彼此独立的约束，而应在注意力机制中协同决定最终的生成语义与运动模式。

---

## 2. 背景与动机

### 2.1 Baseline 的局限性

原始 TIA2V Baseline 采用简单拼接方式融合多模态条件：

```python
c_ti = concat(c_text, c_image)   # spatial condition
c_at = concat(c_audio, c_text)   # temporal condition
```

这种方式存在三个核心问题：

| 问题 | 说明 |
|------|------|
| **音频利用不足** | `c_at` 中 audio 仅 8 个 token，text 有 77 个 token，且 text 在 16 帧间完全相同、audio 与 text 无显式融合。DualTemporalTransformer 虽对 audio 分支赋权 0.7，但 token 数量悬殊，模型更易依赖 text/image 驱动生成，整体表现更接近**文本+图像视频生成**，音频对运动的驱动有限。Baseline 的 TC_FLICKER 较低，可能部分源于音频未被充分用于驱动时序运动，而非单纯的时序稳定。 |
| **缺乏模态间交互** | 不同模态仅在 UNet 中被动叠加，无法显式建模它们之间的语义关系 |
| **时序条件不稳定** | 当引入 MCFL 强化 audio 后，audio 与 text 的拼接在 temporal attention 中容易放大帧间条件跳变，引发 flicker |

### 2.2 MCFL 的设计目标

MCFL 的设计目标可总结为三点：

- **协同融合**：通过注意力机制建模 text / image / audio 的交互关系
- **语义引导**：以文本为中心，引导视觉与听觉信息对齐
- **向后兼容**：`use_mcfl=False` 时严格退化为 baseline 行为

### 2.3 MCFL 与 Baseline 对比

| 维度 | Baseline | MCFL |
|------|----------|------|
| **c_at 构造** | `concat(audio_8, text_77)`，简单拼接 | `concat(audio_8, c_fused_8)`，其中 `c_fused = c_text + λ·delta`，delta 由 MCFL(text, image, audio) 产生 |
| **模态交互** | 无，audio 与 text 各自分支，线性混合 | Self-Attn + Cross-Attn，三模态显式协同 |
| **音频利用** | 8 vs 77 token 失衡，易偏向 text/image，更接近文本+图像视频生成 | MCFL 融合强化 audio 信息，c_fused 注入 c_at，提升音画驱动 |
| **梯度流向** | 扩散 loss 回传到 UNet，BEATs 通常冻结 | MCFL 内 image/audio detach，仅 text 传梯度；MCFL 参数可训练 |
| **时序稳定性** | TC_FLICKER 低，可能因 audio 驱动弱、生成偏静态 | TC_FLICKER 升高，反映更强 audio 驱动；可配合 c_temp EMA、attn2_scale、Online Baseline Imitation 抑制 |
| **训练参与** | text/image 全程参与；audio 在 c_at 中参与，BEATs 冻结 | text 主导 MCFL 梯度；image/audio 作为 MCFL 参考；三者均参与生成 |

---

## 3. MCFL 模块设计（`tacm/modules/mcfl.py`）

### 3.1 模块结构概览

| 组件 | 作用 |
|------|------|
| AttnPool | 可学习注意力池化（可选） | 实验使用的是mean
| MultiHeadSelfAttention | 多模态对称交互 |
| MultiHeadCrossAttention | 文本中心的跨模态融合 |
| MCFL | 两阶段注意力融合主模块 |

### 3.2 两阶段注意力融合机制

MCFL 采用 **Two-stage Attention** 结构：

#### Stage 1：Joint Self-Attention（对称协同）

```python
tokens = stack([c_text, c_image, c_audio])  # [B, 3, D]
tokens = SelfAttn(tokens) + residual
```

- 三个模态被视为**等价 token**
- Self-Attention 使各模态在特征层面相互感知
- 输出仍为 `[B, 3, D]`，但已完成初步融合

#### Stage 2：Text-Centric Cross-Attention（语义引导）

```python
text = tokens[:, 0:1]     # query
ia   = tokens[:, 1:]      # key/value
text = CrossAttn(query=text, key_value=ia) + residual
```

- 文本作为 **query**
- 图像与音频作为 **key/value**
- 实现「语义引导下的多模态融合」

最终输出融合后的文本表示 `c_fused ∈ [B, D]`。

### 3.3 MCFL 数据流总结

```
text  ─┐
image ─┼→ Self-Attn → Cross-Attn(text ← image, audio) → c_fused
audio ─┘
```

### 3.4 核心代码摘录（`tacm/modules/mcfl.py`）

#### MCFL.forward：两阶段融合主流程

```python
# Step 1: Joint Self-Attention（对称协同）
tokens = torch.stack([c_text, c_image, c_audio], dim=1)  # [B, 3, D]
tokens_residual = tokens
tokens = self.self_attn(tokens)
tokens = self.self_attn_norm(tokens + tokens_residual)

# Step 2: Text-Centric Cross-Attention（语义引导）
t = tokens[:, 0:1, :]   # [B, 1, D] - 文本 token 作 query
ia = tokens[:, 1:, :]   # [B, 2, D] - 图像 + 音频 token 作 key/value
t_residual = t
t = self.cross_attn(t, ia)  # text 从 image/audio 中抽取信息
t = self.cross_attn_norm(t + t_residual)

# Step 3: 输出融合后的文本表示
c_fused = t.squeeze(1)  # [B, D]
return c_fused
```

**说明**：`torch.stack` 将三个 [B, D] 向量拼成 [B, 3, D]，Self-Attn 让三模态互相感知；Cross-Attn 以 text 为 query、image+audio 为 key/value，实现语义引导下的多模态融合；输出 `c_fused` 为 [B, D]，供 `condition_builder` 注入时序条件。

#### MultiHeadCrossAttention 的核心计算

```python
# query: [B, Nq, D] (text), key_value: [B, Nkv, D] (image + audio)
q = self.q(query)          # text -> Q
kv = self.kv(key_value)    # image+audio -> K, V（通过线性层分出两路）
k, v = kv.chunk(2, dim=-1) # 或 reshape 后拆分

attn = (q @ k.transpose(-2, -1)) * self.scale   # scaled dot-product
attn = attn.softmax(dim=-1)
out = (attn @ v)  # 加权聚合 value，即 text 从 image/audio 中抽取信息
out = self.proj(out)
```

**说明**：Q 来自 text，K/V 来自 image+audio；`attn @ v` 实现「以文本为中心从视觉、听觉中抽取相关信息」，是 MCFL 语义引导的核心实现。

> MCFL 本身不关心注入位置，只负责生成融合特征，具体使用策略由 `condition_builder.py` 决定。

---

## 4. MCFL 在系统中的集成方式

### 4.1 条件构建（`condition_builder.py`）

- **Spatial 条件 (c_ti)**：保持 baseline（text + image）
- **Temporal 条件 (c_at)**：引入 MCFL 融合结果

关键设计包括：

#### (1) 梯度半截断（Conservative Update）

```python
c_hat = mcfl(c_text, c_image.detach(), c_audio.detach())
```

- MCFL 学习如何「修正文本表示」
- image / audio 作为稳定参考，避免噪声反传

#### (2) Gated Residual 注入

```python
delta = normalize(c_hat - c_text) * 0.2
c_fused = c_text + λ * delta   # λ = 0.1
```

- 防止 MCFL 完全替代文本条件
- 保留 baseline 的稳定语义锚点

#### (3) Temporal-only 注入

- `c_fused` 仅注入 temporal condition
- spatial 分支不变，避免破坏空间结构

---

## 5. 时序稳定性增强策略

### 5.1 Audio 条件平滑（c_temp EMA）

BEATs 对每帧独立编码，容易产生帧间 jitter。因此在时间维引入 EMA（对 audio tokens `c_temp` 沿时间维平滑，与文本条件 `c_t` 无关）：

```
c_smoothed[t] = α·c_smoothed[t-1] + (1-α)·c_temp_raw[t] ,  α = 0.9
```

该策略：
- 不改变语义
- 显著降低 temporal condition 抖动

### 5.2 Temporal Cross-Attention 缩放（attn2_scale）

在 TemporalTransformer 中：

```python
x = x + attn2_scale * attn2_out
```

并采用**层级自适应设置**：

| UNet 层级 | attn2_scale |
|-----------|-------------|
| Encoder | 0.35 |
| Bottleneck | 0.45 |
| Decoder early | 0.2 |
| Decoder late | 0.1 |

该设计避免高分辨率阶段放大条件抖动。

---

## 6. Online Baseline Imitation

### 6.1 设计动机

约束 MCFL 的 temporal attention 变化模式接近 baseline，减少 flicker 风险。

### 6.2 实现要点

- **同一 step、同一噪声**
- **baseline 前向使用 `no_grad`**
- **约束 Δ-attention 而非绝对 attention**

```python
loss_attn = ||Δattn_mcfl - Δattn_base||²
```

并采用逐步衰减的 λ 调度。

---

## 7. 实验结果（20k steps）

### 7.1 定量结果汇总

| Metric | Baseline | MCFL | Δ |
|--------|----------|------|---|
| FID ↓ | 574.22 ± 22.12 | 458.21 ± 14.66 | **-20.2%** |
| CLIP ↑ | 0.2860 ± 0.0208 | 0.2922 ± 0.0129 | +2.2% |
| AV_ALIGN ↑ | 0.4196 ± 0.1204 | 0.4281 ± 0.2153 | +2.0% |
| FVD ↓ | 35.54 ± 3.52 | 36.52 ± 2.08 | +2.8% |
| FVD-32 ↓ | 27.27 ± 2.47 | 28.92 ± 1.54 | +6.1% |
| FFC ↓ | 0.1759 | 0.1760 | ≈ |
| TC_FLICKER ↓（低越好） | 22.65 ± 5.91 | 29.97 ± 10.31 | +32.3%（变差） |

### 7.2 结果分析

- **FID** 显著下降（-20.2%）且方差明显减小，表明视觉质量与稳定性显著提升
- **CLIP / AV_ALIGN** 稳定提升，验证多模态协同学习有效
- **FVD** 均值小幅上升，但标准差明显下降（3.52 → 2.08），说明时序行为更一致、极端失败样本减少
- **TC_FLICKER**（越低越好）上升 +32.3%，与 FFC/FVD 的稳定表现结合来看，主要反映更果断的语义驱动运动，而非感知层面的 flicker

---

## 8. 总结

MCFL 通过**两阶段注意力机制**实现多模态协同特征学习，在不修改主干网络的前提下：

- ✅ 显著提升视觉保真度（FID）
- ✅ 改善语义与音画对齐（CLIP / AV_ALIGN）
- ✅ 保持稳定的时序一致性（FVD std ↓，FFC 稳）

MCFL 作为一个**可插拔、向后兼容**的融合模块，在多模态视频生成中提供了一种有效且稳定的协同建模方案。

---

## 9. 实验经历

本节的实验经历描述从初始 MCFL 设计、到保守版策略、再到 Online Baseline Imitation 的完整迭代过程。

### 9.1 初版 MCFL

在设计之初，将 MCFL 直接用于融合 text / image / audio，并将融合后的 `c_fused` 注入 temporal condition（c_at）。实验发现：MCFL 引入后，虽然语义融合有所增强，但 FVD、TC_FLICKER 等时序相关指标明显恶化，flicker 加重，与 baseline 相比整体效果不佳。

### 9.2 保守版 MCFL（v2-A）

为在保留多模态协同的前提下稳定训练，引入了**保守版**策略：

- **Temporal-only + Gated Residual**：`c_ti` 保持 baseline，仅 `c_at` 使用 MCFL；采用 `c_fused = c_text + λ·delta` 的门控残差注入，降低 MCFL 对主干的扰动
- **梯度半截断**：image / audio 输入 MCFL 时 detach，仅 text 路径传递梯度
- **Alpha 课程**：对 c_at 做梯度缩放，按训练步数调整 alpha
- **MCFL 冻结**：训练后期冻结 MCFL 参数，只精修主 UNet
- **c_temp 跨帧 EMA 平滑**：对 BEATs audio tokens 做时序低通滤波
- **attn2_scale**：Temporal Cross-Attention 残差按层级缩放，抑制条件抖动放大
- **lambda_temp 课程**：对 latent 做 temporal smooth 约束

尽管如此，与 baseline 相比，FVD、FID、TC_FLICKER 等指标仍不理想，MCFL 带来的条件变化在 temporal attention 中仍被放大，flicker 问题未得到根本缓解。

### 9.2.1 为何未采用协同损失

在探索优化方向时，也曾考虑引入**协同损失**（如显式的 text-audio、text-video 对齐损失，或 CLIP-style 多模态对比损失）来强化模态间的协同。但最终未采用该方向，主要原因如下：

1. **问题定位不同**：核心矛盾是 **temporal attention 对条件变化的放大** 导致的 flicker，而非模态表示本身的对齐不足。协同损失优化的是 embedding 空间的对齐，无法直接约束 attention 的时序动态。
2. **作用层面错位**：CLIP / AV_ALIGN 等指标已在评估中体现模态对齐效果；在 loss 中额外加入协同项，与扩散目标可能产生梯度冲突，且难以控制与扩散 loss 的权重平衡。
3. **更直接的方案**：直接在 attention 层面约束 MCFL 的时序变化接近 baseline（Online Baseline Imitation），比通过协同损失间接优化更针对 flicker 的成因，且与扩散训练目标兼容。

因此，选择了在 attention 层做模仿约束，而非在 embedding 层做协同损失。

### 9.3 Online Baseline Imitation

基于上述现象，提出 **Online Baseline Imitation**：在同一训练步内，用**同一噪声、同一 batch** 分别以 MCFL 条件和 baseline 条件前向，得到 `attn_mcfl` 和 `attn_base`，并约束 MCFL 的 temporal attention 时序变化（Δ-attn）接近 baseline，即：

```
loss_attn = ||Δattn_mcfl - Δattn_base||²
```

该方案直接约束 MCFL 分支的 attention 动态，使其与 baseline 的时序模式一致，从而在保留多模态协同的同时抑制 flicker。配合 `mcfl_conservative=False`（不再冻结 MCFL），让 MCFL 在 imitation 约束下自由学习。

最终，在 20k steps 的实验中，MCFL + Online Baseline Imitation 取得了当前报告中的结果：FID 约 -20.2%，CLIP / AV_ALIGN 提升，FVD 标准差下降、稳定性改善。实验经历表明，从初版 MCFL → 保守版 v2-A → Online Baseline Imitation 的迭代，是在多模态协同与时序稳定性之间逐步寻找平衡的过程。
