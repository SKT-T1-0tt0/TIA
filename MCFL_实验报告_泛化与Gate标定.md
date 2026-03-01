# MCFL 多数据集泛化实验报告：结果对比、分析与改动汇总

---

## 一、实验结果数据对比

多数据集上 Baseline（无 MCFL）与 MCFL 的指标对比如下（均值 ± 标准差，Δ = MCFL − Baseline）。

### 1. FVD（Fréchet Video Distance，越低越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 35.54 ± 3.52 | 36.52 ± 2.08 | +0.98 |
| post_landscape | 65.16 ± 9.75 | 52.27 ± 4.70 | **−12.89** |
| post_audioset_drums | 91.56 ± 8.71 | 115.01 ± 6.91 | **+23.45** |
| **平均值** | 64.09 | 67.93 | +3.84 |

### 2. FID（Fréchet Inception Distance，越低越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 569.04 ± 10.82 | 456.03 ± 11.70 | **−113.00** |
| post_landscape | 809.24 ± 30.33 | 664.65 ± 19.79 | **−144.58** |
| post_audioset_drums | 927.54 ± 68.06 | 1840.19 ± 23.68 | **+912.65** |
| **平均值** | 768.61 | 986.96 | +218.35 |

### 3. FFC（First Frame Consistency，越低越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 0.1759 ± 0.0003 | 0.1761 ± 0.0003 | +0.0002 |
| post_landscape | 0.1752 ± 0.0029 | 0.1957 ± 0.0035 | +0.0205 |
| post_audioset_drums | 0.2734 ± 0.0014 | 0.2329 ± 0.0050 | **−0.0405** |
| **平均值** | 0.21 | 0.20 | −0.0066 |

### 4. CLIP（文本-视频一致性，越高越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 0.2860 ± 0.0208 | 0.2922 ± 0.0129 | **+0.0062** |
| post_landscape | 0.1940 ± 0.0084 | 0.1899 ± 0.0141 | −0.0041 |
| post_audioset_drums | 0.2233 ± 0.0059 | 0.2224 ± 0.0072 | −0.0008 |
| **平均值** | 0.23 | 0.23 | +0.0004 |

### 5. AV_ALIGN（音视频对齐，越高越好）

| 数据集 | Baseline | MCFL | Δ |
|--------|----------|------|-----|
| post_URMP | 0.4196 ± 0.1204 | 0.4281 ± 0.2153 | +0.0085 |
| post_landscape | 0.4450 ± 0.2693 | 0.5034 ± 0.1839 | +0.0584 |
| post_audioset_drums | 0.4260 ± 0.1787 | 0.2788 ± 0.2202 | **−0.1473** |
| **平均值** | 0.43 | 0.40 | −0.0268 |

### 6. TC_FLICKER（时间一致性 / 闪烁，flicker 越低越好，tc 为负值、绝对值越小越稳）

| 数据集 | Baseline (flicker) | MCFL (flicker) | Δ |
|--------|--------------------|----------------|-----|
| post_URMP | 22.65 ± 5.91 | 29.97 ± 10.31 | +7.32 |
| post_landscape | 309.80 ± 373.32 | 333.75 ± 331.19 | +23.95 |
| post_audioset_drums | 69.25 ± 50.51 | 72.66 ± 33.15 | +3.41 |
| **平均值** | 133.90 | 145.46 | +11.56 |

---

## 二、分析

### 2.1 现象归纳

- **post_URMP**：MCFL 在 **FID 上大幅优于 Baseline**（−113.00），CLIP、AV_ALIGN 略升（+0.0062、+0.0085），FVD 略差（+0.98）、FFC 基本持平。**TC_FLICKER** 上升（22.65→29.97，Δ +7.32），方差 10.31，属适度变差、仍可控，整体上 MCFL 在 URMP 是有益的。
- **post_landscape**：MCFL 在 FVD、FID 上明显优于 Baseline（−12.89、−144.58），FFC 略差（+0.02）、CLIP 略降，整体收益明确。
- **post_audioset_drums**：**FID +912、FVD +23**，AV_ALIGN 明显下降（−0.15），属于典型的“跨分布崩坏”；FFC 反而略好（−0.04），说明问题主要在**生成质量与音视频对齐**，而非首帧一致性。

**汇总**：三数据集平均看，FID 被 drums 严重拉高（+218），FVD 略升（+3.84），TC_FLICKER 平均上升（+11.56）。**主要矛盾在 drums 的跨分布崩坏**；URMP 与 landscape 上 MCFL 整体是正向的，URMP 仅 flicker 有适度代价。

### 2.2 原因分析

1. **Gate 标定与分布不匹配**  
   - 原 gate 置信度区间为 **[5, 30]**，中心约 17.5；而 drums 的 BEATs pooled norm 集中在 **8.3–8.9**。  
   - 在此区间内 conf 偏小、gate 偏低，本应“该开时开”的条件被压得过弱；URMP、landscape 的 norm 分布（如 URMP p5–p95 约 7.44–9.79）与 drums 差异大，同一套 [5, 30] 无法同时合理覆盖三数据集，导致**跨集行为不一致**，drums 上易出现条件异常。

2. **双重非线性压缩**  
   - 原默认：数据侧 `audio_soft_clip=tanh` + BEATs 前 `audio_response=tanh`。  
   - 对 drums 等强瞬态信号，**两次非线性**易改变波形形状、使 BEATs 特征分布偏移，进而 MCFL 融合后条件异常，FID/FVD 崩坏。  
   - 更合理的做法是**只压一次**（数据侧或 BEATs 前二选一）。

3. **条件注入时间不稳定**  
   - drums 的 **per-frame norm** 波动大（max 11.49，std 0.50），而 **pooled norm** 很稳（std 0.16）。  
   - 若 gate 或融合输出**未做时间平滑**，帧间条件强度跳变会推高 **TC_FLICKER**；URMP 上 flicker 的适度上升（+7.32）与此一致，可通过 gate/embedding 时间平滑进一步缓解。

4. **幅度与分布差异**  
   - drums 原始音频 peak/RMS 差异大，且多数未做峰值归一化；若训练时没有 **Random Gain** 等增广，模型对“幅度不可靠”的鲁棒性不足，跨集泛化差。

### 2.3 改进方向（与后续改动对应）

- **Gate**：用三数据集 BEATs pooled norm 的 p5–p95 做**统一标定**（如 [7.2, 10.0]），或改为 **z-score** 标定，使“该开时开、异常时关”。  
- **单次压缩**：采用 **方案 A**（peak + soft_clip=none + response=compand）或 **方案 B**（peak + soft_clip=compand + response=none），避免双重 tanh/compand。  
- **时间平滑**：对 **gate** 和 **audio embedding** 做时间 EMA（如 gate_ema=0.9），减轻 flicker。  
- **训练增广**：**Random Gain**（peak 之后）+ 可选 **Modality Dropout**，提升对幅度与缺失音频的鲁棒性。

---

## 三、本次更改汇总

围绕“**gate 标定 + 条件平滑 + 单次压缩 + 训练增广**”，对代码与配置做了如下改动与新增。

### 3.1 数据与音频管线（`tacm/data.py`）

- **音频幅度归一化**：`normalize_and_clip_audio()`，支持 `audio_norm_mode`：`none` / `peak` / `rms`。  
- **数据侧 soft clipping**：`audio_soft_clip`：`none` / `tanh` / `compand`；**默认改为 `none`**，避免与 BEATs 前 response 形成双重压缩。  
- 新增参数：`--audio_norm_mode`、`--audio_soft_clip`、`--audio_rms_target`。

### 3.2 训练侧音频与条件（`diffusion/tacm_train_temp_util.py`）

- **Random Gain**：在 BEATs 前对音频乘 `g ~ log_uniform(audio_gain_low, audio_gain_high)`，默认 [0.25, 4]，可改为 [0.5, 2] 起步。  
- **随机响应强度**：`audio_response=tanh` 时 k~U(0.5,2)，`compand` 时 μ~U(2,10)。  
- **单次压缩**：仅在 BEATs 前做 `audio_response`（tanh/compand），数据侧默认不再做 soft_clip。  
- **Modality Dropout**：以 `modality_dropout_prob`（默认 0.2）将 audio 条件置零，学会无可靠音频时仍稳定生成。  
- **c_temp 时间平滑**：对 BEATs 输出做跨帧 EMA（alpha=0.9），与推理一致。

### 3.3 MCFL 与 Gate（`diffusion/condition_builder.py`）

- **Gate 区间默认**：由 [5, 30] 改为 **[7.2, 10.0]**，基于三数据集 pooled norm 的 p5–p95 统一标定。  
- **Gate 时间平滑**：支持 per-frame gate + 时间 EMA，`mcfl_gate_ema` 默认 **0.9**。  
- **Z-score 标定（可选）**：`mcfl_gate_use_zscore=True` 时，用 `z=(norm−μ)/σ` 与 `z_low/z_high` 计算 conf，便于跨数据集只更新 μ/σ。  
- **L2 归一化**：送入 MCFL 前对 image/audio 做 L2 归一化（`mcfl_norm_modality=True`），减弱幅值差异带来的影响。

### 3.4 推理一致性与参数传递

- **采样脚本**（`scripts/sample_motion_optim.py`）：  
  - 对 BEATs 输出做与训练相同的 **c_temp EMA 平滑**；  
  - 传入与训练一致的 gate 与 z-score 参数（norm_low/high、ema、use_zscore、mu、sigma、z_low/z_high）。  
- **训练脚本**（`scripts/train_temp.py`）：  
  - 默认及传参更新为上述 gate、z-score、Random Gain、Modality Dropout、单次压缩等。

### 3.5 统计与标定工具

- **`scripts/audio_beats_stats.py`**：  
  - 对指定数据集统计**原始音频**（peak、RMS、时长、是否近似归一）；  
  - 可选 **BEATs pooled / per-frame norm** 分布，并给出单数据集 gate 建议（low/high）。  
- **`scripts/unify_gate_calibration.py`**：  
  - 读入多份 stats JSON，输出**统一**的 `mcfl_gate_norm_low/high` 及 z-score 的 `mu`、`sigma`，便于训练/推理直接使用。

### 3.6 文档与推荐配置

- **`scripts/MCFL_GATE_CALIBRATION.md`**：  
  - 三数据集 BEATs pooled norm 统计表；  
  - 统一标定策略（绝对区间 + z-score）；  
  - 推荐默认参数、单次压缩方案 A/B、Random Gain 顺序、验证矩阵 4 组实验说明。  
- **推荐训练默认**：  
  - Gate [7.2, 10.0]，gate_ema=0.9；  
  - 单次压缩方案 A：`audio_norm_mode=peak`，`audio_soft_clip=none`，`audio_response=compand`；  
  - Random Gain 开启，可选先 [0.5, 2] 再 [0.25, 4]；  
  - Modality Dropout=0.2。

### 3.7 其它

- **`diffusion/dist_util.py`**：`setup_dist()` 中 hostname 解析失败时回退到 `127.0.0.1`，避免无 DNS 环境报错。  
- **`evaluation_report_audio_beats_stats.json`**：post_audioset_drums 的音频与 BEATs 统计结果（示例）。

---

## 四、小结

- **实验数据**：  
  - **post_URMP**：MCFL 明显提升 FID（−113），CLIP/AV_ALIGN 略升，FVD 略差（+0.98），TC_FLICKER 适度上升（+7.32，29.97±10.31），整体有益。  
  - **post_landscape**：MCFL 在 FVD、FID 上明显优于 Baseline，整体收益明确。  
  - **post_audioset_drums**：FID/FVD 崩坏、AV_ALIGN 下降，为当前主要问题；三集平均被 drums 拉高。  
- **原因**：gate 区间与真实 BEATs norm 分布脱节、双重非线性压缩、条件时间不平滑、以及跨集幅度/分布差异未在训练中显式覆盖。  
- **本次改动**：统一 gate 标定（[7.2, 10.0] + 可选 z-score）、单次压缩默认、gate/embedding 时间平滑、Random Gain + Modality Dropout、统计与标定脚本及文档，形成一套可复现的“默认矩阵”与验证流程。  

后续建议：使用**默认矩阵**在 drums/URMP/landscape 上复训并重新评估；若 drums 仍敏感，可再按 `MCFL_GATE_CALIBRATION.md` 中的验证矩阵做只压一次 + tight gate [8.3, 8.9] 等消融，并视情况启用 z-score 标定。
