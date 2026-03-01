# MCFL Gate 统一标定策略（基于三数据集 BEATs pooled norm）

## 1. 三数据集 BEATs pooled norm 统计（16 段 clip + mean-pool 8 tokens）

| 数据集 | mean | std | p5 | p95 | min | max |
|--------|------|-----|-----|-----|-----|-----|
| post_audioset_drums | 8.57 | 0.16 | 8.30 | 8.82 | 8.26 | 8.86 |
| post_URMP           | 8.08 | 0.69 | 7.44 | 9.79 | 7.33 | 9.96 |
| post_landscape      | 8.58 | 0.23 | 8.19 | 8.90 | 8.18 | 9.07 |

- **结论**：drums/landscape 很集中，URMP 分布最宽（std 0.69）、p5–p95 跨 [7.44, 9.79]。

---

## 2. 统一标定策略（同时覆盖三数据集、无需手调）

### 策略 A：绝对区间（三角形 conf）

- **推荐默认**（覆盖三数据集 p5–p95 并留余量）：
  - `mcfl_gate_norm_low = 7.2`
  - `mcfl_gate_norm_high = 10.0`
- 中心 ≈ 8.6，norm 在 [7.2, 10.0] 内 conf 从 0→1→0，超出则 conf→0。
- **仅 drums 紧标定**（若只关心 drums、且希望 conf 更“可控”）：low=8.3, high=8.9。

### 策略 B：Z-score 标定（更稳，跨数据集只更新统计）

- 训练阶段用 `scripts/audio_beats_stats.py` 在**训练集**上跑一遍，得到 pooled norm 的 **μ_train, σ_train**（或合并多数据集的加权均值/方差）。
- 推理时：
  - `z = (norm - μ_train) / (σ_train + eps)`
  - `conf = clamp((z - z_low)/(z_high - z_low), 0, 1)`，例如 `z_low=-1.5, z_high=1.5`。
- 基于当前三数据集近似：
  - `mcfl_gate_norm_mu = 8.4`
  - `mcfl_gate_norm_sigma = 0.5`
  - `mcfl_gate_z_low = -1.5`, `mcfl_gate_z_high = 1.5`
- 换数据集时只需重新跑 stats 更新 μ/σ，不必改 low/high。

---

## 3. 默认参数建议（“不炸、泛化更稳”）

| 参数 | 推荐默认 | 说明 |
|------|----------|------|
| mcfl_gate_norm_low | **7.2** | 统一覆盖三数据集 p5 以下 |
| mcfl_gate_norm_high | **10.0** | 统一覆盖三数据集 p95 以上 |
| mcfl_gate_ema | **0.9** | gate 时间平滑更强，减轻 flicker |
| mcfl_gate_use_zscore | False（可选 True） | 用策略 B 时设为 True 并设 mu/sigma |

若使用 **z-score**（`mcfl_gate_use_zscore=True`）：

| 参数 | 推荐默认 |
|------|----------|
| mcfl_gate_norm_mu | 8.4 |
| mcfl_gate_norm_sigma | 0.5 |
| mcfl_gate_z_low | -1.5 |
| mcfl_gate_z_high | 1.5 |

---

## 4. 单次压缩 + 验证矩阵（快速定位问题）

### 原则：只压一次，避免“数据侧 soft_clip + BEATs 前 response”双重非线性。

- **方案 A（推荐）**：`audio_norm_mode=peak`, `audio_soft_clip=none`, `audio_response=compand`
- **方案 B**：`audio_norm_mode=peak`, `audio_soft_clip=compand`, `audio_response=none`

### Random Gain 顺序（训练，当前实现已按此顺序）

1. **数据侧**：`audio_norm_mode=peak`（保证不爆，再乘 g 时意义稳定）
2. **Train loop**：Random Gain（peak 之后）：`x *= g`, `g ~ log_uniform(audio_gain_low, audio_gain_high)`
   - 起步建议 `--audio_gain_low 0.5 --audio_gain_high 2`，不稳再扩到 `[0.25, 4]`
3. **单次压缩**：`audio_response=compand` 或 `tanh`（只做一次，数据侧 `audio_soft_clip=none`）
4. BEATs

### 验证矩阵（只在 post_audioset_drums 上小规模跑）

| 组 | 配置 | 观察目标 |
|----|------|----------|
| 1 | 固定：gate 平滑 + embedding 平滑；只压一次（方案 A）；当前 gate [5,30] | 若明显变好 → 主要是“双重非线性” |
| 2 | 只压一次（A）+ tight gate [8.3, 8.9] | 若变好 → gate 标定关键 |
| 3 | 只压一次（B）+ tight gate | 对比 A/B |
| 4 | 在 (2) 上加 Random Gain 训练增广 | 泛化/稳定性 |

若 (1) 不变但 (2) 变好 → gate 标定是关键；若 (2)(3) 都好但 flicker 仍高 → 加强 gate/embedding EMA。

---

## 5. 如何生成/更新统计

```bash
# 各数据集分别跑（含 BEATs）
python scripts/audio_beats_stats.py --data_path datasets/post_audioset_drums --with_beats --out stats_drums.json
python scripts/audio_beats_stats.py --data_path datasets/post_URMP --with_beats --out stats_urmp.json
python scripts/audio_beats_stats.py --data_path datasets/post_landscape --with_beats --out stats_landscape.json

# 合并得到统一 low/high 与 z-score 默认（见 scripts/unify_gate_calibration.py）
python scripts/unify_gate_calibration.py stats_drums.json stats_urmp.json stats_landscape.json
```
