# MCFL 设计修改详细报告

本文档记录针对 TIA2V 项目中 MCFL（Multi-modal Condition Fusion Layer）及 Online Baseline Imitation 的全部代码修改。

---

## 一、修改概览

| 模块 | 文件 | 修改类型 |
|------|------|----------|
| MCFL 开关与课程 | `diffusion/tacm_train_temp_util.py` | 新增参数、课程逻辑、开关 |
| MCFL 条件构建 | `diffusion/condition_builder.py` | 已有，无改动 |
| Baseline Imitation | `diffusion/attention.py` | CrossAttention 支持 return_attn |
| | `diffusion/attention_dual.py` | DualTemporalTransformer 传递 return_attn |
| | `diffusion/tacm_unet_temp_dual.py` | UNet/TimestepEmbedSequential 传递 return_attn |
| | `diffusion/tacm_train_temp_util.py` | 双前向、L_attn、λ_attn 调度 |
| 训练入口 | `scripts/train_temp.py` | 新增参数 |

---

## 二、MCFL 开关与课程设计

### 2.1 设计目标

- **Baseline 模式**：`use_mcfl=False` 时，训练行为与原版一致，无 MCFL 相关逻辑。
- **保守版 MCFL**：`mcfl_conservative=True` 时，启用 Alpha 课程、MCFL 冻结、Lambda_temp 课程。
- **Baseline Imitation 模式**：`mcfl_conservative=False` 且 `use_baseline_imitation=True` 时，MCFL 保持“自由”，不限制梯度，用于配合 attention imitation loss。

### 2.2 修改文件：`diffusion/tacm_train_temp_util.py`

#### 2.2.1 TrainLoop 新增参数

```python
# __init__ 新增
mcfl_conservative=True,   # True: alpha + freeze + lambda_temp 课程; False: 用于 baseline imitation
use_baseline_imitation=False,  # True: 启用 Temporal Δ-Attention Imitation
```

#### 2.2.2 Alpha Schedule（仅当 mcfl_conservative=True）

| 步数 | alpha | 说明 |
|------|-------|------|
| s < 4000 | 0.2 * (s/4000) | Stage 1: warmup |
| 4000 ≤ s < 8000 | 0.2 → 0.7 线性 | Stage 2: 学习 AV 对齐 |
| 8000 ≤ s < 10000 | 0.5 | Stage 3: 稳定 |
| s ≥ 10000 | 0.2 | Stage 4: FINAL trade-off |

实现：`c_at = alpha * c_at + (1 - alpha) * c_at.detach()`，对 c_at 做梯度缩放。

#### 2.2.3 MCFL 冻结（仅当 mcfl_conservative=True）

- 步数：s ≥ 8000
- 操作：`for p in self.mcfl.parameters(): p.requires_grad = False`
- 目的：后期只精修主 UNet，稳定视觉质量

#### 2.2.4 Lambda_temp Schedule（仅当 use_mcfl + mcfl_conservative + lambda_temp>0）

| 步数 | lambda_temp_now | 说明 |
|------|-----------------|------|
| s < 4000 | 0.0 | Stage 1: 无约束 |
| 4000 ≤ s < 8000 | 0.005 * ((s-4000)/4000) | Stage 2: 逐步增加 |
| 8000 ≤ s < 10000 | 0.005 → 0.02 线性 | Stage 3: 稳定 |
| s ≥ 10000 | 0.02 | Stage 4: 保持 |

Loss：`loss_temp = (micro_seq[:,1:] - micro_seq[:,:-1]).pow(2).mean()`，`loss += lambda_temp_now * loss_temp`。

---

## 三、Online Baseline Imitation（方案 A）

### 3.1 设计思路

- **位置**：UNet → TemporalTransformer → CrossAttention（`context_temp = c_at`），不是 MCFL 内部。
- **目标**：让 MCFL 的 temporal attention 在时间上的变化（Δ-attn）接近 baseline，减少 flicker。
- **Loss**：`L_attn = ((Δ_mcfl - Δ_base) ** 2).mean()`，其中 **Δ 必须沿 query/time 维**：`Δ = attn[:, 1:, :] - attn[:, :-1, :]`（❌ 错误：`attn[:, :, 1:] - attn[:, :, :-1]` 是 context 维）。

### 3.2 修改 1：`diffusion/attention.py`

#### CrossAttention.forward 新增 return_attn

```python
def forward(self, x, context=None, mask=None, n_times_crossframe_attn_in_self=0, return_attn=False, attn_cache=None):
    # ... 原有逻辑 ...
    attn = sim.softmax(dim=-1)

    if return_attn:
        # MCFL 分支必须保留计算图，不能 detach；baseline 用 no_grad 自动无梯度
        self.last_attn = attn  # [B*H, F, M]，softmax 后、dropout 前
        if attn_cache is not None:
            attn_cache.append(attn)  # collector：明确取最后一个 temporal block

    out = einsum('b i j, b j d -> b i d', attn, v)
    # ...
```

#### BasicTemporalTransformerBlock 传递 return_attn（含 checkpoint 修复）

**重要**：`return_attn`、`attn_cache`、`n_times_crossframe_attn_in_self` 不能作为 checkpoint 的 positional 参数，否则 backward 会对 bool/list/int 调 `.detach()` 导致 `AttributeError`。必须通过 module 状态传递：

```python
# __init__ 中
self._return_attn = False
self._attn_cache = None
self._n_times_crossframe_attn_in_self = 0

def forward(self, x, context=None, n_times_crossframe_attn_in_self=0, return_attn=False, attn_cache=None):
    self._return_attn = return_attn
    self._attn_cache = attn_cache
    self._n_times_crossframe_attn_in_self = n_times_crossframe_attn_in_self
    return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

def _forward(self, x, context=None):
    # ...
    x = self.attn2(self.norm2(x), context=context, return_attn=self._return_attn, attn_cache=self._attn_cache) + x
    # ...
```

#### TemporalTransformer.forward 传递 return_attn

```python
def forward(self, x, context_temp=None, return_attn=False):
    # ...
    for i, block in enumerate(self.transformer_blocks):
        x = block(x, context=context_temp, ..., return_attn=return_attn)
    # ...
```

### 3.3 修改 2：`diffusion/attention_dual.py`

#### DualTemporalTransformer.forward

```python
def forward(self, hidden_states, encoder_hidden_states, return_attn=False):
    for i in range(2):
        encoded_state = self.transformers[transformer_index](
            input_states,
            context_temp=condition_state,
            return_attn=return_attn,
        )[0]
    # ...
```

### 3.4 修改 3：`diffusion/tacm_unet_temp_dual.py`

#### TimestepEmbedSequential.forward

```python
def forward(self, x, emb, context=None, context_temp=None, return_attn=False):
    for layer in self:
        # ...
        elif isinstance(layer, TemporalTransformer):
            x = layer(x, context_temp=context_temp, return_attn=return_attn)
        elif isinstance(layer, DualTemporalTransformer):
            x = layer(x, context_temp, return_attn=return_attn)
    return x
```

#### UNet.forward

```python
def forward(self, x, timesteps, context=None, context_temp=None, y=None, **kwargs):
    return_attn = kwargs.get("return_attn", False)
    # ...
    for module in self.input_blocks:
        h = module(h, emb, context, context_temp, return_attn=return_attn)
    h = self.middle_block(h, emb, context, context_temp, return_attn=return_attn)
    for module in self.output_blocks:
        h = module(h, emb, context, context_temp, return_attn=return_attn)
    # ...
```

### 3.5 修改 4：`diffusion/tacm_train_temp_util.py`

#### 辅助函数 _get_last_temporal_attn

```python
def _get_last_temporal_attn(model):
    """从 attn_cache collector 获取最后一个 temporal cross-attn（attn2）"""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_attn_cache") and m._attn_cache is not None and len(m._attn_cache) > 0:
        return m._attn_cache[-1]
    return None
```

#### compute_losses 调用（use_baseline_imitation 时）

```python
if use_baseline_imitation and self.use_mcfl:
    micro_cond["return_attn"] = True
    noise = th.randn_like(micro)
    losses = compute_losses(noise=noise)  # 使用相同 noise，保证 x_t 一致
else:
    losses = compute_losses()
```

#### L_attn 计算与 λ_attn 调度

```python
if use_baseline_imitation and self.use_mcfl:
    attn_mcfl = _get_last_temporal_attn(self.ddp_model)  # 来自 compute_losses 前向

    # 构建 c_at_baseline（无 MCFL）
    c_t = c[:, :-1]
    image_cat = c[:, -1:]
    c_temp_raw = c_temp[:, :8, :]  # 前 8 个 token 为 audio
    _, c_at_baseline = build_conditions(c_t, image_cat, c_temp_raw, use_mcfl=False, mcfl=None)

    x_t = self.diffusion.q_sample(micro, t, noise=noise)

    # Baseline 前向（无梯度）
    with th.no_grad():
        _ = self.ddp_model(x_t, t_scaled, c, c_at_baseline, return_attn=True)
    attn_base = _get_last_temporal_attn(self.ddp_model)

    if attn_base is not None and attn_mcfl is not None:
        assert attn_mcfl.shape == attn_base.shape
        # Δ 沿 query/time 维 (dim 1)，attn: [*, F, M]
        delta_mcfl = attn_mcfl[:, 1:, :] - attn_mcfl[:, :-1, :]
        delta_base = attn_base[:, 1:, :] - attn_base[:, :-1, :]
        loss_attn = ((delta_mcfl - delta_base.detach()) ** 2).mean()

        # λ_attn 调度
        s = self.step + self.resume_step
        lambda_attn = 0.1 if s < 5000 else (0.05 if s < 10000 else 0.01)

        loss = loss + lambda_attn * loss_attn
        logger.logkv_mean("loss_attn", loss_attn.item())
        logger.logkv_mean("lambda_attn", lambda_attn)
```

---

## 四、condition_builder 与 MCFL 注入位置

### 4.1 MCFL 注入位置（未改代码，仅说明）

- **不是** latent 或 pixel 空间。
- **是** Attention 的条件：`c_fused` 注入 `c_at` → `context_temp` → TemporalTransformer → CrossAttention。
- 数据流：`text+image+audio → MCFL → c_fused → c_at → context_temp → temporal cross-attention`。

### 4.2 build_conditions 逻辑（已有）

- `use_mcfl=False` 或 `mcfl=None`：`c_at = concat(c_temp, c_t_expanded)`，baseline 分支。
- `use_mcfl=True` 且 `mcfl` 非空：`c_fused = c_text + λ*(MCFL(...) - c_text)`，`c_at = concat(c_temp, c_fused_expanded_seq)`。

---

## 五、训练入口参数

### 5.1 `scripts/train_temp.py` 新增/相关参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--use_mcfl` | False | 是否启用 MCFL |
| `--mcfl_embed_dim` | 768 | MCFL 嵌入维度 |
| `--mcfl_pooling_mode` | mean | mean / attention |
| `--mcfl_gate_lambda` | 0.2 | Gated residual 的 λ |
| `--lambda_temp` | 0.0 | Temporal smooth 权重，>0 时启用 |
| `--mcfl_conservative` | True | 是否使用保守课程（alpha + freeze + lambda_temp） |
| `--use_baseline_imitation` | False | 是否启用 Temporal Δ-Attention Imitation |

### 5.2 典型命令

**保守版 MCFL：**
```bash
python -m scripts.train_temp \
  --use_mcfl True \
  --mcfl_conservative True \
  --mcfl_gate_lambda 0.1 \
  --lambda_temp 0.01 \
  # ... 其他参数
```

**Baseline Imitation：**
```bash
python -m scripts.train_temp \
  --use_mcfl True \
  --mcfl_conservative False \
  --use_baseline_imitation True \
  # ... 其他参数
```

**Baseline（无 MCFL）：**
```bash
python -m scripts.train_temp \
  # 不加 --use_mcfl，默认 False
```

---

## 六、数据流与开关关系

```
                    use_mcfl?
                        │
        ┌───────────────┴───────────────┐
        │ False                         │ True
        ▼                               ▼
   build_conditions              build_conditions
   (use_mcfl=False)              (use_mcfl=True)
   c_at = baseline                c_at = MCFL fused
        │                               │
        └───────────────┬───────────────┘
                        ▼
                 UNet(context, context_temp)
                        │
        ┌───────────────┴───────────────────────────┐
        │ use_baseline_imitation?                    │
        │ True: 双前向 + L_attn                      │
        │ False: 仅扩散 loss（+ 可选 lambda_temp）   │
        └───────────────────────────────────────────┘
```

---

## 七、建议监控指标

| 指标 | 含义 |
|------|------|
| `loss_attn` | 应随训练下降 |
| `lambda_attn` | 当前 L_attn 权重 |
| `alpha_audio` | 当前 alpha（mcfl_conservative=True 时） |
| `lambda_temp` | 当前 temporal smooth 权重 |
| `loss_temp` | Temporal smooth loss |

---

## 八、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `diffusion/attention.py` | CrossAttention 增加 return_attn；BasicTemporalTransformerBlock、TemporalTransformer 传递 return_attn |
| `diffusion/attention_dual.py` | DualTemporalTransformer.forward 增加 return_attn |
| `diffusion/tacm_unet_temp_dual.py` | TimestepEmbedSequential、UNet.forward 增加 return_attn 传递 |
| `diffusion/tacm_train_temp_util.py` | mcfl_conservative、use_baseline_imitation；Alpha/Lambda_temp 课程；_get_last_temporal_attn；双前向与 L_attn |
| `scripts/train_temp.py` | 新增 mcfl_conservative、use_baseline_imitation 等参数 |
| `diffusion/condition_builder.py` | 无改动（已有 MCFL v2-A 逻辑） |

---

## 九、必须修改项（Sanity Check 修正）

### 9.1 已修正

| 项 | 修正内容 |
|----|----------|
| **Δ 差分维度** | `delta = attn[:, 1:, :] - attn[:, :-1, :]`（沿 query/time 维 F，非 context 维 M） |
| **MCFL attn 不 detach** | `self.last_attn = attn`（无 `.detach()`），baseline 分支用 `no_grad` |
| **同一 noise/x_t** | `noise = randn_like(micro)`，MCFL 与 baseline 共用同一 `x_t` |
| **保存 softmax 后** | 保存 `attn = softmax(sim)`，dropout 在 to_out，不保存 out |
| **shape 断言** | `assert attn_mcfl.shape == attn_base.shape` |
| **attn_cache collector** | 用 `attn_cache` 列表替代 `modules()` 扫描，取 `attn_cache[-1]` |
| **只约束 attn2** | 仅 BasicTemporalTransformerBlock.attn2 参与 collector |
| **baseline imitation 时** | `mcfl_conservative=False`，保持 MCFL 自由 |

### 9.2 训练日志必看

- `loss_attn`：前期下降，后期不归零
- `lambda_attn`：按 schedule 变化（0.1→0.05→0.01）
- Temporal attn 的 frame-to-frame 方差：应下降
- CLIP / AV_ALIGN：不明显下降
- TC_FLICKER：方差先降

### 9.3 λ_attn 建议

- 0–5k：0.1
- 5k–10k：0.05
- 10k+：0.01 或关
- ❌ 不要 >0.2，❌ 不要常数跑全程

---

*报告生成时间：2026-01-30*
