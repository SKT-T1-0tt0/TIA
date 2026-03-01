# 整合版视频评估脚本使用说明

## 概述

`eval_all.py` 是一个整合版评估脚本，统一管理所有视频评估指标，包括：

- **FVD** (Fréchet Video Distance) - 视频质量评估
- **FID** (Fréchet Inception Distance) - 图像质量评估（视频帧）
- **FFC** (Frame-to-Frame Consistency) - 帧间一致性评估
- **CLIP** - 文本-视频相似度评估
- **AV-Align** - 音频-视频对齐评估
- **TC-Flicker** - 时间一致性/闪烁评估

## 快速开始

### 基本用法

```bash
# 运行所有评估指标
python eval_all.py \
    --real_dir results/1_tacm_/real \
    --baseline_dir results/0_tacm_/fake1_30fps \
    --mcfl_dir results/1_tacm_/fake1_30fps \
    --metrics fvd fid ffc clip av_align tc_flicker
```

### 只运行特定指标

```bash
# 只运行 FVD 和 FID
python eval_all.py \
    --real_dir results/1_tacm_/real \
    --baseline_dir results/0_tacm_/fake1_30fps \
    --mcfl_dir results/1_tacm_/fake1_30fps \
    --metrics fvd fid
```

### 保存报告到文件

```bash
python eval_all.py \
    --real_dir results/1_tacm_/real \
    --baseline_dir results/0_tacm_/fake1_30fps \
    --mcfl_dir results/1_tacm_/fake1_30fps \
    --metrics fvd fid ffc \
    --output evaluation_report.txt
```

报告会保存为：
- 文本报告：`evaluation_report.txt`
- JSON 数据：`evaluation_report.json`

## 命令行参数

### 必需参数

- `--real_dir`: 真实视频目录（groundtruth）

### 可选参数

- `--baseline_dir`: Baseline 生成视频目录
- `--mcfl_dir`: MCFL 生成视频目录
- `--metrics`: 要运行的评估指标列表（默认：所有指标）
  - 可选值：`fvd`, `fid`, `ffc`, `clip`, `av_align`, `tc_flicker`
- `--prompt_file`: 提示词文件路径（用于 CLIP 评估，默认：`prompts.txt`）
- `--device`: 计算设备（默认：`cuda`）
- `--bootstrap_k`: Bootstrap 采样次数（默认：`10`）
- `--no_bootstrap`: 禁用 bootstrap 采样
- `--no_cache`: 禁用特征缓存
- `--output`: 输出报告文件路径

## 评估指标说明

### FVD (Fréchet Video Distance)
- **用途**: 评估生成视频与真实视频的分布差异
- **越低越好**: ✓
- **需要**: I3D 模型

### FID (Fréchet Inception Distance)
- **用途**: 评估生成视频帧与真实视频帧的分布差异
- **越低越好**: ✓
- **需要**: FID Inception 模型

### FFC (Frame-to-Frame Consistency)
- **用途**: 评估视频的帧间一致性（时间平滑度）
- **越低越好**: ✓
- **需要**: RAFT 光流模型（或使用 OpenCV 作为后备）

### CLIP Text-Video Similarity
- **用途**: 评估生成视频与文本提示词的语义相似度
- **越高越好**: ✓
- **需要**: CLIP 模型和 `prompts.txt` 文件

### AV-Align (Audio-Video Alignment)
- **用途**: 评估视频运动与音频的同步性
- **越高越好**: ✓
- **需要**: 音频文件（`.wav` 格式）

### TC-Flicker (Temporal Consistency - Flicker)
- **用途**: 评估视频的时间一致性（闪烁程度）
- **越低越好**: ✓（Flicker），越高越好（TC）

## 输出格式

脚本会输出一个表格，包含所有评估指标的结果：

```
Metric              | Baseline              | MCFL                  | Δ
--------------------------------------------------------------------------------
FVD                 | 123.45 ± 3.21        | 115.67 ± 2.98         | -7.78
FID                 | 45.23 ± 1.12         | 42.15 ± 0.98          | -3.08
FFC                 | 0.0234 ± 0.0012      | 0.0198 ± 0.0010       | -0.0036
CLIP                | 0.8234 ± 0.0123      | 0.8456 ± 0.0109        | +0.0222
AV_ALIGN            | 0.7234 ± 0.0234      | 0.7456 ± 0.0212       | +0.0222
TC_FLICKER (flicker)| 0.001234 ± 0.000123  | 0.000987 ± 0.000098   | -0.000247
TC_FLICKER (tc)     | -0.001234 ± 0.000123 | -0.000987 ± 0.000098  | +0.000247
```

## 注意事项

1. **特征缓存**: 默认启用特征缓存，可以显著加速重复评估。缓存文件保存在：
   - FVD: `fvd_cache/`
   - FID: `fid_cache/`
   - FFC: `ffc_cache/`

2. **Bootstrap 采样**: 默认启用 bootstrap 采样，提供更稳健的统计结果（均值 ± 标准差）

3. **视频数量对齐**: 如果真实视频、Baseline 和 MCFL 的视频数量不一致，脚本会自动对齐到最小数量

4. **依赖项**: 确保已安装所有必需的依赖：
   - PyTorch
   - NumPy
   - OpenCV
   - CLIP (`pip install git+https://github.com/openai/CLIP.git`)
   - I3D 模型（会自动下载或使用本地版本）
   - RAFT 模型（可选，用于更准确的 FFC）

## 示例

### 示例 1: 完整评估

```bash
python eval_all.py \
    --real_dir results/1_tacm_/real \
    --baseline_dir results/0_tacm_/fake1_30fps \
    --mcfl_dir results/1_tacm_/fake1_30fps \
    --metrics fvd fid ffc clip av_align tc_flicker \
    --output full_evaluation_report.txt
```

### 示例 2: 快速评估（只运行 FVD 和 FID）

```bash
python eval_all.py \
    --real_dir results/1_tacm_/real \
    --baseline_dir results/0_tacm_/fake1_30fps \
    --mcfl_dir results/1_tacm_/fake1_30fps \
    --metrics fvd fid \
    --bootstrap_k 5
```

### 示例 3: 只评估 Baseline（不评估 MCFL）

```bash
python eval_all.py \
    --real_dir results/1_tacm_/real \
    --baseline_dir results/0_tacm_/fake1_30fps \
    --metrics fvd fid ffc
```

## 故障排除

### 问题 1: 模块导入失败

如果某个评估模块导入失败，脚本会跳过该指标并继续运行其他指标。

### 问题 2: 内存不足

如果遇到内存不足问题，可以：
- 禁用缓存（`--no_cache`）
- 减少 bootstrap 采样次数（`--bootstrap_k 5`）
- 分批运行不同的指标

### 问题 3: CUDA 内存不足

如果 GPU 内存不足，可以：
- 使用 CPU（`--device cpu`）
- 减少同时运行的评估指标数量

## 与原始脚本的兼容性

整合脚本完全兼容原始的独立评估脚本（`eval_fvd.py`, `eval_fid_video.py` 等）。你可以继续使用原始脚本，也可以使用整合脚本进行批量评估。

## 更新日志

- **v1.0** (2025-01-17): 初始版本，整合所有评估指标
