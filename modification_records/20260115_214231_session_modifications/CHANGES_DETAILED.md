# 详细修改说明

本文档详细记录了本次会话中所有文件的修改内容。

## 文件修改清单

### 1. `diffusion/tacm_train_util.py`

**修改位置1**: 第 12 行
- **原始**: `from tacm.download import load_vqgan`
- **修改后**: `# from tacm.download import load_vqgan`

**修改位置2**: 第 125 行
- **原始**: `self.init_first_stage_from_ckpt()`
- **修改后**: `#self.init_first_stage_from_ckpt()`

**修改位置3**: 第 128-135 行
- **原始**: 
```python
def init_first_stage_from_ckpt(self):
    self.first_stage_model = load_vqgan(self.vqgan_ckpt)
    for p in self.first_stage_model.parameters():
        p.requires_grad = False
    self.first_stage_model.codebook._need_init = False
    self.first_stage_model.eval()
    self.first_stage_model.train = disabled_train
    self.first_stage_vocab_size = self.first_stage_model.codebook.n_codes
```
- **修改后**: 整个方法被注释掉

---

### 2. `tacm/modules/__init__.py`

**修改位置1**: 第 3-7 行
- **原始**: `from .lpips import LPIPS`
- **修改后**: 
```python
try:
    from .lpips import LPIPS
except ImportError:
    # LPIPS module is optional, only needed for VQGAN training
    LPIPS = None
```

**修改位置2**: 第 11-15 行
- **原始**: `from .audioclip import AudioCLIP`
- **修改后**: 
```python
try:
    from .audioclip import AudioCLIP
except ImportError:
    # AudioCLIP module is optional, only needed when audio_emb_model='audioclip'
    AudioCLIP = None
```

---

### 3. `tacm/data.py`

**修改位置**: 第 30-34 行
- **原始**: `from tacm.modules import AudioCLIP`
- **修改后**: 
```python
try:
    from tacm.modules import AudioCLIP
except ImportError:
    # AudioCLIP is optional, only needed when audio_emb_model='audioclip'
    AudioCLIP = None
```

---

### 4. `tacm/vqgan.py`

**修改位置1**: 第 65-69 行（初始化）
- **原始**: `self.perceptual_model = LPIPS().eval()`
- **修改后**: 
```python
# LPIPS is optional, only needed for perceptual loss
if LPIPS is not None:
    self.perceptual_model = LPIPS().eval()
else:
    self.perceptual_model = None
```

**修改位置2**: 第 132-134 行（forward方法中使用）
- **原始**: 
```python
perceptual_loss = 0
if self.perceptual_weight > 0:
    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
```
- **修改后**: 
```python
perceptual_loss = 0
if self.perceptual_weight > 0 and self.perceptual_model is not None:
    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
```

**修改位置3**: 第 194-196 行（validation_step中使用）
- **原始**: `perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight`
- **修改后**: 
```python
perceptual_loss = 0
if self.perceptual_weight > 0 and self.perceptual_model is not None:
    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
```

---

### 5. `tacm/cm_vqgan.py`

**修改位置1**: 第 66-70 行（初始化）
- **原始**: `self.perceptual_model = LPIPS().eval().to(torch.device("mps"))`
- **修改后**: 
```python
# LPIPS is optional, only needed for perceptual loss
if LPIPS is not None:
    self.perceptual_model = LPIPS().eval().to(torch.device("mps"))
else:
    self.perceptual_model = None
```

**修改位置2**: 第 127-128 行（forward方法中使用）
- **原始**: 
```python
perceptual_loss = 0
if self.perceptual_weight > 0:
    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
```
- **修改后**: 
```python
perceptual_loss = 0
if self.perceptual_weight > 0 and self.perceptual_model is not None:
    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
```

**修改位置3**: 第 188-189 行（validation_step中使用）
- **原始**: `perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight`
- **修改后**: 
```python
perceptual_loss = 0
if self.perceptual_weight > 0 and self.perceptual_model is not None:
    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
```

---

### 6. `tacm/utils.py`

**修改位置**: 第 14-47 行（整个导入部分）
- **原始**: `import tacm.modules.ignite_trainer as it`
- **修改后**: 
```python
# Delay import of ignite_trainer to avoid circular import
# ignite_trainer is imported lazily when needed
_it_module = None

def _get_ignite_trainer():
    """Lazy import of ignite_trainer to avoid circular import."""
    global _it_module
    if _it_module is None:
        try:
            import tacm.modules.ignite_trainer as it
            _it_module = it
        except (ImportError, ModuleNotFoundError):
            # ignite_trainer is optional, create a simple AbstractTransform base class
            import abc
            from typing import Callable
            
            class AbstractTransform(abc.ABC, Callable[[torch.Tensor], torch.Tensor]):
                @abc.abstractmethod
                def __call__(self, x: torch.Tensor) -> torch.Tensor:
                    pass
            
            # Create a module-like object with AbstractTransform attribute
            class _IgniteTrainerModule:
                pass
            
            _it_module = _IgniteTrainerModule()
            _it_module.AbstractTransform = AbstractTransform
    return _it_module

# Create a proxy object for backward compatibility
class _ITProxy:
    @property
    def AbstractTransform(self):
        return _get_ignite_trainer().AbstractTransform

it = _ITProxy()
```

---

### 7. `scripts/sample_motion_optim.py`

**修改位置1**: 第 32-43 行（导入部分）
- **原始**: 
```python
from optimization.video_editor_simple import VideoEditor
from optimization.arguments import get_arguments
```
- **修改后**: 
```python
try:
    from optimization.video_editor_simple import VideoEditor
except (ImportError, ModuleNotFoundError):
    # VideoEditor is optional, only needed for video editing functionality
    VideoEditor = None

try:
    from optimization.arguments import get_arguments
except (ImportError, ModuleNotFoundError):
    # get_arguments is optional, only needed when VideoEditor is used
    def get_arguments(parser):
        return parser
```

**修改位置2**: 第 218-230 行（使用部分）
- **原始**: 
```python
video_editor = VideoEditor(args)
pred_video = video_editor.edit_video_by_prompt(video, audio=None, raw_text=None, text=batch['text'].to(dist_util.dev()))
```
- **修改后**: 
```python
if VideoEditor is not None:
    video = sample_recon.squeeze()
    video = Func.interpolate(video, size=(128, 128), mode='bilinear',align_corners=False)
    logger.log("creating video editor...")
    video_editor = VideoEditor(args)
    pred_video = video_editor.edit_video_by_prompt(video, audio=None, raw_text=None, text=batch['text'].to(dist_util.dev()))

    logger.log("save to mp4 format...")
    os.makedirs("./results/%d_tacm_%s/fake2_6fps"%(args.run, args.dataset), exist_ok=True)
    save_video_grid(pred_video+0.5, os.path.join("./results/%d_tacm_%s"%(args.run, args.dataset), "fake2_6fps", f"video_%d.mp4"%(i)), 1)
else:
    logger.log("VideoEditor not available, skipping video editing step...")
```

---

### 8. `beats/BEATs.py`

**修改位置**: 第 118-136 行（preprocess方法）
- **原始**: 
```python
def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
) -> torch.Tensor:
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank
```
- **修改后**: 
```python
def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
) -> torch.Tensor:
    # Save original device to move results back later
    original_device = source.device
    
    fbanks = []
    for waveform in source:
        # Move waveform to CPU for fbank computation to avoid cuFFT errors
        waveform_cpu = waveform.cpu().unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform_cpu, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        # Move result back to original device
        fbank = fbank.to(original_device)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank
```

---

### 9. `diffusion/tacm_train_temp_util.py`

**修改位置**: 第 19 行
- **原始**: `from tacm import AudioCLIP`
- **修改后**: 
```python
try:
    from tacm import AudioCLIP
except (ImportError, ModuleNotFoundError):
    # AudioCLIP is optional, only needed when audio_emb_model='audioclip'
    AudioCLIP = None
```

---

## 恢复方法

要恢复原始代码，请按照以下步骤：

1. 对于注释掉的代码：取消注释即可
2. 对于可选导入：将 try-except 块替换为直接导入
3. 对于容错处理：移除 None 检查，直接使用
4. 对于 CPU 计算：移除 `.cpu()` 和 `.to(original_device)` 调用

所有修改都是可逆的，可以随时恢复。
