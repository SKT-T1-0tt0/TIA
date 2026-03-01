import copy
import functools
import os

import transformers.image_transforms
from einops import rearrange, repeat

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from tacm.download import load_vqgan
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler
from .tacm_nn import update_ema
from .condition_builder import build_conditions
from tacm import AudioCLIP
import wav2clip
from beats.BEATs import BEATs, BEATsConfig

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def _get_last_temporal_attn(model):
    """Get attn from collector (last temporal cross-attn block, attn2 only)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_attn_cache") and m._attn_cache is not None and len(m._attn_cache) > 0:
        return m._attn_cache[-1]
    return None


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class TrainLoop():
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        save_dir,
        vqgan_ckpt,
        sequence_length,
        audio_emb_model,
        use_mcfl=False,
        mcfl_embed_dim=768,
        mcfl_pooling_mode="mean",  # "mean" or "attention"
        mcfl_gate_lambda=0.1,  # MCFL v2-A gate parameter (0.1 降低 TC_FLICKER，原 0.2)
        lambda_temp=0.0,  # Temporal smooth regularization weight (default 0.0 = disabled, set > 0 to enable, e.g., 0.01)
        mcfl_conservative=True,  # If True: alpha curriculum + MCFL freeze at 8k + lambda_temp curriculum. If False: full MCFL, no curriculum (for use with baseline imitation).
        use_baseline_imitation=False,  # If True: add online baseline output imitation loss (placeholder; implement when needed).
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.save_dir = save_dir
        self.vqgan_ckpt = vqgan_ckpt
        self.sequence_length = sequence_length
        self.audio_emb_model = audio_emb_model
        self.use_mcfl = use_mcfl
        self.mcfl_pooling_mode = mcfl_pooling_mode
        self.mcfl_gate_lambda = mcfl_gate_lambda  # MCFL v2-A gate parameter
        self.lambda_temp = lambda_temp  # Temporal smooth regularization weight
        self.mcfl_conservative = mcfl_conservative  # Conservative curriculum (alpha, freeze, lambda_temp) vs full MCFL
        self.use_baseline_imitation = use_baseline_imitation  # Online baseline imitation (placeholder)
        
        # Initialize MCFL if enabled
        if self.use_mcfl:
            from tacm import MCFL, AttnPool
            self.mcfl = MCFL(
                embed_dim=mcfl_embed_dim,
                num_heads=8,
                dropout=0.1  # Changed from 0.0 to 0.1 to prevent overfitting
            ).to(dist_util.dev())
            
            # Initialize attention pooling modules if using attention pooling
            if self.mcfl_pooling_mode == "attention":
                self.attn_pool_text = AttnPool(dim=mcfl_embed_dim).to(dist_util.dev())
                self.attn_pool_audio = AttnPool(dim=mcfl_embed_dim).to(dist_util.dev())
            else:
                self.attn_pool_text = None
                self.attn_pool_audio = None
        else:
            self.mcfl = None
            self.attn_pool_text = None
            self.attn_pool_audio = None

        self._mcfl_frozen = False  # Flag for curriculum: freeze MCFL in Stage 3

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        #for pn, p in self.mp_trainer.model.named_parameters():
        #    if 'temporal_conv' in pn:
        #        continue
        #    elif '2.transformer_blocks' in pn:
        #        continue
        #    else:
        #        p.requires_grad = False
               
        #params = filter(lambda p : p.requires_grad, self.mp_trainer.model.parameters())   
        #self.opt = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
             
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            
        # self.audioclip = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')
        # self.audioclip = self.audioclip.to(dist_util.dev())
        
        # self.wav2clip_model = wav2clip.get_model()
        # self.wav2clip_model = self.wav2clip_model.to(dist_util.dev())
        # for p in self.wav2clip_model.parameters():
        #     p.requires_grad = False
            
        checkpoint = th.load('saved_ckpts/BEATs_iter3_plus_AS20K.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.BEATs_model = BEATs(cfg)
        self.BEATs_model = self.BEATs_model.to('cpu')  # 固定在 CPU 上运行
        self.BEATs_model.load_state_dict(checkpoint['model'])
        self.BEATs_model.eval()
        
        self.processor = CLIPProcessor.from_pretrained("tacm/modules/cache/clip-vit-large-patch14",return_unused_kwargs=False)
        self.clipmodel = CLIPModel.from_pretrained("tacm/modules/cache/clip-vit-large-patch14").to(dist_util.dev())

        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        self.model.to(dist_util.dev())
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            for i, sample in enumerate(self.data):
                batch, cond = sample['video'], {}
                # ----get text----
                c_t = sample['text'].squeeze(1).to(dist_util.dev())
                
                # ----get image----
                image = batch[:,:,0]+0.5
                image_cat=None
                for j in range(image.shape[0]):
                    image_j = transformers.image_transforms.to_pil_image(image[j])
                    image_input = self.processor(images=image_j, return_tensors="pt", padding=True).to(dist_util.dev())
                    with th.no_grad():
                        image_features = self.clipmodel.get_image_features(image_input.pixel_values)

                    if image_cat is None:
                        image_cat = image_features.unsqueeze(0)
                    else:
                        image_cat = th.concat((image_cat, image_features.unsqueeze(0)), dim=0) #torch.Size([1, 1, 768])

                batch = rearrange(batch, "b c t h w -> (b t) c h w")
                c_ti = th.concat((c_t,image_cat), dim=1)
                #c_i = image_cat
                
                # ----get audio----
                if self.audio_emb_model == 'STFT':      
                    stft = sample['stft'] #torch.Size([1, 1, 16, 64, 16])
                else:
                    audio = sample['audio'].to(dist_util.dev()) #torch.Size([1, 16, 1600])
                
                # if self.audio_emb_model == 'audioclip':
                #     ((audio_embed, _, _), _), _ = self.audioclip(audio=audio.squeeze())
                #     c_temp = audio_embed.unsqueeze(0) #(1,16,1024)
                # elif self.audio_emb_model == 'wav2clip':
                #     audio_embed = th.from_numpy(wav2clip.embed_audio(audio.cpu().numpy().squeeze(), self.wav2clip_model)) #(16,512)
                #     c_temp = audio_embed.unsqueeze(1) #(1,16,512) #(16,1,512)
                if self.audio_emb_model == 'STFT':
                    c_temp = stft.squeeze(1)              
                elif self.audio_emb_model == 'beats':
                    audio = rearrange(audio, "b f g -> (b f) g")
                    # 将音频移到 CPU 上进行处理，因为 BEATs 模型在 CPU 上
                    c_temp = self.BEATs_model.extract_features(audio.cpu(), padding_mask=None)[0] #torch.Size([16, 8, 768])
                    # 处理完成后移回 GPU
                    c_temp = c_temp.to(dist_util.dev())

                # 🔧 修改 1：对 c_temp (BEATs audio tokens) 做跨帧 EMA 平滑
                # BEATs 每帧独立编码 → 条件帧间跳变 → flicker
                # EMA 低通滤波 → 平滑帧间过渡
                T = self.sequence_length  # 16
                BT, M, D = c_temp.shape
                B = BT // T
                c_temp_reshaped = c_temp.view(B, T, M, D)  # [B, T, 8, 768]
                
                alpha = 0.9  # EMA 系数，越大越平滑
                c_smoothed = c_temp_reshaped.clone()
                for t in range(1, T):
                    c_smoothed[:, t] = alpha * c_smoothed[:, t-1] + (1 - alpha) * c_temp_reshaped[:, t]
                
                c_temp = c_smoothed.view(BT, M, D)  # [B*T, 8, 768]

                # Use common condition builder (shared with sampling scripts)
                # MCFL v2-A: Temporal-only + Gated Residual
                c_ti, c_at = build_conditions(
                    c_t=c_t,
                    image_cat=image_cat,
                    c_temp=c_temp,
                    mcfl=self.mcfl,
                    use_mcfl=self.use_mcfl,
                    pooling_mode=self.mcfl_pooling_mode,
                    attn_pool_text=self.attn_pool_text,
                    attn_pool_audio=self.attn_pool_audio,
                    mcfl_gate_lambda=getattr(self, 'mcfl_gate_lambda', 0.2),  # Default 0.2 for v2-A
                )
                c_at = c_at.to(dist_util.dev())

                s = self.step + self.resume_step  # global step

                # =====================================================
                # MCFL: conservative curriculum only when mcfl_conservative=True
                # mcfl_conservative=True: alpha schedule + MCFL freeze at 8k (saves current MCFL behavior)
                # mcfl_conservative=False: full c_at, no freeze (for use with online baseline imitation)
                # =====================================================
                if self.use_mcfl and getattr(self, "mcfl_conservative", True):
                    # Alpha schedule (final)
                    if s < 4000:
                        alpha = 0.2 * (s / 4000.0)
                    elif s < 8000:
                        alpha = 0.2 + (0.7 - 0.2) * ((s - 4000.0) / 4000.0)
                    elif s < 10000:
                        alpha = 0.5
                    else:
                        # FINAL trade-off (0.2 for better FID/FVD/FFC)
                        alpha = 0.2
                    c_at = alpha * c_at + (1.0 - alpha) * c_at.detach()
                    logger.logkv_mean("alpha_audio", alpha)

                    # Curriculum: freeze MCFL in refinement stage (8k for more visual refinement)
                    if (
                        (not self._mcfl_frozen)
                        and (s >= 8000)
                        and self.mcfl is not None
                    ):
                        for p in self.mcfl.parameters():
                            p.requires_grad = False
                        self._mcfl_frozen = True
                        logger.log("MCFL frozen at step %d for refinement stage." % s)

                # batch = batch.to(dist_util.dev())
                self.run_step(batch, cond, c_ti, c_at)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, c, c_temp):
        self.forward_backward(batch, cond, c, c_temp)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, c, c_temp):
        self.mp_trainer.zero_grad()
        self.microbatch = self.batch_size * self.sequence_length
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                c,
                c_temp,
                model_kwargs=micro_cond,
            )

            # For baseline imitation: use same noise so x_t matches, and capture attn from MCFL forward
            if getattr(self, "use_baseline_imitation", False) and self.use_mcfl:
                micro_cond["return_attn"] = True
                noise = th.randn_like(micro)
                if last_batch or not self.use_ddp:
                    losses = compute_losses(noise=noise)
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses(noise=noise)
            else:
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            # ========== OPTIONAL: Temporal Δ-Attention Imitation (use_baseline_imitation=True) ==========
            # Imitate baseline's temporal attention dynamics to reduce flicker.
            # L_attn = ((Δ_mcfl - Δ_base) ** 2).mean(), Δ = attn[:, 1:, :] - attn[:, :-1, :] (沿 query/time 维 dim 1)
            # ========================================================================
            if getattr(self, "use_baseline_imitation", False) and self.use_mcfl:
                # attn_mcfl was captured from compute_losses forward (return_attn=True)
                attn_mcfl = _get_last_temporal_attn(self.ddp_model)

                # Build c_at_baseline (no MCFL)
                c_t = c[:, :-1]  # [B, N, D]
                image_cat = c[:, -1:]  # [B, 1, D]
                c_temp_raw = c_temp[:, :8, :]  # [B*T, 8, D] - audio tokens
                _, c_at_baseline = build_conditions(
                    c_t=c_t,
                    image_cat=image_cat,
                    c_temp=c_temp_raw,
                    use_mcfl=False,
                    mcfl=None,
                )
                c_at_baseline = c_at_baseline.to(dist_util.dev())

                x_t = self.diffusion.q_sample(micro, t, noise=noise)

                # Baseline forward (no grad)
                with th.no_grad():
                    _ = self.ddp_model(
                        x_t,
                        self.diffusion._scale_timesteps(t),
                        c,
                        c_at_baseline,
                        return_attn=True,
                    )
                attn_base = _get_last_temporal_attn(self.ddp_model)

                if attn_base is not None and attn_mcfl is not None:
                    # Δ must be along query/time dim (dim 1), NOT context (dim 2)
                    # attn: [(b*h*w*heads), F, M]  F=time frames, M=context tokens
                    assert attn_mcfl.shape == attn_base.shape, (
                        f"attn shape mismatch: mcfl {attn_mcfl.shape} vs base {attn_base.shape}"
                    )
                    # F=query/time dim (通常 16), M=context dim
                    if attn_mcfl.shape[1] != 16:
                        logger.log(
                            f"WARN: attn time dim F={attn_mcfl.shape[1]} (expected 16). "
                            "Check sequence_length / token align."
                        )
                    delta_mcfl = attn_mcfl[:, 1:, :] - attn_mcfl[:, :-1, :]   # [*, F-1, M]
                    delta_base = attn_base[:, 1:, :] - attn_base[:, :-1, :]
                    loss_attn = ((delta_mcfl - delta_base.detach()) ** 2).mean()

                    # 幅度约束：限制 Δ 抖得有多狠，不改方向
                    loss_attn_energy = (delta_mcfl ** 2).mean()
                    beta = getattr(self, "attn_energy_beta", 0.01)  # 比 imitation 小一数量级

                    s = self.step + self.resume_step
                    # 更激进衰减：FVD 已受益，再强 imitation 会限制 MCFL 自由
                    if s < 3000:
                        lambda_attn = 0.1
                    elif s < 6000:
                        lambda_attn = 0.03
                    else:
                        lambda_attn = 0.005

                    loss = loss + lambda_attn * loss_attn + beta * loss_attn_energy
                    logger.logkv_mean("loss_attn", loss_attn.item())
                    logger.logkv_mean("loss_attn_energy", loss_attn_energy.item())
                    logger.logkv_mean("lambda_attn", lambda_attn)
            # ========================================================================

            # ========== OPTIONAL: Temporal Smooth (only when use_mcfl + mcfl_conservative) ==========
            # mcfl_conservative=True: lambda_temp curriculum (0 -> 0.02 over 0-10k steps).
            # mcfl_conservative=False: skip (for use with online baseline imitation).
            # ========================================================================
            if (
                self.use_mcfl
                and getattr(self, "mcfl_conservative", True)
                and getattr(self, "lambda_temp", 0.0) > 0
            ):
                T = self.sequence_length
                BT, C, H, W = micro.shape
                assert BT % T == 0, f"micro batch {BT} not divisible by sequence_length {T}"
                B = BT // T

                # reshape back to [B, T, C, H, W]
                micro_seq = micro.view(B, T, C, H, W)

                # first-order temporal difference
                diff = micro_seq[:, 1:] - micro_seq[:, :-1]   # [B, T-1, C, H, W]
                loss_temp = (diff * diff).mean()

                # =====================================================
                # Final temporal smooth curriculum (20k steps)
                # =====================================================
                s = self.step + self.resume_step

                if s < 4000:
                    # Stage 1: no temporal constraint (learn motion freely)
                    lambda_temp_now = 0.0

                elif s < 8000:
                    # Stage 2: gently suppress early jitter
                    lambda_temp_now = 0.005 * ((s - 4000.0) / 4000.0)

                elif s < 10000:
                    # Stage 3: stabilize audio-driven motion
                    lambda_temp_now = 0.005 + (0.02 - 0.005) * ((s - 8000.0) / 2000.0)

                else:
                    # Stage 4: FINAL trade-off (do NOT increase further)
                    lambda_temp_now = 0.02

                # add temporal smooth loss
                loss = loss + lambda_temp_now * loss_temp

                # optional logging
                logger.logkv_mean("lambda_temp", lambda_temp_now)
                logger.logkv_mean("loss_temp", loss_temp.item())
            # ========================================================================

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.save_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
