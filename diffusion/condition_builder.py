# Copyright (c) Meta Platforms, Inc. All Rights Reserved

"""
Common condition building logic for training and sampling.
This module provides a unified interface for building conditions with optional MCFL fusion.
"""

import torch as th
from einops import rearrange, repeat
from typing import Optional, Tuple, Literal


def build_conditions(
    c_t: th.Tensor,  # [B, N, D] - text condition
    image_cat: th.Tensor,  # [B, 1, D] - image condition
    c_temp: th.Tensor,  # [B, T, D] or [B*T, T_seq, D] - audio condition
    mcfl=None,  # Optional MCFL module
    use_mcfl: bool = False,
    pooling_mode: Literal["mean", "attention"] = "mean",  # Pooling strategy for sequences
    attn_pool_text=None,  # Optional AttnPool for text (if pooling_mode="attention")
    attn_pool_audio=None,  # Optional AttnPool for audio (if pooling_mode="attention")
    mcfl_gate_lambda: float = 0.1,  # Gate parameter for MCFL v2-A (0.1 降低 TC_FLICKER，原 0.2)
    mcfl_norm_modality: bool = True,  # 跨分布鲁棒：送入 MCFL 前对 image/audio 做 L2 归一化，减弱幅值差异
    mcfl_gate_adaptive: bool = True,  # 异常输入时 gate→0：用音频 embedding 范数做置信度
    mcfl_gate_norm_low: float = 7.2,  # 置信度三角形下界（统一标定：覆盖 drums/URMP/landscape p5）
    mcfl_gate_norm_high: float = 10.0,  # 置信度三角形上界（统一标定：覆盖三数据集 p95）
    mcfl_gate_time_smooth: bool = True,  # 对 gate 做时间 EMA，减轻 flicker
    mcfl_gate_ema: float = 0.9,  # gate_t = ema*gate_{t-1} + (1-ema)*gate_raw_t，0.9 更强平滑
    mcfl_gate_use_zscore: bool = False,  # True: 用 z=(norm-mu)/sigma 标定，跨数据集更稳
    mcfl_gate_norm_mu: float = 8.4,  # z-score 均值（来自训练/多数据集统计）
    mcfl_gate_norm_sigma: float = 0.5,  # z-score 标准差
    mcfl_gate_z_low: float = -1.5,  # z 下界，conf 线性映射
    mcfl_gate_z_high: float = 1.5,  # z 上界
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Build conditions for diffusion model with optional MCFL fusion.
    
    MCFL v2-A: Temporal-only + Gated Residual
    - Spatial condition (c_ti): Keep baseline (text + image) to protect spatial quality
    - Temporal condition (c_at): Use gated fused condition for AV alignment
    
    Args:
        c_t: [B, N, D] - text condition sequence
        image_cat: [B, 1, D] - image condition (single token)
        c_temp: [B, T, D] or [B*T, T_seq, D] - audio condition sequence
        mcfl: Optional MCFL module for condition fusion
        use_mcfl: Whether to use MCFL fusion
        pooling_mode: Pooling strategy - "mean" (average) or "attention" (learned attention weights)
        attn_pool_text: Optional AttnPool module for text pooling (required if pooling_mode="attention")
        attn_pool_audio: Optional AttnPool module for audio pooling (required if pooling_mode="attention")
        mcfl_gate_lambda: Gate parameter for gated residual (default 0.1, range [0, 1])
    
    Returns:
        c_ti: [B, N+1, D] - text+image condition (batch dimension NOT expanded)
        c_at: [B*T, T_seq+N, D] or [B*T, T_seq+T_seq, D] (with MCFL) - audio+text condition
    """
    # MCFL v2-A: Spatial condition (c_ti) keeps baseline to protect spatial quality
    # c_t: [B, N, D], image_cat: [B, 1, D] → c_ti: [B, N+1, D]
    c_ti = th.concat((c_t, image_cat), dim=1)  # [B, N+1, D] - BASELINE (unchanged)
    
    # For c_at: need to expand c_t to match c_temp's batch dimension
    # c_temp might be [B*T, T_seq, D] where T is sequence_length (e.g., 16)
    # Calculate expansion factor based on c_temp's batch dimension
    B_orig = c_t.shape[0]  # Original batch size B
    B_temp = c_temp.shape[0]  # c_temp's batch dimension (might be B*T)
    expansion_factor = B_temp // B_orig if B_temp > B_orig else 1
    
    # MCFL branch: condition fusion
    if use_mcfl and mcfl is not None:
        # Extract single embeddings from sequences (pooling) - use original batch B
        if pooling_mode == "attention":
            # Attention pooling: let model learn which tokens/timesteps are important
            if attn_pool_text is None or attn_pool_audio is None:
                raise ValueError(
                    "attn_pool_text and attn_pool_audio must be provided when pooling_mode='attention'"
                )
            c_text_single = attn_pool_text(c_t)  # [B, D] - attention-weighted text
            c_image_single = image_cat.squeeze(1)  # [B, D] - image (already single)
            
            # For audio, handle batch dimension correctly
            if expansion_factor > 1:
                # c_temp is [B*T, T_seq, D], reshape for pooling to get [B, D]
                c_audio_reshaped = c_temp.view(B_orig, expansion_factor, c_temp.shape[1], c_temp.shape[2])  # [B, T, T_seq, D]
                # Pool over temporal dimension first, then batch expansion
                B_audio, T_exp, T_seq, D_audio = c_audio_reshaped.shape
                c_audio_reshaped_flat = c_audio_reshaped.view(B_audio * T_exp, T_seq, D_audio)  # [B*T, T_seq, D]
                c_audio_pooled = attn_pool_audio(c_audio_reshaped_flat)  # [B*T, D]
                c_audio_single = c_audio_pooled.view(B_audio, T_exp, D_audio).mean(dim=1)  # [B, D]
            else:
                c_audio_single = attn_pool_audio(c_temp)  # [B, D] - attention-weighted audio
        else:
            # Mean pooling: simple average (baseline)
            c_text_single = c_t.mean(dim=1)  # [B, D] - text sequence average
            c_image_single = image_cat.squeeze(1)  # [B, D] - image (already single)
            
            # For audio, handle batch dimension correctly
            if expansion_factor > 1:
                # c_temp is [B*T, T_seq, D], reshape for pooling to get [B, D]
                c_audio_reshaped = c_temp.view(B_orig, expansion_factor, c_temp.shape[1], c_temp.shape[2])  # [B, T, T_seq, D]
                c_audio_single = c_audio_reshaped.mean(dim=(1, 2))  # [B, D] - average over batch expansion and sequence
            else:
                c_audio_single = c_temp.mean(dim=1)  # [B, D] - audio sequence average
        
        # 置信度自适应 gate：异常输入时 gate→0（在 L2 norm 前用原始范数）
        def _conf_from_norm(norm_1d):
            if mcfl_gate_use_zscore:
                z = (norm_1d - mcfl_gate_norm_mu) / (mcfl_gate_norm_sigma + 1e-6)
                return th.clamp(
                    (z - mcfl_gate_z_low) / (mcfl_gate_z_high - mcfl_gate_z_low + 1e-6), 0.0, 1.0
                )
            t_low, t_high = mcfl_gate_norm_low, mcfl_gate_norm_high
            t_mid = (t_low + t_high) * 0.5
            span = (t_high - t_low) + 1e-8
            return th.clamp(1.0 - 2.0 * th.abs(norm_1d - t_mid) / span, 0.0, 1.0)
        
        if mcfl_gate_adaptive and expansion_factor > 1 and mcfl_gate_time_smooth:
            # 每帧 gate + 时间 EMA 平滑，减轻 flicker
            c_audio_per_frame = c_temp.mean(dim=1)  # [B*T, D]
            raw_norm_frame = c_audio_per_frame.norm(dim=-1)  # [B*T]
            conf_frame = _conf_from_norm(raw_norm_frame)  # [B*T]
            conf_2d = conf_frame.view(B_orig, expansion_factor)  # [B, T]
            conf_smooth = conf_2d.clone()
            for t in range(1, expansion_factor):
                conf_smooth[:, t] = mcfl_gate_ema * conf_smooth[:, t - 1] + (1.0 - mcfl_gate_ema) * conf_2d[:, t]
            gate_per_frame = (mcfl_gate_lambda * conf_smooth.view(-1)).unsqueeze(1)  # [B*T, 1]
        elif mcfl_gate_adaptive:
            raw_norm_single = c_audio_single.norm(dim=-1)  # [B]
            conf = _conf_from_norm(raw_norm_single).unsqueeze(1)  # [B, 1]
            gate_per_frame = mcfl_gate_lambda * conf  # [B, 1]，后面 expand 到 [B*T, 1]
        else:
            gate_per_frame = None  # 用固定 lam
        
        # 跨分布鲁棒：对 image/audio 做 L2 归一化，使 MCFL 更依赖方向而非幅值
        if mcfl_norm_modality:
            c_image_single = c_image_single / (c_image_single.norm(dim=-1, keepdim=True) + 1e-6)
            c_audio_single = c_audio_single / (c_audio_single.norm(dim=-1, keepdim=True) + 1e-6)
        
        # MCFL v2-A: Gated Residual
        c_hat = mcfl(
            c_text_single,
            c_image_single.detach(),
            c_audio_single.detach()
        )  # [B, D] - MCFL output
        
        delta = c_hat - c_text_single  # [B, D]
        delta_norm = delta.norm(dim=-1, keepdim=True) + 1e-6
        delta = delta / delta_norm
        delta_scale = 0.2
        delta = delta * delta_scale  # [B, D]
        
        if gate_per_frame is not None:
            if gate_per_frame.dim() == 2 and gate_per_frame.shape[0] == B_orig and gate_per_frame.shape[1] == 1:
                # per-sample gate，expand 到 [B*T, 1]
                gate_expanded = repeat(gate_per_frame, "b 1 -> (b f) 1", f=expansion_factor)
            else:
                gate_expanded = gate_per_frame  # 已是 [B*T, 1]
            c_fused_expanded = repeat(c_text_single, "b d -> (b f) d", f=expansion_factor) + gate_expanded * repeat(delta, "b d -> (b f) d", f=expansion_factor)
        else:
            lam = mcfl_gate_lambda
            c_fused = c_text_single + lam * delta  # [B, D]
            c_fused_expanded = repeat(c_fused, "b d -> (b f) d", f=expansion_factor)
        
        # Expand to sequence format [B*T, T_seq, D]
        c_fused_expanded_seq = c_fused_expanded.unsqueeze(1).repeat(1, c_temp.shape[1], 1)  # [B*T, T_seq, D]
        c_at = th.concat((c_temp, c_fused_expanded_seq), dim=1)  # [B*T, T_seq+T_seq, D]
    else:
        # Original logic (baseline): expand c_t to match c_temp's batch dimension for c_at
        # c_t: [B, N, D] → [B*T, N, D] to match c_temp: [B*T, T_seq, D]
        c_t_expanded = repeat(c_t, "b n d -> (b f) n d", f=expansion_factor)  # [B*T, N, D]
        c_at = th.concat((c_temp, c_t_expanded), dim=1)  # [B*T, T_seq+N, D]
    
    return c_ti, c_at
