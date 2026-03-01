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
        
        # MCFL v2-A: Gated Residual
        # Core formula: c_fused = c_text + λ * (MCFL(...) - c_text)
        # This makes fused condition an "increment" rather than full replacement
        
        # 🔪 第二刀：对 MCFL 梯度做"半截断"（image/audio detach，text 保留梯度）
        c_hat = mcfl(
            c_text_single,
            c_image_single.detach(),
            c_audio_single.detach()
        )  # [B, D] - MCFL output
        
        # 🔪 第一刀：对 delta 做 norm + scale（保留方向，抹掉幅值不稳定性）
        delta = c_hat - c_text_single  # [B, D]
        delta_norm = delta.norm(dim=-1, keepdim=True) + 1e-6  # [B, 1]
        delta = delta / delta_norm  # 归一化方向
        delta_scale = 0.2  # 固定幅度（可调：0.1 ~ 0.3）
        delta = delta * delta_scale  # [B, D]
        
        lam = mcfl_gate_lambda  # Gate parameter (now 0.1, was 0.2)
        c_fused = c_text_single + lam * delta  # [B, D] - Gated residual
        
        # MCFL v2-A: Temporal-only injection
        # c_ti remains baseline (already set above) - protects spatial quality
        # Only c_at uses fused condition for temporal/AV alignment
        
        # For c_at: expand fused condition to match c_temp's batch dimension [B*T, ...]
        c_fused_expanded = repeat(c_fused, "b d -> (b f) d", f=expansion_factor)  # [B*T, D]
        
        # Expand fused to sequence format: [B*T, T_seq, D]
        # Match the sequence length of c_temp for concatenation
        c_fused_expanded_seq = c_fused_expanded.unsqueeze(1).repeat(1, c_temp.shape[1], 1)  # [B*T, T_seq, D]
        
        # Temporal condition: concat audio + fused (replaces original text expansion)
        c_at = th.concat((c_temp, c_fused_expanded_seq), dim=1)  # [B*T, T_seq+T_seq, D]
    else:
        # Original logic (baseline): expand c_t to match c_temp's batch dimension for c_at
        # c_t: [B, N, D] → [B*T, N, D] to match c_temp: [B*T, T_seq, D]
        c_t_expanded = repeat(c_t, "b n d -> (b f) n d", f=expansion_factor)  # [B*T, N, D]
        c_at = th.concat((c_temp, c_t_expanded), dim=1)  # [B*T, T_seq+N, D]
    
    return c_ti, c_at
