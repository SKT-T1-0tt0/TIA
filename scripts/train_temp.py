"""
Generate a large batch of video samples from a model.
"""
import sys
sys.path.append('/data/workspace/TACM')
import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import pytorch_lightning as pl

from diffusion.resample import create_named_schedule_sampler
from diffusion import dist_util, logger
from diffusion.tacm_script_temp_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.tacm_train_temp_util import TrainLoop
from diffusion.dist_util import save_video_grid
from tacm import VideoData
from einops import rearrange, repeat

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    new_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # load original model parameters
    original_model_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    # get new model parameters dictionary
    new_model_dict = new_model.state_dict()
    # delete keys in original parameters which are not same as new_model
    pretrained_dict = {k: v for k, v in original_model_dict.items() if k in new_model_dict}
    # keep original parameters (spatial layers) fixed
    for v in original_model_dict.values():
        v.requires_grad = False
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)

    #print(new_model)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.train_dataloader()

    logger.log("training...")
    TrainLoop(
        model=new_model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir,
        vqgan_ckpt=args.vqgan_ckpt,
        sequence_length=args.sequence_length,
        audio_emb_model=args.audio_emb_model,
        use_mcfl=getattr(args, 'use_mcfl', False),
        mcfl_embed_dim=getattr(args, 'mcfl_embed_dim', 768),
        mcfl_pooling_mode=getattr(args, 'mcfl_pooling_mode', 'mean'),  # "mean" or "attention"
        mcfl_gate_lambda=getattr(args, 'mcfl_gate_lambda', 0.1),  # MCFL v2-A gate (0.1 降低 TC_FLICKER)
        lambda_temp=getattr(args, 'lambda_temp', 0.0),  # Temporal smooth regularization weight (default 0.0 = disabled)
        mcfl_conservative=getattr(args, 'mcfl_conservative', True),  # True = current MCFL curriculum; False = full MCFL (for baseline imitation)
        use_baseline_imitation=getattr(args, 'use_baseline_imitation', False),  # Online baseline imitation (implement when needed)
        mcfl_norm_modality=getattr(args, 'mcfl_norm_modality', True),
        audio_response=getattr(args, 'audio_response', 'tanh'),
        audio_random_gain=getattr(args, 'audio_random_gain', True),
        audio_gain_range=(getattr(args, 'audio_gain_low', 0.25), getattr(args, 'audio_gain_high', 4.0)),
        audio_random_response_strength=getattr(args, 'audio_random_response_strength', True),
        modality_dropout_prob=getattr(args, 'modality_dropout_prob', 0.2),
        mcfl_gate_adaptive=getattr(args, 'mcfl_gate_adaptive', True),
        mcfl_gate_norm_low=getattr(args, 'mcfl_gate_norm_low', 7.2),
        mcfl_gate_norm_high=getattr(args, 'mcfl_gate_norm_high', 10.0),
        mcfl_gate_time_smooth=getattr(args, 'mcfl_gate_time_smooth', True),
        mcfl_gate_ema=getattr(args, 'mcfl_gate_ema', 0.9),
        mcfl_gate_use_zscore=getattr(args, 'mcfl_gate_use_zscore', False),
        mcfl_gate_norm_mu=getattr(args, 'mcfl_gate_norm_mu', 8.4),
        mcfl_gate_norm_sigma=getattr(args, 'mcfl_gate_norm_sigma', 0.5),
        mcfl_gate_z_low=getattr(args, 'mcfl_gate_z_low', -1.5),
        mcfl_gate_z_high=getattr(args, 'mcfl_gate_z_high', 1.5),
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # MCFL switches (all off = baseline; pass --use_mcfl to enable MCFL)
        use_mcfl=False,
        mcfl_embed_dim=768,
        mcfl_pooling_mode="mean",
        mcfl_gate_lambda=0.1,
        lambda_temp=0.0,
        mcfl_conservative=True,  # True = alpha + freeze + lambda_temp curriculum (current MCFL); False = full MCFL for baseline imitation
        use_baseline_imitation=False,  # Online baseline output imitation (placeholder; implement when needed)
        mcfl_norm_modality=True,
        audio_response='tanh',
        audio_random_gain=True,
        audio_gain_low=0.25,
        audio_gain_high=4.0,
        audio_random_response_strength=True,
        modality_dropout_prob=0.2,
        mcfl_gate_adaptive=True,
        mcfl_gate_norm_low=7.2,
        mcfl_gate_norm_high=10.0,
        mcfl_gate_time_smooth=True,
        mcfl_gate_ema=0.9,
        mcfl_gate_use_zscore=False,
        mcfl_gate_norm_mu=8.4,
        mcfl_gate_norm_sigma=0.5,
        mcfl_gate_z_low=-1.5,
        mcfl_gate_z_high=1.5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--vqgan_ckpt', type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
