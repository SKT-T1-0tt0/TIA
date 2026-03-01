# Copyright (c) Meta Platforms, Inc. All Rights Reserved

try:
    from .lpips import LPIPS
except ImportError:
    # LPIPS module is optional, only needed for VQGAN training
    LPIPS = None

from .codebook import Codebook

try:
    from .audioclip import AudioCLIP
except ImportError:
    # AudioCLIP module is optional, only needed when audio_emb_model='audioclip'
    AudioCLIP = None
