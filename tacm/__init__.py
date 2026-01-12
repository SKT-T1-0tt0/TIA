# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from .data import VideoData
from .download import load_vqgan, download
from .vqgan import VQGAN

# Export AudioCLIP from modules (may be None if esresnet is missing)
try:
    from .modules import AudioCLIP
except ImportError:
    AudioCLIP = None