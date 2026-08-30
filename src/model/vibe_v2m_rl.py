from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.io import VideoReader
from transformers import CLIPImageProcessor

from modules.video_encoder import VideoEncoder

from .utils import get_dtype
from .vibe_ttm import LoRAConfig, VIBEConfig, VIBETextToMusic
from .vibe_v2m import (
    inject_video_embeddings,
    prep_video_frames,
    sample_video_frames,
)

__all__ = ["VIBEVideo2MusicRL", "LoRAConfig", "VIBEConfig"]


class VIBEVideo2MusicRL(VIBETextToMusic):

    def __init__(self, *args, video_embed_name: str = "openai/clip-vit-base-patch32", **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        self.n_video_frames = getattr(config, "n_video_frames", 8)
        self.clip_processor = CLIPImageProcessor.from_pretrained(video_embed_name)
        self.video_encoder = VideoEncoder(config.video_encoder_config)
        self.video_to_lm_proj = nn.Linear(
            config.video_encoder_config.projection_dim,
            config.lm_config.hidden_size,
        )
        # Frozen for RL, matching stage 4: only the LoRA adapters and the
        # projection receive gradients.
        self.video_encoder.requires_grad_(False)

    # ------------------------------------------------------- hooks (overrides)

    def _prepare_video(self, video_path: str):
        if not video_path:
            return None, 0, None
        frames, duration = sample_video_frames(VideoReader(video_path), self.n_video_frames)
        video = prep_video_frames(self.clip_processor, frames).unsqueeze(0).to(self.device)
        return video, video.shape[1], duration

    def _inject_video(self, combined_embed, text_mask, video):
        if video is None:
            return combined_embed
        video_embed = self.video_encoder(video)              # [B, T_v, D_v]
        video_embed = self.video_to_lm_proj(video_embed)     # [B, T_v, H]
        return inject_video_embeddings(combined_embed, text_mask, video_embed)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_local(
        cls,
        path: str,
        baselm_path: str,
        audiovae_path: str,
        video_embed_name: str = "openai/clip-vit-base-patch32",
        **kwargs,
    ) -> "VIBEVideo2MusicRL":
        model = super().from_local(
            path, baselm_path=baselm_path, audiovae_path=audiovae_path, **kwargs
        )
        if not _has_encoder_weights(model.video_encoder):
            model.video_encoder.from_pretrained(video_embed_name)
        model.video_encoder = model.video_encoder.to(get_dtype(model.config.dtype))
        model.video_encoder.requires_grad_(False)
        return model


def _has_encoder_weights(module: nn.Module) -> bool:
    for p in module.parameters():
        if p.numel() and torch.any(p != 0):
            return True
    return False
