import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPModel
from einops import rearrange
import sys


class VideoEncoder(nn.Module):
    def __init__(self, config: CLIPConfig = CLIPConfig()):
        super().__init__()
        self.config = config
        self.encoder = CLIPModel(config)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")  # [B*T, C, H, W]

        encoder_dtype = next(self.encoder.parameters()).dtype
        with torch.no_grad():
            embedding = self.encoder.get_image_features(pixel_values=x.to(encoder_dtype)).pooler_output  # [B*T, D]

        print("====Video Encoder Embedding Shape:", embedding.shape)
        return rearrange(embedding, "(b t) d -> b t d", b=B, t=T)  # [B, T, D]
    
    def from_pretrained(self, pretrained_model_name_or_path: str):
        self.encoder = CLIPModel.from_pretrained(pretrained_model_name_or_path)
        
