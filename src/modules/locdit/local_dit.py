import torch
from modules.minicpm4 import MiniCPMModel, MiniCPM4Config
import torch.nn as nn
import math
from typing import Optional, List, Tuple
from torch.nn import functional as F

# locdit.py
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class VIBELocDiT(nn.Module):

    def __init__(
        self,
        config: MiniCPM4Config,
        in_channels: int = 64,
        num_lm_layers: int = 0,
        lm_hidden_size: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config

        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)

        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )

        assert config.vocab_size == 0, "vocab_size must be 0 for local DiT"
        self.decoder = MiniCPMModel(config)

        # Depth-wise semantic routing
        # One learnable weight vector per DiT block (zero-init → uniform softmax at start)
        if num_lm_layers > 0 and lm_hidden_size > 0:
            self.routing_weights = nn.ParameterList([
                nn.Parameter(torch.zeros(num_lm_layers))
                for _ in range(config.num_hidden_layers)
            ])
            self.routing_proj = nn.Linear(lm_hidden_size, config.hidden_size, bias=False)
        else:
            self.routing_weights = None
            self.routing_proj = None


    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
        all_lm_hiddens: Optional[List[torch.Tensor]] = None,
    ):
        # --- depth-wise routing: compute per-block conditioning vectors ---
        per_layer_cond = None
        if self.routing_weights is not None and all_lm_hiddens is not None:
            dtype = x.dtype
            stacked = torch.stack(all_lm_hiddens, dim=0).to(dtype)  # (L, N, D_lm)
            stacked = F.layer_norm(stacked, (stacked.shape[-1],))
            # K routing weight vectors, each over L LLM layers
            weight_stack = torch.stack(
                [F.softmax(w.to(dtype), dim=0) for w in self.routing_weights], dim=0
            )  # (K, L)
            fused = torch.einsum("kl,lnd->knd", weight_stack, stacked)  # (K, N, D_lm)
            del stacked, weight_stack
            per_layer_cond = [
                self.routing_proj(fused[k])
                for k in range(fused.shape[0])
            ]  # list of K tensors, each (N, D_dit)

        x = self.in_proj(x.transpose(1, 2).contiguous())

        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)

        t = self.time_embeddings(t).to(x.dtype)
        t = self.time_mlp(t)
        dt = self.time_embeddings(dt).to(x.dtype)
        dt = self.delta_time_mlp(dt)
        t = t + dt

        x = torch.cat([(mu + t).unsqueeze(1), cond, x], dim=1)
        hidden, _ = self.decoder(x, is_causal=False, per_layer_cond=per_layer_cond)
        hidden = hidden[:, prefix + 1 :, :]
        hidden = self.out_proj(hidden)

        return hidden.transpose(1, 2).contiguous()
