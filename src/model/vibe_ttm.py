# Copyright 2025 OpenBMB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from typing import Tuple, Union, Generator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import warnings
from einops import rearrange
from pydantic import BaseModel, ConfigDict
import numpy as np

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from transformers import AutoModelForCausalLM, CLIPConfig

from modules.audiovae import AudioVAE, AudioVAEConfig
from modules.layers import ScalarQuantizationLayer
from modules.layers.lora import apply_lora_to_named_linear_modules
from modules.locdit import CfmConfig, UnifiedCFM, VIBELocDiT
from modules.locenc import VIBELocEnc
from modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens

def log_shapes_and_stats(tensor: torch.Tensor, name: str):
    # Suppressed for cleaner training output
    pass
    # if tensor.numel() > 0 and tensor.dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
    # else:


def log_shapes_and_stats_for_all(
    **args
):
    for key, tensor in args.items():
        if tensor is not None:
            log_shapes_and_stats(tensor, key)


class VIBEEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VIBEDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None

    cfm_config: CfmConfig


class VIBEConfig(BaseModel):
    # CLIPConfig is a HF transformers object, not a pydantic model.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    rite_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9

    # Video-conditioning fields. Unused by this text-only policy, but declared
    # so a stage-4 V2M config.json round-trips losslessly and
    # VIBEVideo2MusicRL can build its encoder from the same config object.
    n_video_frames: int = 8
    video_encoder_config: CLIPConfig = CLIPConfig()

    encoder_config: VIBEEncoderConfig
    dit_config: VIBEDitConfig
    audio_vae_config: Optional[AudioVAEConfig] = None

    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"
    dit_mean_mode: bool = False


class LoRAConfig(BaseModel):
    enable_lm: bool = False        # Apply LoRA to multimodal_semantic_lm + rite
    enable_dit: bool = False       # Apply LoRA to VIBELocDiT
    enable_proj: bool = False      # Apply LoRA to projection Linear layers

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    # Target linear layer names for LM & DiT (matched by attribute name)
    target_modules_lm: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    target_modules_dit: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Projection layer attribute names to find on VIBETextToMusic
    target_proj_modules: list[str] = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]


VIBEConfig.model_rebuild()


class VIBETextToMusic(nn.Module):
    def __init__(
        self,
        config: VIBEConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
        lora_config: LoRAConfig = None,
        patch_size: int = 4,
        use_stop_loss: bool = True,
    ):
        print("patch_size:", patch_size)
        print("use_stop_loss:", use_stop_loss)

        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = patch_size
        self.device = config.device
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}", file=sys.stderr)

        # Text-Semantic LM
        self.multimodal_semantic_lm = MiniCPMModel(config.lm_config)
        self.multimodal_semantic_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102

        # Residual Acoustic LM
        rite_config = config.lm_config.model_copy(deep=True)
        rite_config.num_hidden_layers = config.rite_num_layers
        rite_config.vocab_size = 0
        self.rite = MiniCPMModel(rite_config)
        self.rite.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VIBELocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VIBELocDiT(
                decoder_config,
                in_channels=config.feat_dim,
                num_lm_layers=config.lm_config.num_hidden_layers,
                lm_hidden_size=config.lm_config.hidden_size,
            ),
            mean_mode=config.dit_mean_mode,
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        if use_stop_loss:
            self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
            self.stop_actn = nn.SiLU()
            self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
            self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.downsampling_ratio  # SongBloom music VAE
        self.sample_rate = audio_vae.sample_rate

        self.use_stop_loss = use_stop_loss
        if self.lora_config is not None:
            self._apply_lora()


    def _apply_lora(self):
        cfg = self.lora_config
        lora_kwargs = dict(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)

        # LM: multimodal_semantic_lm + rite
        if cfg.enable_lm:
            for lm in [self.multimodal_semantic_lm, self.rite]:
                apply_lora_to_named_linear_modules(
                    lm, target_submodule_names=cfg.target_modules_lm, **lora_kwargs
                )

        # DiT: feat_decoder.estimator
        if cfg.enable_dit:
            apply_lora_to_named_linear_modules(
                self.feat_decoder.estimator, target_submodule_names=cfg.target_modules_dit, **lora_kwargs
            )

        # 投影层
        if cfg.enable_proj:
            from modules.layers.lora import LoRALinear
            for attr_name in cfg.target_proj_modules:
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Linear):
                    setattr(self, attr_name, LoRALinear(base=module, **lora_kwargs))

    def optimize(self, disable: bool = False):
        if disable:
            return self
        try:
            if self.device != "cuda":
                raise ValueError("VIBETextToMusic can only be optimized on CUDA device")
            try:
                import triton
            except:
                raise ValueError("triton is not installed")
            self.multimodal_semantic_lm.forward_step = torch.compile(self.multimodal_semantic_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.rite.forward_step = torch.compile(self.rite.forward_step, mode="reduce-overhead", fullgraph=True)
            self.feat_encoder = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"Warning: torch.compile disabled - {e}", file=sys.stderr)
        return self

    def compute_dit_conditioning(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        debug: bool = True,
        video: Optional[torch.Tensor] = None,
    ) -> dict:
        def _d(name, t):
            if debug:
                if isinstance(t, torch.Tensor):
                    print(
                        f"  [compute_dit_conditioning] {name}: "
                        f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                        f"min={t.float().min().item():.4e} max={t.float().max().item():.4e} "
                        f"mean={t.float().mean().item():.4e}"
                        f"num_zeros={torch.sum(t == 0).item()}"
                    )
                else:
                    print(f"  [compute_dit_conditioning] {name}: {t}")

        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())

        #     "feat_encoder.in_proj.weight.dtype",
        #     next(self.feat_encoder.parameters()).dtype,
        # )

        B, T, P, D = audio_feats.shape
        assert text_tokens.dim() == 2 and text_tokens.shape[0] == B, (
            f"text_tokens.shape={tuple(text_tokens.shape)}, expected (B={B}, T)"
        )
        assert text_mask.shape == text_tokens.shape, (
            f"text_mask.shape={tuple(text_mask.shape)} vs text_tokens.shape={tuple(text_tokens.shape)}"
        )
        assert audio_mask.shape == text_tokens.shape, (
            f"audio_mask.shape={tuple(audio_mask.shape)} vs text_tokens.shape={tuple(text_tokens.shape)}"
        )

        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)

        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.multimodal_semantic_lm.embed_tokens(text_tokens) * scale_emb

        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed
        combined_embed = self._inject_video(combined_embed, text_mask, video)

        enc_outputs, _, all_lm_layer_hiddens = self.multimodal_semantic_lm(
            inputs_embeds=combined_embed, is_causal=True, output_hidden_states=True
        )
        enc_outputs = enc_outputs.to(self._dtype())

        audio_part_before_fsq = enc_outputs * audio_mask.unsqueeze(-1)
        audio_part = self.fsq_layer(audio_part_before_fsq)
        text_part = enc_outputs * text_mask.unsqueeze(-1)
        enc_outputs = audio_part + text_part

        lm_hidden = torch.cat(
            (torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1
        )

        residual_inputs = enc_outputs + audio_mask.unsqueeze(-1) * feat_embed
        residual_outputs, _ = self.rite(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())

        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Depth-wise semantic routing: shift each layer's hidden states by 1 (causal),
        # flatten to (B*T, lm_hidden) so each (b,t) patch gets routing from the previous
        # sequence position's multimodal_semantic_lm hidden at that layer.
        all_lm_hiddens_for_dit = [
            rearrange(
                torch.cat((torch.zeros_like(h[:, 0:1, :]), h[:, :-1, :]), dim=1),
                "b t c -> (b t) c",
            ).to(self._dtype())
            for h in all_lm_layer_hiddens
        ]

        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        # Transpose to LocDiT's (N, C, T) layout.
        feat_gt_transposed = feat_gt.transpose(1, 2).contiguous()
        feat_cond_transposed = feat_cond.transpose(1, 2).contiguous()
        loss_seq_mask_transposed = loss_seq_mask.transpose(1, 2).contiguous()


        # --- Invariants ----------------------------------------------------
        BT = B * T
        assert dit_hidden.shape[0] == BT, (
            f"dit_hidden.shape[0]={dit_hidden.shape[0]} != B*T={BT}"
        )
        assert feat_gt_transposed.shape[0] == BT, (
            f"feat_gt.shape[0]={feat_gt_transposed.shape[0]} != B*T={BT}"
        )
        assert feat_gt_transposed.shape == feat_cond_transposed.shape, (
            f"feat_gt {feat_gt_transposed.shape} vs feat_cond {feat_cond_transposed.shape}"
        )
        assert feat_gt_transposed.shape[2] == P, (
            f"feat_gt last dim {feat_gt_transposed.shape[2]} != patch_size {P}"
        )
        assert loss_seq_mask_transposed.shape == (BT, 1, P), (
            f"loss_seq_mask shape {loss_seq_mask_transposed.shape} != ({BT},1,{P})"
        )

        return {
            "dit_hidden": dit_hidden,                          # [B*T, D_dit]
            "feat_cond": feat_cond_transposed,                 # [B*T, D_feat, P]
            "loss_seq_mask": loss_seq_mask_transposed,         # [B*T, 1, P]
            "feat_gt": feat_gt_transposed,                     # [B*T, D_feat, P]
            "all_lm_hiddens": all_lm_hiddens_for_dit,          # list of [B*T, lm_hidden] or None
            "B": B,
            "T": T,
        }

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
        video: Optional[torch.Tensor] = None,
    ):
        del position_ids  # not used yet

        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())
        labels = labels.to(self.device, dtype=torch.long)
        
        print("=== Forward Pass Input ===")
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels)
        # export text tokens and audio tokens as npy array
        # np_text_tokens = text_tokens.float().cpu().numpy()
        # np_audio_feats = audio_feats.float().cpu().numpy()
        # np.save("debug_text_tokens.npy", np_text_tokens)
        # np.save("debug_audio_feats.npy", np_audio_feats)
        
        # np_text_mask = text_mask.float().cpu().numpy()
        # np_audio_mask = audio_mask.float().cpu().numpy()
        # np_loss_mask = loss_mask.float().cpu().numpy()
        # np_labels = labels.float().cpu().numpy()
        # np.save("debug_text_mask.npy", np_text_mask)
        # np.save("debug_audio_mask.npy", np_audio_mask)
        # np.save("debug_loss_mask.npy", np_loss_mask)
        # np.save("debug_labels.npy", np_labels)
        

        B, T, P, D = audio_feats.shape
        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              feat_embed = feat_embed)

        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.multimodal_semantic_lm.embed_tokens(text_tokens) * scale_emb
        
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed)
        
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed
        combined_embed = self._inject_video(combined_embed, text_mask, video)

        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed)


        enc_outputs, _, all_lm_layer_hiddens = self.multimodal_semantic_lm(
            inputs_embeds=combined_embed, is_causal=True, output_hidden_states=True
        )
        enc_outputs = enc_outputs.to(self._dtype())

        #                              audio_feats = audio_feats,
        #                              text_mask = text_mask,
        #                              audio_mask = audio_mask,
        #                              loss_mask = loss_mask,
        #                              labels = labels,
        #                              text_embed = text_embed,
        #                              feat_embed = feat_embed,
        #                              combined_embed = combined_embed,
        #                              baselm_outputs_before_fsq = enc_outputs)
        audio_part_before_fsq = enc_outputs * audio_mask.unsqueeze(-1)
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed, 
        #                              baselm_outputs = enc_outputs,
        #                              before_fsq_audio_part = audio_part_before_fsq)
        # audio_part = self.fsq_layer(enc_outputs * audio_mask.unsqueeze(-1))
        audio_part = self.fsq_layer(audio_part_before_fsq)
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed, 
        #                              baselm_outputs = enc_outputs,
        #                              after_fsq_audio_part = audio_part)
        text_part = enc_outputs * text_mask.unsqueeze(-1)
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed, 
        #                              baselm_outputs = enc_outputs,
        #                              text_part_before_fsq = text_part)
        
        enc_outputs = audio_part + text_part
        
        # enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed, 
        #                              fsq_outputs = enc_outputs)
        
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)

        residual_inputs = enc_outputs + audio_mask.unsqueeze(-1) * feat_embed
        
        residual_outputs, _ = self.rite(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())
        
        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed, 
        #                              fsq_outputs = enc_outputs, 
        #                              residual_outputs = residual_outputs)
        
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Depth-wise semantic routing: shift each layer's hidden states by 1 (causal),
        # flatten to (B*T, lm_hidden) so each (b,t) patch gets routing from the previous
        # sequence position's multimodal_semantic_lm hidden at that layer.
        all_lm_hiddens_for_dit = [
            rearrange(
                torch.cat((torch.zeros_like(h[:, 0:1, :]), h[:, :-1, :]), dim=1),
                "b t c -> (b t) c",
            ).to(self._dtype())
            for h in all_lm_layer_hiddens
        ]

        if getattr(self, "_debug_return_dit_hidden", False):
            # Phase 1 verification: short-circuit before diffusion so callers can
            # compare forward()'s dit_hidden against compute_dit_conditioning()'s.
            return {"dit_hidden": dit_hidden.detach()}

        #                              audio_feats = audio_feats, 
        #                              text_mask = text_mask, 
        #                              audio_mask = audio_mask, 
        #                              loss_mask = loss_mask, 
        #                              labels = labels, 
        #                              text_embed = text_embed, 
        #                              feat_embed = feat_embed, 
        #                              combined_embed = combined_embed, 
        #                              fsq_outputs = enc_outputs, 
        #                              residual_outputs = residual_outputs,
        #                              dit_hidden = dit_hidden)

        # Keep diffusion inputs in the same dtype as the model (e.g., bfloat16)
        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        diff_loss = self.feat_decoder.compute_loss(
            feat_gt.transpose(1, 2).contiguous(),
            dit_hidden,
            cond=feat_cond.transpose(1, 2).contiguous(),
            tgt_mask=loss_seq_mask.transpose(1, 2).contiguous(),
            progress=progress,
            all_lm_hiddens=all_lm_hiddens_for_dit,
        )

        if self.use_stop_loss:
            stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
            stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
            denom = torch.clamp(loss_mask.sum(), min=1.0)
            stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=self.config.dit_config.cfm_config.inference_cfg_rate
                if hasattr(self.config.dit_config.cfm_config, "inference_cfg_rate")
                else 10,
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss if self.use_stop_loss else torch.tensor(0.0, device=diff_loss.device),
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }
    
    def extract_fsq_codes(self, continuous_embedding: torch.Tensor) -> torch.Tensor:
        fsq = self.fsq_layer
        # The FSQ layer has: encoder (linear → latent_dim), then quantize, then decoder
        # We need to go through the encoder to get the latent, then extract codes
        
        # Access the internal projection to latent space
        # ScalarQuantizationLayer structure: encoder_proj -> quantize -> decoder_proj
        # The quantized values are: Δ * round(x / Δ) clamped to [-L, L]
        # So the code index = round(x / Δ) + L (shift to non-negative)
        
        with torch.no_grad():
            # Project to latent space using FSQ's internal encoder
            if hasattr(fsq, 'encoder') and fsq.encoder is not None:
                latent = fsq.encoder(continuous_embedding)
            elif hasattr(fsq, 'proj_in'):
                latent = fsq.proj_in(continuous_embedding)
            else:
                # Fallback: the FSQ might operate directly on the input
                latent = continuous_embedding
            
            # Get quantization parameters
            scale = getattr(fsq, 'scale', 9)  # L parameter
            delta = 1.0  # Step size (typically 1.0 for FSQ)
            
            # Quantize and extract codes
            codes = torch.round(latent / delta).clamp(-scale, scale)
            # Shift to non-negative: [-L, L] → [0, 2L]
            codes = (codes + scale).long()
        
        return codes

    # ------------------------------------------------------------------
    # Video-conditioning hooks.
    #
    # This class is the text-only RL policy: both hooks are no-ops, so the
    # generation path below is identical to what it always was.
    # VIBEVideo2MusicRL overrides them to splice encoded video frames into the
    # LM prefill. Keeping the seam here means the RL surface (dual adapters,
    # return_latents, generate_batch, sync_old_adapter) has exactly one
    # implementation shared by both policies.
    # ------------------------------------------------------------------

    def _prepare_video(self, video_path: str):
        return None, 0, None

    def _inject_video(self, combined_embed, text_mask, video):
        return combined_embed

    def _dtype(self):
        return get_dtype(self.config.dtype)


    def generate(self, *args, return_latents: bool = False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return next(self._generate(*args, return_latents=return_latents, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0, # setting acceptable ratio of audio length to text length (for badcase detection)
        streaming: bool = False,
        return_latents: bool = False,
        video_path: str = "",
    ) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        # No-op for the text-only policy; VIBEVideo2MusicRL returns a real tensor.
        video, n_video_tokens, _video_duration = self._prepare_video(video_path)
        if len(prompt_wav_path) == 0:
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            video_pad = (
                [torch.zeros((n_video_tokens,), dtype=torch.int32, device=text_token.device)]
                if n_video_tokens > 0 else []
            )
            text_token = torch.cat(
                [
                    text_token,
                    *video_pad,
                    torch.tensor(
                        [self.audio_start_token],
                        dtype=torch.int32,
                        device=text_token.device,
                    ),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length).type(torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length).type(torch.int32).to(text_token.device)

        else:
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio, sr = torchaudio.load(prompt_wav_path)
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)    

            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            patch_len = self.patch_size * self.chunk_size

            if audio.size(1) % patch_len != 0:
                # 左填充：在音频开头填充，保持有效音频数据在序列末尾
                padding_size = patch_len - audio.size(1) % patch_len
                audio = torch.nn.functional.pad(audio, (padding_size, 0))

            # (B, D, T)
            audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate).cpu()
            audio_feat = audio_feat.view(
                self.audio_vae.latent_dim,
                -1,
                self.patch_size,
            ).permute(1, 2, 0)
            audio_length = audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_token = torch.cat([text_token, text_pad_token])
            audio_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            audio_feat = torch.cat([audio_pad_feat, audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
            )

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        target_text_length = len(self.text_tokenizer(target_text))
        
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                video=video,
                min_len=min_len,
                max_len=max_len if not retry_badcase else min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len), # unconditionally use max_len if retry_badcase is disabled
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...", file=sys.stderr)
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break   
                
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()
            if return_latents:
                yield decode_audio, latent_pred
            else:
                yield decode_audio        
    
    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str,
    ):
        if not prompt_text or not prompt_wav_path:
            raise ValueError("prompt_text and prompt_wav_path are required")

        # load audio
        audio, sr = torchaudio.load(prompt_wav_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        patch_len = self.patch_size * self.chunk_size

        if audio.size(1) % patch_len != 0:
            # Left padding: pad at the beginning of the audio to keep valid audio data at the end of the sequence
            padding_size = patch_len - audio.size(1) % patch_len
            audio = torch.nn.functional.pad(audio, (padding_size, 0))

        # extract audio features
        audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate).cpu()

        audio_feat = audio_feat.view(
            self.audio_vae.latent_dim,
            -1,
            self.patch_size,
        ).permute(1, 2, 0) # (D, T, P)
        # build prompt cache - only save raw text and audio features
        prompt_cache = {
            "prompt_text": prompt_text,
            "audio_feat": audio_feat,
        }
        
        return prompt_cache

    
    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        if original_cache is None:
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
            }
        original_prompt_text = original_cache["prompt_text"]
        original_audio_feat = original_cache["audio_feat"]
        # Merge text by concatenation
        merged_prompt_text = original_prompt_text + new_text
        merged_audio_feat = torch.cat([original_audio_feat, new_audio_feat], dim=0)

        # build new cache
        merged_cache = {
            "prompt_text": merged_prompt_text,
            "audio_feat": merged_audio_feat,
        }
        
        return merged_cache

            
    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))


    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)


    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        # get prompt from cache
        if prompt_cache is None:
            prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            text = target_text
        else:
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_text = prompt_cache["prompt_text"]
            text = prompt_text + target_text
        
        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor(
                    [self.audio_start_token],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ],
            dim=-1,
        )
        
        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
        audio_pad_feat = torch.zeros(
            (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
            device=text_token.device,
        )
        text_token = torch.cat([text_token, text_pad_token])
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)
    
        # run inference
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=max_len if not retry_badcase else min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len), # unconditionally use max_len if retry_badcase is disabled
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, pred_audio_feat in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield (
                        decode_audio,
                        target_text_token,
                        pred_audio_feat
                    )
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...", file=sys.stderr)
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            patch_len = self.patch_size * self.chunk_size
            if audio_mask.sum().item() > 0:
                decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1):]
            else:
                decode_audio = decode_audio[..., :]
            yield (
                decode_audio.cpu(),
                target_text_token,
                pred_audio_feat
            )

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._inference(*args, streaming=False, **kwargs))
    
    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        return self._inference(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
        video: Optional[torch.Tensor] = None,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        
        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0
       
        text_embed = self.multimodal_semantic_lm.embed_tokens(text) * scale_emb
        
        
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
        combined_embed = self._inject_video(combined_embed, text_mask, video)


        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None
        

        # Prepare prompt context patches for streaming mode
        # When there's a prompt audio, use its last (streaming_prefix_len - 1) patches as initial context
        prompt_context_patches = []
        audio_patch_count = int(feat_mask.sum().item())
        if audio_patch_count > 0:
            context_len = min(streaming_prefix_len - 1, audio_patch_count)
            # Take the last context_len patches from prompt audio as initial context
            # Split into list of [b, 1, p, d] tensors to match pred_feat_seq format
            prompt_context_patches = list(feat[:, -context_len:, :, :].split(1, dim=1))
            pred_feat_seq = prompt_context_patches + pred_feat_seq

        enc_outputs, kv_cache_tuple, all_lm_layer_hiddens_prefill = self.multimodal_semantic_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
            output_hidden_states=True,
        )
        all_lm_hiddens = [h[:, -1, :].to(self._dtype()) for h in all_lm_layer_hiddens_prefill]
        self.multimodal_semantic_lm.kv_cache.fill_caches(kv_cache_tuple)


        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)


        lm_hidden = enc_outputs[:, -1, :]

         
        residual_enc_outputs, residual_kv_cache_tuple = self.rite(
            inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        self.rite.kv_cache.fill_caches(residual_kv_cache_tuple)


        residual_hidden = residual_enc_outputs[:, -1, :]


        for i in range(max_len):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]
            

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                all_lm_hiddens=all_lm_hiddens,
            ).transpose(
                1, 2
            )  # [b, p, d]
            
            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)
            
            
            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # return the last three predicted latent features to provide enough context for smooth decoding
                pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                
                yield feat_pred, pred_feat_seq
            
            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break

            lm_hidden_raw, all_lm_layer_hiddens_step = self.multimodal_semantic_lm.forward_step(
                curr_embed[:, 0, :],
                torch.tensor([self.multimodal_semantic_lm.kv_cache.step()], device=curr_embed.device),
                output_hidden_states=True,
            )
            lm_hidden_raw = lm_hidden_raw.clone()
            all_lm_hiddens = [h.clone().to(self._dtype()) for h in all_lm_layer_hiddens_step]

            lm_hidden = self.fsq_layer(lm_hidden_raw)
            residual_hidden = self.rite.forward_step(
                lm_hidden + curr_embed[:, 0, :], torch.tensor([self.rite.kv_cache.step()], device=curr_embed.device)
            ).clone()


        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)  
            yield feat_pred, pred_feat_seq.squeeze(0).cpu()
            
            
    @torch.inference_mode()
    def generate_batch(
        self,
        target_text: str,
        G: int,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        min_len: int = 2,
        max_len: int = 2000,
        video_path: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        dtype = get_dtype(self.config.dtype)

        # ---- build text prefix (same for all G) ----
        # One video per prompt, shared by all G candidates.
        video, n_video_tokens, _ = self._prepare_video(video_path)
        text_token = torch.LongTensor(self.text_tokenizer(target_text))
        video_pad = ([torch.zeros((n_video_tokens,), dtype=torch.int32)]
                     if n_video_tokens > 0 else [])
        text_token = torch.cat([
            text_token,
            *video_pad,
            torch.tensor([self.audio_start_token], dtype=torch.int32),
        ], dim=-1)
        text_length = text_token.shape[0]

        audio_feat = torch.zeros(
            (text_length, self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
        )
        text_mask = torch.ones(text_length, dtype=torch.int32)
        audio_mask = torch.zeros(text_length, dtype=torch.int32)

        # Expand to batch G
        text_token_b = text_token.unsqueeze(0).expand(G, -1).to(device)           # [G, S]
        text_mask_b  = text_mask.unsqueeze(0).expand(G, -1).to(device)            # [G, S]
        audio_feat_b = audio_feat.unsqueeze(0).expand(G, -1, -1, -1).to(device).to(dtype)  # [G, S, P, D]
        audio_mask_b = audio_mask.unsqueeze(0).expand(G, -1).to(device)           # [G, S]

        # ---- temporarily widen KV cache to batch_size=G ----
        self.multimodal_semantic_lm.setup_cache(G, self.config.max_length, device, dtype)
        self.rite.setup_cache(G, self.config.max_length, device, dtype)

        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feat_embed = self.feat_encoder(audio_feat_b)
                feat_embed = self.enc_to_lm_proj(feat_embed)

                scale_emb = self.config.lm_config.scale_emb if self.config.lm_config.use_mup else 1.0
                text_embed = self.multimodal_semantic_lm.embed_tokens(text_token_b) * scale_emb

                combined_embed = text_mask_b.unsqueeze(-1) * text_embed + audio_mask_b.unsqueeze(-1) * feat_embed
                # One video per prompt -> broadcast it across the G candidates.
                video_b = video.expand(G, *video.shape[1:]) if video is not None else None
                combined_embed = self._inject_video(combined_embed, text_mask_b, video_b)
                prefix_feat_cond = audio_feat_b[:, -1, ...]  # [G, P, D]

                enc_outputs, kv_cache_tuple, all_lm_layer_hiddens_prefill = self.multimodal_semantic_lm(
                    inputs_embeds=combined_embed, is_causal=True, output_hidden_states=True,
                )
                all_lm_hiddens = [h[:, -1, :].to(dtype) for h in all_lm_layer_hiddens_prefill]
                self.multimodal_semantic_lm.kv_cache.fill_caches(kv_cache_tuple)

                enc_outputs = (self.fsq_layer(enc_outputs) * audio_mask_b.unsqueeze(-1)
                               + enc_outputs * text_mask_b.unsqueeze(-1))
                lm_hidden = enc_outputs[:, -1, :]  # [G, H]

                residual_enc_outputs, residual_kv_cache_tuple = self.rite(
                    inputs_embeds=enc_outputs + audio_mask_b.unsqueeze(-1) * feat_embed,
                    is_causal=True,
                )
                self.rite.kv_cache.fill_caches(residual_kv_cache_tuple)
                residual_hidden = residual_enc_outputs[:, -1, :]  # [G, H]

                pred_feat_seqs = [[] for _ in range(G)]
                stop_flags = [False] * G

                for i in range(max_len):
                    dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)  # [G, H_dit]

                    # DiT decode: runs all G in parallel
                    pred_feat = self.feat_decoder(
                        mu=dit_hidden,
                        patch_size=self.patch_size,
                        cond=prefix_feat_cond.transpose(1, 2).contiguous(),  # [G, D, P]
                        n_timesteps=inference_timesteps,
                        cfg_value=cfg_value,
                        all_lm_hiddens=all_lm_hiddens,
                    ).transpose(1, 2)  # [G, P, D]

                    for g in range(G):
                        pred_feat_seqs[g].append(pred_feat[g:g+1].unsqueeze(1))  # [1, 1, P, D]

                    curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # [G, 1, C]
                    curr_embed = self.enc_to_lm_proj(curr_embed)
                    prefix_feat_cond = pred_feat

                    # Check stop flags (per-sample, use lm_hidden[g])
                    stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))  # [G, 2]
                    new_stops = stop_logits.argmax(dim=-1).cpu().tolist()  # [G]
                    if i > min_len:
                        for g in range(G):
                            if new_stops[g] == 1:
                                stop_flags[g] = True
                        if all(stop_flags):
                            break

                    # AR step for all G in parallel.
                    # All G samples share the same sequence position, so step_idx is a
                    # scalar — key_cache[:, :, position_id, :] writes all batch rows at once.
                    # Advance both cache counters together so overflow guards stay in sync.
                    pos = self.multimodal_semantic_lm.kv_cache.step()
                    self.rite.kv_cache.step()
                    step_idx = torch.tensor(pos, device=device)
                    lm_hidden_raw, all_lm_layer_hiddens_step = self.multimodal_semantic_lm.forward_step(
                        curr_embed[:, 0, :], step_idx, output_hidden_states=True,
                    )
                    lm_hidden_raw = lm_hidden_raw.clone()
                    all_lm_hiddens = [h.clone().to(dtype) for h in all_lm_layer_hiddens_step]

                    lm_hidden = self.fsq_layer(lm_hidden_raw)
                    residual_hidden = self.rite.forward_step(
                        lm_hidden + curr_embed[:, 0, :], step_idx,
                    ).clone()

            # ---- decode each sample ----
            wavs = []
            latents = []
            for g in range(G):
                seq = torch.cat(pred_feat_seqs[g], dim=1)  # [1, T, P, D]
                feat_pred = rearrange(seq, "b t p d -> b d (t p)", p=self.patch_size)  # [1, D, T*P]
                wav = self.audio_vae.decode(feat_pred.to(torch.float32)).squeeze(1).cpu()  # [1, T_audio]
                wavs.append(wav)
                latents.append(feat_pred.squeeze(0).cpu())  # [D, T*P]

            wavs_tensor    = torch.cat(wavs, dim=0)           # [G, T_audio]
            latents_tensor = torch.stack(latents, dim=0)      # [G, D, T*P]

        finally:
            # Always restore KV cache to batch_size=1 for regular generate() calls
            self.multimodal_semantic_lm.setup_cache(1, self.config.max_length, device, dtype)
            self.rite.setup_cache(1, self.config.max_length, device, dtype)

        return wavs_tensor, latents_tensor

    @classmethod
    def load_songbloom_vae(cls, audiovae_path: str):
        # Load Audio VAE
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
        import json
        
        config_path = f"{audiovae_path}/stable_audio_1920_vae.json"
        ckpt_path   = f"{audiovae_path}/autoencoder_music_dsp1920.ckpt"

        with open(config_path) as f:
            model_config = json.load(f)

        # This builds the autoencoder (and only the autoencoder if the config is for it)
        vae = create_model_from_config(model_config)

        state_dict = load_ckpt_state_dict(ckpt_path)
        copy_state_dict(vae, state_dict)

        vae.eval()
        return vae

    @classmethod
    def from_local(cls, path: str, baselm_path: str, audiovae_path: str, start_from_weight: str = "", optimize: bool = True, training: bool = False, lora_config: LoRAConfig = None, patch_size: int = 4, use_stop_loss: bool = True):
        print(f"[from_local] path: {path}")
        print(f"[from_local] baselm_path: {baselm_path}")
        print(f"[from_local] audiovae_path: {audiovae_path}")
        print(f"[from_local] start_from_weight: {start_from_weight}")
        print(f"[from_local] optimize: {optimize}")
        print(f"[from_local] training: {training}")
        print(f"[from_local] lora_config: {lora_config}")
        print(f"[from_local] patch_size: {patch_size}")
        print(f"[from_local] use_stop_loss: {use_stop_loss}")
        config = VIBEConfig.model_validate_json(open(os.path.join(path, "config.json")).read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)

        audio_vae = cls.load_songbloom_vae(audiovae_path)

        model = cls(config, tokenizer, audio_vae, lora_config, patch_size=patch_size, use_stop_loss=use_stop_loss)
        if not training:
            print("not training")
            lm_dtype = get_dtype(model.config.dtype)
            model = model.to(lm_dtype)
        else: # training mode
            print("training")
            for name, param in model.named_parameters():
                if "audio_vae" in name: # freeze VAE weights
                    param.requires_grad = False
                    continue
                if lora_config is not None:
                    if "lora" not in name: # freeze non-LoRA weights
                        param.requires_grad = False
        model.audio_vae = model.audio_vae.to(torch.float32)
        
        # Detect whether start_from_weight is a LoRA-only checkpoint (dir
        # containing lora_weights.safetensors / lora_weights.ckpt) so we can
        # overlay it on top of the pretrained base, instead of switching the
        # whole base path to the LoRA dir (which previously caused the next
        # training stage to silently start from a partial state dict —
        # essentially un-finetuned base weights). Mirrors the fix in
        # vibe_v2m.VIBEVideo2Music.from_local.
        is_lora_ckpt = False
        _lora_st = ""
        _lora_ckpt = ""
        if len(start_from_weight) > 0:
            _lora_st   = os.path.join(start_from_weight, "lora_weights.safetensors")
            _lora_ckpt = os.path.join(start_from_weight, "lora_weights.ckpt")
            is_lora_ckpt = os.path.exists(_lora_st) or os.path.exists(_lora_ckpt)

        if not training or len(start_from_weight) > 0:
            # For a full checkpoint, switch path; for a LoRA checkpoint, keep
            # path pointing at the pretrained base — we'll overlay LoRA on top.
            if len(start_from_weight) > 0 and training and not is_lora_ckpt:
                path = start_from_weight
            # Try to load from safetensors first, fallback to pytorch_model.bin
            safetensors_path = os.path.join(path, "model.safetensors")
            pytorch_model_path = os.path.join(path, "pytorch_model.bin")

            if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
                print(f"Loading model from safetensors: {safetensors_path}", file=sys.stderr)
                model_state_dict = load_file(safetensors_path)
            elif os.path.exists(pytorch_model_path):
                print(f"Loading model from pytorch_model.bin: {pytorch_model_path}", file=sys.stderr)
                checkpoint = torch.load(
                    pytorch_model_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model_state_dict = checkpoint.get("state_dict", checkpoint)
            else:
                raise FileNotFoundError(
                    f"Model file not found. Expected either {safetensors_path} or {pytorch_model_path}"
                )

            # Overlay any prior LoRA weights on top of the base state dict.
            # The freshly-attached LoRA layers (from `lora_config`) get their
            # parameters overwritten by these prior values via load_state_dict
            # below, so the next training stage starts from the previously
            # finetuned LoRA position rather than random init.
            if is_lora_ckpt:
                if os.path.exists(_lora_st) and SAFETENSORS_AVAILABLE:
                    print(f"Loading LoRA weights from: {_lora_st}", file=sys.stderr)
                    lora_state_dict = load_file(_lora_st)
                else:
                    print(f"Loading LoRA weights from: {_lora_ckpt}", file=sys.stderr)
                    _ckpt = torch.load(_lora_ckpt, map_location="cpu", weights_only=True)
                    lora_state_dict = _ckpt.get("state_dict", _ckpt)
                model_state_dict.update(lora_state_dict)
                print(f"Merged {len(lora_state_dict)} LoRA parameters into base model.",
                      file=sys.stderr)
        # print model param dtypes
        # for name, param in model.named_parameters():
            
        if training and len(start_from_weight) == 0:
            print("Loading only base language model for training.")
            # Only load multimodal_semantic_lm weights while training
            path = 'openbmb/MiniCPM4-0.5B'
            # path = "openbmb/MiniCPM4-8B"
            device = "cuda"
            lm = AutoModelForCausalLM.from_pretrained(
                    path, 
                    torch_dtype=torch.bfloat16, 
                    device_map=device, 
                    trust_remote_code=True
                )
            baselm_state_dict = lm.state_dict()
            # model.multimodal_semantic_lm.load_state_dict(lm_state_dict, strict=False)
        
            model_state_dict = model.state_dict()
        
            for model_key in model_state_dict:
                if "multimodal_semantic_lm." in model_key and model_key[len("multimodal_semantic_lm."):] in baselm_state_dict:
                    model_state_dict[model_key] = baselm_state_dict[model_key[len("multimodal_semantic_lm."):]]
            model_state_dict = {k: v for k, v in model_state_dict.items() if "multimodal_semantic_lm." in k}
                    
        # LoRALinear holds weight/bias directly, compatible with nn.Linear state_dict keys.
        # Using strict=False since pretrained weights don't contain lora_A/lora_B.
        # if not training:
        model.load_state_dict(model_state_dict, strict=False)
        # print all param dtypes
        # for name, param in model.named_parameters():
        if training:
            return model
        return model.to(model.device).eval().optimize(disable=not optimize)

    # ------------------------------------------------------------------ #
    # LoRA Weight Management
    # ------------------------------------------------------------------ #
    def _iter_lora_modules(self):
        from modules.layers.lora import LoRALinear
        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module

    def load_lora_weights(self, lora_path: str, device: str = None):
        from pathlib import Path
        
        device = device or self.device
        lora_path = Path(lora_path)
        
        # Try safetensors first, then fallback to .ckpt
        if lora_path.is_dir():
            safetensors_file = lora_path / "lora_weights.safetensors"
            ckpt_file = lora_path / "lora_weights.ckpt"
        else:
            safetensors_file = lora_path if lora_path.suffix == ".safetensors" else None
            ckpt_file = lora_path if lora_path.suffix in [".ckpt", ".pth"] else None
        
        # Load from safetensors if available
        if safetensors_file and safetensors_file.exists() and SAFETENSORS_AVAILABLE:
            state_dict = load_file(str(safetensors_file), device=device)
        elif ckpt_file and ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(
                f"LoRA checkpoint not found. Expected either {safetensors_file} or {ckpt_file}"
            )
        
        # Build param mapping (handle torch.compile's _orig_mod prefix)
        model_params = dict(self.named_parameters())
        key_mapping = {k.replace("._orig_mod.", "."): k for k in model_params if "._orig_mod." in k}
        
        loaded_keys, skipped_keys = [], []
        for key, value in state_dict.items():
            target_key = key if key in model_params else key_mapping.get(key)
            if target_key:
                model_params[target_key].data.copy_(value.to(device))
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        
        return loaded_keys, skipped_keys

    def set_lora_enabled(self, enabled: bool):
        for module in self._iter_lora_modules():
            module.set_enabled(enabled)

    def reset_lora_weights(self):
        for module in self._iter_lora_modules():
            module.reset_lora_parameters()

    def get_lora_state_dict(self) -> dict:
        return {name: param.data.clone()
                for name, param in self.named_parameters()
                if "lora_" in name}

    # ------------------------------------------------------------------ #
    # Dual-Adapter Management (for DiffusionNFT LoRA training)
    # ------------------------------------------------------------------ #

    def _init_dual_adapters(self):
        self._adapter_storage = {
            "old": {}  # name → tensor (clones of current lora params)
        }
        self._active_adapter = "default"

        # Clone current LoRA weights as the initial "old" adapter
        for name, param in self.named_parameters():
            if "lora_" in name:
                self._adapter_storage["old"][name] = param.data.clone()

    def set_adapter(self, adapter_name: str):
        if not hasattr(self, "_adapter_storage"):
            raise RuntimeError("Call _init_dual_adapters() before set_adapter().")
        if adapter_name == self._active_adapter:
            return

        current = self._active_adapter
        target = adapter_name

        if current == "default" and target == "old":
            if "_default_backup" not in self._adapter_storage:
                self._adapter_storage["_default_backup"] = {}
            for name, param in self.named_parameters():
                if "lora_" in name:
                    self._adapter_storage["_default_backup"][name] = param.data.clone()
                    param.data.copy_(self._adapter_storage["old"][name])

        elif current == "old" and target == "default":
            for name, param in self.named_parameters():
                if "lora_" in name:
                    self._adapter_storage["old"][name] = param.data.clone()
                    param.data.copy_(self._adapter_storage["_default_backup"][name])

        else:
            raise ValueError(f"Unknown adapter transition: {current!r} → {target!r}")

        self._active_adapter = target

    def disable_adapters(self):
        self.set_lora_enabled(False)

    def enable_adapters(self):
        self.set_lora_enabled(True)

    def get_adapter_parameters(self, adapter_name: str) -> dict:
        if not hasattr(self, "_adapter_storage"):
            raise RuntimeError("Call _init_dual_adapters() before get_adapter_parameters().")
        if adapter_name == "default":
            assert self._active_adapter == "default", (
                "Call set_adapter('default') before get_adapter_parameters('default')"
            )
            return {name: param for name, param in self.named_parameters()
                    if "lora_" in name}
        elif adapter_name == "old":
            return self._adapter_storage["old"]
        else:
            raise ValueError(f"Unknown adapter: {adapter_name!r}")

    def sync_old_adapter(self, decay: float = 0.0):
        if not hasattr(self, "_adapter_storage"):
            raise RuntimeError("Call _init_dual_adapters() before sync_old_adapter().")
        assert self._active_adapter == "default", (
            "sync_old_adapter requires default adapter to be active"
        )
        for name, param in self.named_parameters():
            if "lora_" in name:
                old_val = self._adapter_storage["old"][name]
                old_val.mul_(decay).add_(param.data, alpha=1.0 - decay)
