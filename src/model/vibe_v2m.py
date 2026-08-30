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
import sys
from torchvision.io import VideoReader
from transformers import CLIPImageProcessor

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
from modules.video_encoder import VideoEncoder
from modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens

def log_shapes_and_stats(tensor: torch.Tensor, name: str, stop_print: bool = True):
    if stop_print:
        return
    print(f"[Logger] {name} - shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
    if tensor.numel() > 0 and tensor.dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
        print(f"[Logger] {name} - min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}")
    else:
        print(f"[Logger] {name} - tensor is empty.")


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
    model_config = ConfigDict(arbitrary_types_allowed=True) # to allow hf transformers config in pydantic
    
    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    rite_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9
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


# ---------------------------------------------------------------------------
# Video conditioning primitives.
#
# Shared by VIBEVideo2Music (stage 4, SFT) and VIBEVideo2MusicRL (stage 5, RL)
# so frame sampling and embedding placement have exactly one implementation.
# ---------------------------------------------------------------------------

#: clips are truncated to this many seconds before frame sampling
MAX_VIDEO_SECONDS = 10


def get_total_frames(video_reader) -> Tuple[int, float]:
    metadata = video_reader.get_metadata()
    duration = min(metadata['video']['duration'][0], MAX_VIDEO_SECONDS)
    fps = metadata['video']['fps'][0]
    return int(duration * fps), duration


def sample_video_frames(video_reader, n_frames: int) -> Tuple[List[torch.Tensor], float]:
    total, duration = get_total_frames(video_reader)
    indices = set(np.linspace(0, max(total - 1, 0), n_frames, dtype=int))
    frames = [f['data'] for i, f in enumerate(video_reader) if i in indices]
    return frames, duration


def prep_video_frames(clip_processor, frames: List[torch.Tensor]) -> torch.Tensor:
    return clip_processor(images=frames, return_tensors="pt")['pixel_values'].squeeze(0)


def inject_video_embeddings(
    combined_embed: torch.Tensor,
    text_mask: torch.Tensor,
    video_embed: torch.Tensor,
    strict: bool = False,
) -> torch.Tensor:
    text_end = (text_mask == 0).to(torch.int).argmax(dim=1)
    video_len = video_embed.size(1)
    start_idx = text_end - video_len - 1
    if strict and torch.any(start_idx < 0):
        raise ValueError("text_end must be at least video_len + 1 to insert video embeddings")
    insert_idx = start_idx.unsqueeze(1) + torch.arange(video_len, device=combined_embed.device)
    batch_idx = torch.arange(combined_embed.size(0), device=combined_embed.device).unsqueeze(1)
    combined_embed[batch_idx, insert_idx, :] = video_embed.to(combined_embed.dtype)
    return combined_embed


class VIBEVideo2Music(nn.Module):
    def __init__(
        self,
        config: VIBEConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
        lora_config: LoRAConfig = None,
        use_stop_loss: bool = False,
    ):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
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
        
        # Video Encoder
        self.n_video_frames = config.n_video_frames
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.video_encoder = VideoEncoder(config.video_encoder_config) # Placeholder for future video encoder integration
        
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
        self.video_to_lm_proj = nn.Linear(config.video_encoder_config.projection_dim, config.lm_config.hidden_size) # Placeholder for future video to LM tokens mapping
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.use_stop_loss = use_stop_loss
        if use_stop_loss:
            self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
            self.stop_actn = nn.SiLU()
            self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
            self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.downsampling_ratio  # SongBloom music VAE
        self.sample_rate = audio_vae.sample_rate

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
    
    
    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        video_input: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
    ):
        del position_ids  # not used yet
        dtype = self._dtype()
        print("== Dtype of inputs:", dtype, "==")

        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=dtype)
        audio_feats = audio_feats.to(self.device, dtype=dtype)
        audio_mask = audio_mask.to(self.device, dtype=dtype)
        loss_mask = loss_mask.to(self.device, dtype=dtype)
        labels = labels.to(self.device, dtype=torch.long)
        video_input = video_input.to(self.device, dtype=dtype)
        text_mask_unsq = text_mask.unsqueeze(-1)
        audio_mask_unsq = audio_mask.unsqueeze(-1)
        
        # export text tokens and audio tokens as npy array
        

        B, _, _, _ = audio_feats.shape
        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        
        video_embed = self.video_encoder(video_input)  # [B, T_v, video_embed]
        video_embed = self.video_to_lm_proj(video_embed)  # [B, T_v, lm_hidden_size]
        

        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.multimodal_semantic_lm.embed_tokens(text_tokens) * scale_emb
        
        
        combined_embed = text_mask_unsq * text_embed + audio_mask_unsq * feat_embed
        combined_embed = combined_embed.to(dtype)

        text_end = (text_mask == 0).to(torch.int).argmax(dim=1)
        # combined_embed[:, text_end - 9: text_end - 1, :] = video_embed
        
        combined_embed = inject_video_embeddings(combined_embed, text_mask, video_embed, strict=True)


        enc_outputs, _, all_lm_layer_hiddens = self.multimodal_semantic_lm(
            inputs_embeds=combined_embed, is_causal=True, output_hidden_states=True
        )
        del combined_embed, text_embed, video_embed
        enc_outputs = enc_outputs.to(self._dtype())

        audio_part = self.fsq_layer(enc_outputs * audio_mask_unsq)
        enc_outputs = enc_outputs * text_mask_unsq
        enc_outputs.add_(audio_part)
        del audio_part
        
        # enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        
        lm_hidden = F.pad(enc_outputs[:, :-1, :], (0, 0, 1, 0))

        residual_outputs, _ = self.rite(
            inputs_embeds=enc_outputs + audio_mask_unsq * feat_embed, is_causal=True
        )
        del enc_outputs, feat_embed
        residual_outputs = residual_outputs.to(self._dtype())

        residual_hidden = F.pad(residual_outputs[:, :-1, :], (0, 0, 1, 0))
        del residual_outputs

        dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)
        del residual_hidden
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Depth-wise routing: shift each layer's hidden states by 1 (same as lm_hidden)
        # and flatten to (B*T, lm_hidden) so each (b,t) audio patch gets the routing
        # conditioning from the multimodal_semantic_lm hidden at the previous sequence position.
        dtype = self._dtype()
        all_lm_hiddens_for_dit = []
        for h in all_lm_layer_hiddens:
            shifted = torch.cat((torch.zeros_like(h[:, 0:1, :]), h[:, :-1, :]), dim=1)
            all_lm_hiddens_for_dit.append(rearrange(shifted, "b t c -> (b t) c").to(dtype))
            del shifted
        del all_lm_layer_hiddens


        # rearrange directly to (B*T, D, P) — avoids a separate .transpose().contiguous() copy
        feat_gt = rearrange(audio_feats.contiguous(), "b t p d -> (b t) d p")
        feat_cond = rearrange(
            F.pad(audio_feats[:, :-1, ...], (0, 0, 0, 0, 1, 0)).contiguous(),
            "b t p d -> (b t) d p",
        )
        del audio_feats

        loss_seq_mask = rearrange(
            loss_mask.unsqueeze(-1).expand(-1, -1, self.patch_size),
            "b t p -> (b t) 1 p",
        )

        diff_loss = self.feat_decoder.compute_loss(
            feat_gt,
            dit_hidden,
            cond=feat_cond,
            tgt_mask=loss_seq_mask,
            progress=progress,
            all_lm_hiddens=all_lm_hiddens_for_dit,
        )
        del all_lm_hiddens_for_dit

        if self.use_stop_loss:
            stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
            stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
            denom = torch.clamp(loss_mask.sum(), min=1.0)
            stop_loss = (stop_losses * loss_mask).sum() / denom
        else:
            stop_loss = torch.tensor(0.0)
        del lm_hidden

        feat_pred = None
        if sample_generate:
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond,
                n_timesteps=self.config.dit_config.cfm_config.inference_cfg_rate
                if hasattr(self.config.dit_config.cfm_config, "inference_cfg_rate")
                else 10,
            )
            feat_pred = rearrange(feat_pred_seq, "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }

    def _dtype(self):
        return get_dtype(self.config.dtype)


    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        video_path: str,
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
    ) -> Generator[torch.Tensor, None, None]:
        # read video
        ###########!!!!!!!!!!!!!!
        frames, duration = self.__sample_frames(VideoReader(video_path))
        video = self.__prep_video(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        
        _, T, _, _, _ = video.shape
        
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        if len(prompt_wav_path) == 0:
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.zeros((T,), dtype=torch.int32),  # pad text tokens for video frames
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


        # target_text_length = len(self.text_tokenizer(target_text))
        
        retry_badcase_times = 0
        print("Duration in _generate", duration)
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                video,
                min_len=min_len,
                max_len=min(int(duration * 63), max_len), # avoid too long audio
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, _, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat, _ = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= duration * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_duration_ratio={pred_audio_feat.shape[0] / duration}, retrying...", file=sys.stderr)
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break   
                
        if not streaming:
            print("=== Final Decoding Step ===")
            log_shapes_and_stats(latent_pred, "Final latent_pred for decoding")
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()  
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

    def __get_total_frames(self, video_reader) -> Tuple[int, float]:
        return get_total_frames(video_reader)

    def __sample_frames(self, video_reader) -> Tuple[List[torch.Tensor], float]:
        return sample_video_frames(video_reader, self.n_video_frames)

    def __prep_video(self, frames: List[torch.Tensor]) -> torch.Tensor:
        return prep_video_frames(self.clip_processor, frames)  # (T, C, H, W)


    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        video_path: str,
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
        return_planner_latents: bool = False,
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
        
        # read video
        ###########!!!!!!!!!!!!!!
        frames, duration = self.__sample_frames(VideoReader(video_path))
        video = self.__prep_video(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        
        _, T, _, _, _ = video.shape
        
        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.zeros((T,), dtype=torch.int32),  # pad text tokens for video frames
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
        retry_badcase_times = 0
        print("Duration in _generate_with_prompt_cache", duration, int(duration * 6.3))
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                video,
                min_len=min_len,
                max_len=min(int(duration * 6.3), max_len), # avoid too long audio
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
                return_planner_latents=return_planner_latents,
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
                latent_pred, pred_audio_feat, latents = next(inference_result)
                planner_latent, residual_latent, fsq_baselm_latents, dit_latents = None, None, None, None
                if return_planner_latents:
                    planner_latent, residual_latent, fsq_baselm_latents, dit_latents = latents
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= duration * 22 * retry_badcase_ratio_threshold:
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
            yield {
                "audio": decode_audio.cpu(),
                "target_text_tokens": target_text_token,
                "audio_features": pred_audio_feat,
                "planner_latents": planner_latent if return_planner_latents else None,
                "residual_latents": residual_latent if return_planner_latents else None,
                "fsq_baselm_latents": fsq_baselm_latents if return_planner_latents else None,
                "dit_latents": dit_latents if return_planner_latents else None
            }

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
        video: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
        return_planner_latents: bool = False
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        print("=== Inference: After Audio Feature Encoder ===")
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, feat_embed = feat_embed)
        
        video_embed = self.video_encoder(video)  # [B, T_v, video_embed]
        video_embed = self.video_to_lm_proj(video_embed)  # [B, T_v, lm_hidden_size]
        
        print("=== Inference: After Video Encoder ===")
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, feat_embed = feat_embed, video = video, video_embed = video_embed)
        
        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0
       
        text_embed = self.multimodal_semantic_lm.embed_tokens(text) * scale_emb
        
        print("=== Inference: After Text Embedding Layer ===")
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed)
        
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
        
        combined_embed = inject_video_embeddings(combined_embed, text_mask, video_embed)

        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed)

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None
        
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed, prefix_feat_cond = prefix_feat_cond)

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
        # Per-layer hiddens at the last prompt position (shape [B, lm_hidden] each)
        all_lm_hiddens = [h[:, -1, :].to(self._dtype()) for h in all_lm_layer_hiddens_prefill]
        self.multimodal_semantic_lm.kv_cache.fill_caches(kv_cache_tuple)
        
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed, enc_outputs = enc_outputs)
        
        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed, enc_outputs = enc_outputs)
        
        lm_hidden = enc_outputs[:, -1, :]

         
        residual_enc_outputs, residual_kv_cache_tuple = self.rite(
            inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        self.rite.kv_cache.fill_caches(residual_kv_cache_tuple)

        
        log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed, enc_outputs = enc_outputs, residual_enc_outputs = residual_enc_outputs)
        
        residual_hidden = residual_enc_outputs[:, -1, :]

        print(max_len)
        take_idx = 10
        rec_residual_hidden = None
        rec_fsq_baselm_hidden = None
        rec_planner_hidden = None
        rec_dit_latents = None
        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]
            
            
            log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed, enc_outputs = enc_outputs, dit_hidden_1 = dit_hidden_1, dit_hidden_2 = dit_hidden_2, dit_hidden = dit_hidden)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                full_latents=True if return_planner_latents else False,
                all_lm_hiddens=all_lm_hiddens,
            ).transpose(1, 2)  # [b, d, p]
            
            # if return_planner_latents:
            #     pred_feat = pred_feat[-1].transpose(1, 2)  # [b, p, d]
            # else:
            #     pred_feat = pred_feat.transpose(1, 2)  # [b, p, d]
                
            if i == take_idx and return_planner_latents:
                rec_residual_hidden = residual_hidden.clone()
                rec_fsq_baselm_hidden = lm_hidden.clone()
                rec_planner_hidden = dit_hidden.clone()
                rec_dit_latents = None
            
            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)
            
            log_shapes_and_stats_for_all(text = text, feat = feat, text_mask = text_mask, feat_mask = feat_mask, text_embed = text_embed, feat_embed = feat_embed, combined_embed = combined_embed, dit_hidden = dit_hidden, pred_feat = pred_feat, curr_embed = curr_embed)
            
            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # return the last three predicted latent features to provide enough context for smooth decoding
                pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                
                yield feat_pred, pred_feat_seq
            
            # stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            # if i > min_len and stop_flag == 1:
            #     break
    
            lm_hidden_raw, all_lm_layer_hiddens_step = self.multimodal_semantic_lm.forward_step(
                curr_embed[:, 0, :],
                torch.tensor([self.multimodal_semantic_lm.kv_cache.step()], device=curr_embed.device),
                output_hidden_states=True,
            )
            all_lm_hiddens = [h.to(self._dtype()) for h in all_lm_layer_hiddens_step]

            lm_hidden = self.fsq_layer(lm_hidden_raw)
            residual_hidden = self.rite.forward_step(
                lm_hidden + curr_embed[:, 0, :], torch.tensor([self.rite.kv_cache.step()], device=curr_embed.device)
            ).clone()


        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)  
            if return_planner_latents:
                # planner_latent, residual_latent, fsq_baselm_latents, dit_latents
                yield feat_pred, pred_feat_seq.squeeze(0).cpu(), (rec_planner_hidden.detach().float().cpu(), rec_residual_hidden.detach().float().cpu(), rec_fsq_baselm_hidden.detach().float().cpu(), None)
            else:
                yield feat_pred, pred_feat_seq.squeeze(0).cpu(), None
            
            
    def unfreeze_ttm(self, model):
        for name, param in model.named_parameters():
            if "audio_vae" in name: # freeze VAE weights
                param.requires_grad = False
                continue
            
            if "video_encoder" in name: # freeze video encoder weights
                param.requires_grad = False
                continue
            
            if "video_to_lm_proj" not in name: # unfreeze the model except the video_to_lm_proj
                param.requires_grad = True
                continue
            
            if "video_to_lm_proj" in name: 
                param.requires_grad = True
                continue
    
            
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
    def from_local(cls, path: str, patch_size: int, baselm_path: str, audiovae_path: str, video_embed_name: str = "openai/clip-vit-base-patch32", optimize: bool = True, training: bool = False, lora_config: LoRAConfig = None, use_stop_loss: bool = False, start_from_weight: str = ""):
        config = VIBEConfig.model_validate_json(open(os.path.join(path, "config.json")).read())
        config.patch_size = patch_size
        tokenizer = LlamaTokenizerFast.from_pretrained(path)

        audio_vae = cls.load_songbloom_vae(audiovae_path)

        model = cls(config, tokenizer, audio_vae, lora_config, use_stop_loss=use_stop_loss)
        
        if not training:
            lm_dtype = get_dtype(model.config.dtype)
            model = model.to(lm_dtype)
        else: # training mode
            for name, param in model.named_parameters():
                if "audio_vae" in name: # freeze VAE weights
                    param.requires_grad = False
                    continue
                
                if "video_encoder" in name: # freeze video encoder weights
                    param.requires_grad = False
                    continue
                
                # if "video_to_lm_proj" not in name: # freeze the model except the video_to_lm_proj, unfreeze this part later to help convergence since video encoder is frozen
                #     param.requires_grad = False
                #     continue
                
                # if "video_to_lm_proj" in name: 
                #     param.requires_grad = True
                #     continue
                
                if lora_config is not None:
                    if "lora" not in name: # freeze non-LoRA weights
                        param.requires_grad = False
        model.audio_vae = model.audio_vae.to(torch.float32)
        lm_dtype = get_dtype(model.config.dtype)
        model.video_encoder = model.video_encoder.to(lm_dtype)
        for name, param in model.video_encoder.named_parameters():
            param.requires_grad = False
        
        # Detect whether start_from_weight is a LoRA checkpoint before entering blocks.
        is_lora_ckpt = False
        if len(start_from_weight) > 0:
            _lora_st   = os.path.join(start_from_weight, "lora_weights.safetensors")
            _lora_ckpt = os.path.join(start_from_weight, "lora_weights.ckpt")
            is_lora_ckpt = os.path.exists(_lora_st) or os.path.exists(_lora_ckpt)

        if not training or len(start_from_weight) > 0:
            # For a full checkpoint, switch path; for LoRA, keep path as the base model.
            if len(start_from_weight) > 0 and training and not is_lora_ckpt:
                path = start_from_weight

            # Load base model weights (always from `path`)
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

            # Overlay LoRA weights on top of the base model state dict.
            if is_lora_ckpt:
                if os.path.exists(_lora_st) and SAFETENSORS_AVAILABLE:
                    print(f"Loading LoRA weights from: {_lora_st}", file=sys.stderr)
                    lora_state_dict = load_file(_lora_st)
                else:
                    print(f"Loading LoRA weights from: {_lora_ckpt}", file=sys.stderr)
                    _ckpt = torch.load(_lora_ckpt, map_location="cpu", weights_only=True)
                    lora_state_dict = _ckpt.get("state_dict", _ckpt)
                model_state_dict.update(lora_state_dict)
                print(f"Merged {len(lora_state_dict)} LoRA parameters into base model.", file=sys.stderr)

        if training:
            if len(start_from_weight) == 0:
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
                model_state_dict = {k: v for k, v in model_state_dict.items() if "multimodal_semantic_lm." in k or "video_encoder." in k}
            elif not is_lora_ckpt:
                # Full model checkpoint: video encoder weights may not be saved; load from HF.
                model.video_encoder.from_pretrained(video_embed_name)
                model_state_dict = {
                    **model_state_dict,
                    **{k: v for k, v in model_state_dict.items() if "video_encoder." in k}, # override with checkpoint's video encoder if present
                }
            # LoRA checkpoint: base model already contains all non-LoRA weights including
            # the video encoder, so no extra loading is needed.
        
        # LoRALinear holds weight/bias directly, compatible with nn.Linear state_dict keys.
        # Using strict=False since pretrained weights don't contain lora_A/lora_B.
        # if not training:
        model.load_state_dict(model_state_dict, strict=False)
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

