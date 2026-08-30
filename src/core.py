import os
import sys
import re
import tempfile
import numpy as np
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from model.vibe_ttm import VIBETextToMusic, LoRAConfig

class VIBE:
    def __init__(self,
            voxcpm_model_path : str,
            baselm_path : str,
            audiovae_path : str,
            zipenhancer_model_path : str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            enable_denoiser : bool = True,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            patch_size: int = 4,
        ):
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}", file=sys.stderr)
        
        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}", file=sys.stderr)
        
        self.tts_model = VIBETextToMusic.from_local(voxcpm_model_path, 
                                                optimize=optimize, 
                                                lora_config=lora_config, 
                                                baselm_path = baselm_path,
                                                audiovae_path = audiovae_path,
                                                patch_size = patch_size,
                                                )
        
        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}", file=sys.stderr)
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}", file=sys.stderr)
        
        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        if optimize:
            print("Warm up VIBETextToMusic...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM1.5",
            baselm_path : str = "",
            audiovae_path : str = "",
            load_denoiser: bool = True,
            zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            cache_dir: str = None,
            local_files_only: bool = False,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            patch_size: int = 4,
            **kwargs,
        ):
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")
        
        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            baselm_path = baselm_path,
            audiovae_path = audiovae_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            patch_size=patch_size,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            min_len : int = 2,
            max_len : int = 4096,
            normalize : bool = False,
            denoise : bool = False,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
        
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        temp_prompt_wav_path = None
        
        try:
            if prompt_wav_path is not None and prompt_text is not None:
                if denoise and self.denoiser is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                    prompt_wav_path = temp_prompt_wav_path
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text
                )
            else:
                fixed_prompt_cache = None  # will be built from the first inference
            
            if normalize:
                if self.text_normalizer is None:
                    from utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)
            
            generate_result = self.tts_model._generate_with_prompt_cache(
                            target_text=text,
                            prompt_cache=fixed_prompt_cache,
                            min_len=min_len,
                            max_len=max_len,
                            inference_timesteps=inference_timesteps,
                            cfg_value=cfg_value,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                            streaming=streaming,
                        )
        
            for wav, _, _ in generate_result:
                yield wav.squeeze(0).cpu().numpy()
        
        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # LoRA Interface (delegated to VIBETextToMusic)
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
                "Please reinitialize with lora_config or lora_weights_path parameter."
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        self.tts_model.reset_lora_weights()
    
    def set_lora_enabled(self, enabled: bool):
        self.tts_model.set_lora_enabled(enabled)
    
    def get_lora_state_dict(self) -> dict:
        return self.tts_model.get_lora_state_dict()
    
    @property
    def lora_enabled(self) -> bool:
        return self.tts_model.lora_config is not None
