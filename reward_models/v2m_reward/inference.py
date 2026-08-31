from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .schema import JudgeScores, JUDGE_JSON_SCHEMA
from .prompts import SYSTEM_PROMPT, build_user_message_first_turn

log = logging.getLogger(__name__)


# Order matters — must match JudgeRawScores field order and the schema property
# order used by the constrained decoder.
_AXES: Tuple[str, str, str] = (
    "musicality",
    "text_music_alignment",
    "video_music_alignment",
)


@dataclass
class JudgeRawScores:
    musicality: float
    text_music_alignment: float
    video_music_alignment: float
    parsed_ok: bool
    raw_text: str = ""
    # Per-axis 5-way distribution when score_mode="expected_value", else empty.
    score_probs: List[List[float]] = field(default_factory=list)


class JudgeModelInference:

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        max_new_tokens: int = 96,
        n_video_frames: int = 4,
        score_mode: str = "expected_value",
        nan_on_failure: bool = True,
        parse_failure_score: float = float("nan"),
        max_memory: Optional[dict] = None,
    ):
        from transformers import AutoProcessor

        if score_mode not in ("expected_value", "argmax"):
            raise ValueError(f"score_mode must be expected_value|argmax, got {score_mode!r}")

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.n_video_frames = n_video_frames
        self.score_mode = score_mode
        self.nan_on_failure = nan_on_failure
        self.parse_failure_score = parse_failure_score
        # qwen_omni_utils treats NumPy audio arrays as 16 kHz; resample explicitly
        # so 48 kHz training audio reaches the judge at the expected duration.
        self.audio_sample_rate = 16000
        self._family = self._detect_family(model_id)

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

        load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": device}
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        log.info("loading judge %s on %s (family=%s, 4bit=%s, score_mode=%s)",
                 model_id, device, self._family, load_in_4bit, score_mode)
        self.model = self._load_model(model_id, load_kwargs)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        if hasattr(self.model, "disable_talker"):
            try:
                self.model.disable_talker()
                log.info("disabled Qwen-Omni talker (text-only judging)")
            except Exception as e:
                log.warning("disable_talker() failed: %s", e)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self._tokenizer = getattr(self.processor, "tokenizer", None) or self.processor
        # Per-prompt video cache: map video_path -> (audios, images, videos) from
        # process_mm_info, valid only within one score_prompt_group call.
        self._video_cache: dict = {}

        # Constrained-decode artifacts (built lazily on first EV call).
        self._lfe_processor = None
        self._digit_token_ids: Optional[List[int]] = None  # token id for "1".."5"

    @staticmethod
    def _load_model(model_id: str, load_kwargs: dict):
        family = JudgeModelInference._detect_family(model_id)

        # Qwen-Omni: go directly to its explicit class. The Auto cascade triggers
        # multiple parallel HF shard downloads from all ranks simultaneously, which
        # causes race-condition "file not found" errors on multi-GPU runs. The
        # explicit class hits the HF cache in one shot and is safe under parallel load.
        if family == "qwen_omni":
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration  # type: ignore
                log.info("loading Qwen2_5OmniForConditionalGeneration from %s", model_id)
                return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Qwen-Omni model {model_id!r}: {e}"
                ) from e

        # MiniCPM-o: transformers' check_imports scans the remote modeling file and
        # flags 'stepaudio2' (a TTS-only dep inside a try/except) as missing because
        # get_imports strips try/except blocks before scanning. We bypass check_imports
        # by loading the snapshot directly via importlib — stepaudio2 is only needed
        # for init_tts() which the judge never calls.
        if family == "minicpm_o":
            return JudgeModelInference._load_minicpm_o_direct(model_id, load_kwargs)

        # Generic fallback for unknown families: try Auto classes in order.
        # ImportError is included because trust_remote_code models can raise it
        # when their modeling file imports an unavailable package.
        import transformers
        candidates = []
        for name in ("AutoModelForImageTextToText",
                     "AutoModelForVision2Seq",
                     "AutoModelForCausalLM",
                     "AutoModel"):
            cls = getattr(transformers, name, None)
            if cls is not None:
                candidates.append((name, cls))
        last_err = None
        for name, cls in candidates:
            try:
                log.info("trying %s.from_pretrained(%s)", name, model_id)
                return cls.from_pretrained(model_id, **load_kwargs)
            except (ValueError, KeyError, OSError, ImportError) as e:
                last_err = e
                continue
        raise RuntimeError(
            f"could not find an HF model class for {model_id!r}. Last error: {last_err}"
        )

    @staticmethod
    def _load_minicpm_o_direct(model_id: str, load_kwargs: dict):
        import sys
        import types
        import transformers

        stub = types.ModuleType("stepaudio2")
        stub.Token2wav = None
        injected = "stepaudio2" not in sys.modules
        if injected:
            sys.modules["stepaudio2"] = stub
        try:
            log.info("loading MiniCPM-o %s via AutoModelForCausalLM", model_id)
            return transformers.AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        finally:
            if injected:
                sys.modules.pop("stepaudio2", None)

    @staticmethod
    def _detect_family(model_id: str) -> str:
        s = model_id.lower()
        if "qwen" in s and "omni" in s:
            return "qwen_omni"
        if "minicpm" in s:
            return "minicpm_o"
        warnings.warn(f"unknown judge family for {model_id!r}; defaulting to qwen_omni")
        return "qwen_omni"

    # ------------------------------------------------------------ single-turn
    @torch.no_grad()
    def score_one(
        self,
        video_path: str,
        audio_waveform: torch.Tensor,
        sample_rate: int,
        text_caption: str,
    ) -> JudgeRawScores:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": self._build_user_content(
                build_user_message_first_turn(text_caption),
                video_path,
                audio_waveform,
                sample_rate,
            )},
        ]
        return self._generate_and_score(messages)

    # --------------------------------------------------- group-per-prompt
    @torch.no_grad()
    def score_prompt_group(
        self,
        video_path: str,
        text_caption: str,
        audios: List[torch.Tensor],
        sample_rate: int,
    ) -> List[JudgeRawScores]:
        if not audios:
            return []
        self._video_cache.clear()
        try:
            return [
                self.score_one(
                    video_path=video_path,
                    audio_waveform=a,
                    sample_rate=sample_rate,
                    text_caption=text_caption,
                )
                for a in audios
            ]
        finally:
            self._video_cache.clear()

    # ----------------------------------------------------------- shared helpers
    def _build_user_content(
        self,
        user_text: str,
        video_path: str,
        audio: torch.Tensor,
        sample_rate: int,
    ):
        if self._family == "minicpm_o":
            return self._build_user_content_minicpm(user_text, video_path, audio, sample_rate)
        # Qwen-Omni: per-element nframes/max_pixels dict format.
        return [
            {"type": "text", "text": user_text},
            {
                "type": "video",
                "video": video_path,
                "nframes": int(self.n_video_frames),
                "min_pixels": 28 * 28 * 16,
                "max_pixels": 448 * 448,
            },
            {
                "type": "audio",
                "audio": self._prepare_audio_for_model(audio, sample_rate),
                "sampling_rate": self.audio_sample_rate,
            },
        ]

    def _build_user_content_minicpm(
        self,
        user_text: str,
        video_path: str,
        audio: torch.Tensor,
        sample_rate: int,
    ):
        frames = self._get_video_frames_pil(video_path)
        audio_np = self._prepare_audio_for_model(audio, sample_rate)
        # chat() parses content: PIL.Image → image placeholder, np.ndarray → audio placeholder
        return frames + [audio_np, user_text]

    def _get_video_frames_pil(self, video_path: str):
        cached = self._video_cache.get(video_path)
        if cached is not None:
            return cached
        try:
            import torchvision.io as tvio
            vframes, _, info = tvio.read_video(video_path, pts_unit="sec", output_format="TCHW")
            n = len(vframes)
            if n == 0:
                raise ValueError("no frames decoded")
            indices = [int(round(i * (n - 1) / max(self.n_video_frames - 1, 1)))
                       for i in range(self.n_video_frames)]
            from PIL import Image as PILImage
            frames = [PILImage.fromarray(vframes[i].permute(1, 2, 0).numpy()) for i in indices]
        except Exception as e:
            log.warning("video decode failed for %s: %s; using blank frame", video_path, e)
            from PIL import Image as PILImage
            frames = [PILImage.new("RGB", (224, 224), (0, 0, 0))] * self.n_video_frames
        self._video_cache[video_path] = frames
        return frames

    def _prepare_audio_for_model(self, audio: torch.Tensor, sample_rate: int):
        wav = audio.detach().to(torch.float32).cpu()
        if int(sample_rate) != self.audio_sample_rate:
            import torchaudio.functional as F
            wav = F.resample(wav, int(sample_rate), self.audio_sample_rate)
        return wav.contiguous().numpy()

    def _process_mm_cached(self, messages: list, video_path: Optional[str]):
        from qwen_omni_utils import process_mm_info
        cached_videos = self._video_cache.get(video_path) if video_path else None

        if cached_videos is None:
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            if video_path is not None:
                self._video_cache[video_path] = videos
        else:
            # Re-run only audio extraction (cheap) by stripping the video block.
            stripped = []
            for m in messages:
                if isinstance(m.get("content"), list):
                    new_content = [c for c in m["content"] if c.get("type") != "video"]
                    stripped.append({**m, "content": new_content})
                else:
                    stripped.append(m)
            audios, images, _ = process_mm_info(stripped, use_audio_in_video=False)
            videos = cached_videos
        return audios, images, videos

    def _build_inputs_minicpm(self, messages: list):
        import numpy as np
        from PIL import Image as PILImage

        images: List = []
        audios: List = []
        audio_parts: List = []
        copy_msgs = []

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                content = [content]
            cur_parts = []
            for c in content:
                if isinstance(c, dict):
                    # Qwen-style content dicts shouldn't reach here, but handle gracefully
                    cur_parts.append(c.get("text", ""))
                elif isinstance(c, PILImage.Image):
                    images.append(c)
                    cur_parts.append("(<image>./</image>)")
                elif isinstance(c, np.ndarray):
                    audios.append(c)
                    audio_parts.append(i)
                    cur_parts.append("(<audio>./</audio>)")
                else:
                    cur_parts.append(str(c))
            copy_msgs.append({"role": role, "content": "\n".join(cur_parts)})

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            [prompt],
            [images],
            [audios],
            [audio_parts],
            return_tensors="pt",
            sampling_rate=self.audio_sample_rate,
            use_image_id=False,   # video/omni mode: no image_id tokens
        )
        inputs.pop("image_sizes", None)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def _build_inputs(self, messages: list, video_path: Optional[str]):
        if self._family == "minicpm_o":
            return self._build_inputs_minicpm(messages)
        # qwen_omni
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = self._process_mm_cached(messages, video_path)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def _gen_kwargs(self):
        if self._family == "qwen_omni":
            return dict(
                generation_mode="text",
                thinker_max_new_tokens=self.max_new_tokens,
                thinker_do_sample=False,
            )
        # minicpm_o: pass tokenizer so model.generate() can build terminators
        return dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            tokenizer=self.processor.tokenizer,
        )

    @staticmethod
    def _unpack_generate(out):
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "sequences"):
            return out.sequences
        return out

    def _video_path_from_messages(self, messages: list) -> Optional[str]:
        for m in messages:
            if isinstance(m.get("content"), list):
                for c in m["content"]:
                    if c.get("type") == "video":
                        v = c.get("video")
                        if isinstance(v, str):
                            return v
        return None

    # ------------------------------------------------------------- two paths
    def _generate_and_score(self, messages: list) -> JudgeRawScores:
        video_path = self._video_path_from_messages(messages)
        inputs = self._build_inputs(messages, video_path)
        if self.score_mode == "expected_value":
            return self._generate_expected_value(inputs)
        return self._generate_argmax(inputs)

    def _generate_argmax(self, inputs: dict) -> JudgeRawScores:
        if self._family == "minicpm_o":
            out = self.model.generate(**inputs, **self._gen_kwargs(), decode_text=False)
            # decode_text=False → returns raw outputs object with .sequences
            out_ids = out.sequences
            gen_ids = out_ids  # MiniCPM-o sequences contain only completion tokens
            text = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            return self._parse(text)
        out = self.model.generate(**inputs, **self._gen_kwargs())
        out_ids = self._unpack_generate(out)
        prompt_len = inputs["input_ids"].shape[1]
        # Defensive: some HF / Qwen-Omni versions return only the completion.
        gen_ids = out_ids[:, prompt_len:] if out_ids.shape[1] > prompt_len else out_ids
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return self._parse(text)

    # ---------------------- expected-value (constrained) path -----------------
    def _ensure_constrained(self):
        if self._lfe_processor is not None:
            return
        # lm-format-enforcer 0.11.x imports PreTrainedTokenizerBase from
        # transformers.tokenization_utils, but newer transformers exports it
        # only from tokenization_utils_base. Patch the alias before importing.
        import transformers.tokenization_utils as _tu
        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
            _tu.PreTrainedTokenizerBase = _PTB
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        # Build a callable usable as `prefix_allowed_tokens_fn` in HF generate.
        parser = JsonSchemaParser(JUDGE_JSON_SCHEMA)
        self._lfe_processor = build_transformers_prefix_allowed_tokens_fn(
            self._tokenizer, parser
        )
        # Cache the token ids for the digits 1..5. Most BPE tokenizers have a
        # single token for each ASCII digit; we assert that and bail otherwise.
        digit_ids = []
        for d in ("1", "2", "3", "4", "5"):
            ids = self._tokenizer.encode(d, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"tokenizer encodes digit {d!r} into {len(ids)} tokens; "
                    "expected_value path needs single-token digits"
                )
            digit_ids.append(ids[0])
        self._digit_token_ids = digit_ids

    def _generate_expected_value(self, inputs: dict) -> JudgeRawScores:
        self._ensure_constrained()
        # We need per-step logits to read the 5-way digit distribution at the
        # positions where the FSM forces a digit.
        gen_kwargs = self._gen_kwargs()
        gen_kwargs.update(dict(output_scores=True))
        if self._family == "minicpm_o":
            # _decode() forces return_dict_in_generate=True; output_scores passes through **kwargs
            out = self.model.generate(
                **inputs,
                prefix_allowed_tokens_fn=self._lfe_processor,
                **gen_kwargs,
                decode_text=False,
            )
            out_ids = out.sequences
            scores = out.scores if hasattr(out, "scores") else None
            gen_ids = out_ids  # completion-only
            text = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        else:
            # Qwen-Omni: thinker accepts prefix_allowed_tokens_fn under standard generate API.
            gen_kwargs["return_dict_in_generate"] = True
            out = self.model.generate(
                **inputs,
                prefix_allowed_tokens_fn=self._lfe_processor,
                **gen_kwargs,
            )
            if hasattr(out, "sequences"):
                out_ids = out.sequences
                scores = out.scores  # tuple of [B, vocab] tensors, one per gen step
            else:
                out_ids = self._unpack_generate(out)
                scores = None
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[:, prompt_len:] if out_ids.shape[1] > prompt_len else out_ids
            text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        if scores is None:
            return self._parse(text)

        # Walk gen_ids[0]; whenever the emitted token is one of the digit ids
        # AND the matching JSON axis hasn't been filled yet, read the 5-way
        # distribution at that step. Constrained decoding guarantees the digit
        # tokens appear in axis order.
        digit_set = set(self._digit_token_ids)
        seq = gen_ids[0].tolist()
        per_axis_probs: List[List[float]] = []
        per_axis_value: List[float] = []
        for step, tok in enumerate(seq):
            if tok in digit_set and len(per_axis_probs) < 3:
                step_logits = scores[step][0]  # [vocab]
                digit_logits = step_logits[self._digit_token_ids]  # [5]
                p = torch.softmax(digit_logits.float(), dim=-1)
                ev = float((p * torch.tensor([1, 2, 3, 4, 5], dtype=p.dtype, device=p.device)).sum())
                per_axis_probs.append([float(x) for x in p.tolist()])
                per_axis_value.append(ev)
            if len(per_axis_probs) == 3:
                break

        if len(per_axis_value) != 3:
            log.warning("EV path got %d digit positions, expected 3; raw=%r",
                        len(per_axis_value), text[:200])
            return self._failure(text)

        return JudgeRawScores(
            musicality=per_axis_value[0],
            text_music_alignment=per_axis_value[1],
            video_music_alignment=per_axis_value[2],
            parsed_ok=True,
            raw_text=text,
            score_probs=per_axis_probs,
        )

    # ------------------------------------------------------------- parse + fail
    def _parse(self, raw_text: str) -> JudgeRawScores:
        s = raw_text.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end <= start:
            return self._failure(raw_text)
        try:
            data = json.loads(s[start:end + 1])
            scores = JudgeScores(**data)
            return JudgeRawScores(
                musicality=float(scores.musicality),
                text_music_alignment=float(scores.text_music_alignment),
                video_music_alignment=float(scores.video_music_alignment),
                parsed_ok=True,
                raw_text=raw_text,
            )
        except Exception as e:
            log.warning("judge JSON validation failed: %s; raw=%r", e, raw_text[:200])
            return self._failure(raw_text)

    def _failure(self, raw_text: str) -> JudgeRawScores:
        v = float("nan") if self.nan_on_failure else self.parse_failure_score
        return JudgeRawScores(
            musicality=v,
            text_music_alignment=v,
            video_music_alignment=v,
            parsed_ok=False,
            raw_text=raw_text,
        )
