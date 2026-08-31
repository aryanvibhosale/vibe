#!/usr/bin/env python3

from typing import Dict, List, Optional

import torch

from .inference import JudgeModelInference, JudgeRawScores


class JudgeRewardTensor:

    #: axis order returned by score_prompt_group
    AXES = ("musicality", "text_music_alignment", "video_music_alignment")

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        score_mode: str = "expected_value",
        n_video_frames: int = 4,
        max_new_tokens: int = 96,
        nan_on_failure: bool = True,
        max_memory: Optional[dict] = None,
    ):
        self.model = JudgeModelInference(
            model_id=model_id,
            device=device,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            score_mode=score_mode,
            n_video_frames=n_video_frames,
            max_new_tokens=max_new_tokens,
            nan_on_failure=nan_on_failure,
            max_memory=max_memory,
        )
        self.device = device
        self.score_mode = score_mode
        self._stat_calls = 0
        self._stat_parse_failures = 0
        self._stat_prompts = 0
        self._last_raw_scores: List[JudgeRawScores] = []

    # ----------------- one-shot -----------------

    def score_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        video_path: str,
        text: str = "",
    ) -> Dict[str, float]:
        wav = self._ensure_mono(waveform)
        result: JudgeRawScores = self.model.score_one(
            video_path=video_path,
            audio_waveform=wav,
            sample_rate=sample_rate,
            text_caption=text,
        )
        self._last_raw_scores = [result]
        self._stat_calls += 1
        if not result.parsed_ok:
            self._stat_parse_failures += 1
        return {
            "musicality": result.musicality,
            "text_music_alignment": result.text_music_alignment,
            "video_music_alignment": result.video_music_alignment,
        }

    # ----------------- group (one prompt, G candidates) -----------------

    def score_prompt_group(
        self,
        video_path: str,
        text: str,
        waveforms: List[torch.Tensor],
        sample_rate: int,
    ) -> torch.Tensor:
        wavs_mono = [self._ensure_mono(w) for w in waveforms]
        results: List[JudgeRawScores] = self.model.score_prompt_group(
            video_path=video_path,
            text_caption=text,
            audios=wavs_mono,
            sample_rate=sample_rate,
        )
        self._last_raw_scores = list(results)
        self._stat_prompts += 1
        out = torch.empty(len(results), 3, dtype=torch.float32)
        for i, r in enumerate(results):
            self._stat_calls += 1
            if not r.parsed_ok:
                self._stat_parse_failures += 1
            out[i, 0] = r.musicality
            out[i, 1] = r.text_music_alignment
            out[i, 2] = r.video_music_alignment
        return out

    def score_prompt_group_scalar(
        self,
        video_path: str,
        text: str,
        waveforms: List[torch.Tensor],
        sample_rate: int,
    ) -> torch.Tensor:
        per_axis = self.score_prompt_group(video_path, text, waveforms, sample_rate)
        return per_axis.mean(dim=-1)

    # ----------------- diagnostics -----------------

    def stats(self) -> Dict[str, int]:
        return {
            "calls": self._stat_calls,
            "prompts": self._stat_prompts,
            "parse_failures": self._stat_parse_failures,
        }

    def reset_stats(self):
        self._stat_calls = 0
        self._stat_parse_failures = 0
        self._stat_prompts = 0

    def last_raw_outputs(self) -> List[str]:
        return [r.raw_text for r in self._last_raw_scores]

    def last_raw_scores(self) -> List[JudgeRawScores]:
        return list(self._last_raw_scores)

    def last_score_probs(self) -> List[List[List[float]]]:
        return [r.score_probs for r in self._last_raw_scores]

    # ----------------- helpers -----------------

    @staticmethod
    def _ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
        w = waveform.detach()
        if w.dim() == 3:
            w = w.squeeze(0)
        if w.dim() == 2:
            if w.shape[0] <= 4 and w.shape[1] > w.shape[0]:
                w = w.mean(dim=0) if w.shape[0] > 1 else w.squeeze(0)
            else:
                w = w.mean(dim=1) if w.shape[1] > 1 else w.squeeze(1)
        if w.dim() != 1:
            raise ValueError(f"expected 1D waveform after mono-flatten, got {w.shape}")
        return w.to(torch.float32).cpu()

    def __repr__(self) -> str:
        return (
            f"JudgeRewardTensor(model_id={self.model.model_id!r}, "
            f"device={self.device}, score_mode={self.score_mode})"
        )


#: uniform alias, peer of SoftRewardTensor / HardRewardTensor
V2MRewardTensor = JudgeRewardTensor
