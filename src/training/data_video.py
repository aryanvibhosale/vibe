import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Video, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from model.vibe_v2m import VIBEConfig
from modules.audiovae import AudioVAE
from .packers_video import AudioFeatureProcessingPackerForVideoInput
from .data import TokenBudgetBatchSampler
from transformers import CLIPImageProcessor
import numpy as np
import sys


DEFAULT_TEXT_COLUMN  = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_VIDEO_COLUMN = "video"
DEFAULT_ID_COLUMN    = "dataset_id"


@argbind.bind()
def load_video_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    video_column: str = DEFAULT_VIDEO_COLUMN,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        
        if video_column not in ds.column_names:
            raise ValueError(f"Expected '{video_column}' column in manifest.")
        # We cast to Audio to ensure proper handling during training, 
        # but for length calculation we might need raw path or duration if available.
        # HF datasets usually don't compute duration automatically for 'Audio' column.
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        ds = ds.cast_column(video_column, Video())
        
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if video_column != DEFAULT_VIDEO_COLUMN:
            ds = ds.rename_column(video_column, DEFAULT_VIDEO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)
        if dataset_id_column and dataset_id_column in ds.column_names:
            if dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: int = 25,
    patch_size: int = 1,
) -> List[int]:
    # Batch access columns - much faster than per-item access
    text_ids_list = ds["text_ids"]
    text_lens = [len(t) for t in text_ids_list]
    
    has_duration = "duration" in ds.column_names
    if has_duration:
        durations = ds["duration"]
    else:
        # Fallback: need to compute from audio (slow, but unavoidable without duration column)
        durations = []
        for i in range(len(ds)):
            audio = ds[i][DEFAULT_AUDIO_COLUMN]
            durations.append(len(audio["array"]) / float(audio["sampling_rate"]))
    
    # Vectorized length computation
    lengths = []
    for text_len, duration in zip(text_lens, durations):
        t_vae = math.ceil(float(duration) * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)
        total_len = text_len + t_seq + 2
        lengths.append(total_len)

    return lengths


class HFVIBEDatasetForVideoInput(TorchDataset):

    def __init__(self, dataset: Dataset, n_video_frames: int = 8, video_processor_name: str = "openai/clip-vit-base-patch32"):
        self.dataset = dataset
        self.n_video_frames = n_video_frames
        self.clip_processor = CLIPImageProcessor.from_pretrained(video_processor_name)

    def __len__(self):
        return len(self.dataset)
    
    def __get_total_frames(self, video_reader) -> int:
        metadata = video_reader.get_metadata()
        duration = metadata['video']['duration'][0]
        fps = metadata['video']['fps'][0]
        return int(duration * fps)
    
    
    def __sample_frames(self, video_reader) -> List[torch.Tensor]:
        n_frames = self.__get_total_frames(video_reader)
        indices = set(np.linspace(0, n_frames - 1, self.n_video_frames, dtype=int))
        return [
            frame['data']
            for i, frame in enumerate(video_reader)
            if i in indices
        ]
        
        
    def __prep_video(self, frames: List[torch.Tensor]) -> torch.Tensor:
        return self.clip_processor(images = frames, return_tensors="pt")['pixel_values'].squeeze(0)  # (T, C, H, W)


    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        video = item[DEFAULT_VIDEO_COLUMN]
        return {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "video_input": self.__prep_video(self.__sample_frames(video)),
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        video_tensors = [sample["video_input"] for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]
        
        
        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        assert len(video_tensors) == len(batch), "Mismatch in batch size for video inputs. got {}, expected {}".format(len(video_tensors), len(batch))
        assert all(video_tensors[i].ndim == 4 for i in range(len(video_tensors))), "Each video input should have 4 dimensions (T, H, W, C)."
        assert all(video_tensors[i].shape[0] > 0 for i in range(len(video_tensors))), "Each video input should have at least one frame."

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "video_input": torch.stack(video_tensors),
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }


class BatchProcessorForVideoInput:

    def __init__(
        self,
        *,
        config: VIBEConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.packer = AudioFeatureProcessingPackerForVideoInput(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            n_video_frames = config.n_video_frames,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(self.device)
        video_input = batch["video_input"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)
        
        # SongBloom music VAE is stereo
        if audio_tokens.dim() == 2:
            audio_tokens = audio_tokens.unsqueeze(1).repeat(1, 2, 1)  # (2, T)
        

        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
        )
        return {
            "text_tokens": packed["text_tokens"],
            "text_mask": packed["text_mask"],
            "audio_feats": packed["audio_feats"],
            "audio_mask": packed["audio_mask"],
            "loss_mask": packed["loss_mask"],
            "position_ids": packed["position_ids"],
            "labels": packed["labels"],
            "video_input": video_input
        }


try:
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
except Exception:
    StatefulDataLoader = None
    StatefulDistributedSampler = None


def build_dataloader_for_video(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
    seed: int = 1234,
    max_batch_tokens: int = 0,
    max_batch_size: int = 0,
    shuffle_window: int = 1000,
    # kept for API compatibility
    length_bucket_batching: bool = False,
    bucket_size_multiplier: int = 100,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVIBEDatasetForVideoInput(hf_dataset)

    world_size = getattr(accelerator, "world_size", 1)
    rank = getattr(accelerator, "rank", 0)

    if StatefulDataLoader is None:
        raise RuntimeError("StatefulDataLoader not available (torchdata not installed?)")

    if max_batch_tokens and max_batch_tokens > 0:
        text_ids_list = hf_dataset["text_ids"]
        lengths = [len(t) for t in text_ids_list]

        batch_sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size if max_batch_size > 0 else batch_size,
            shuffle_window=shuffle_window,
            drop_last=drop_last,
            shuffle=True,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

        return StatefulDataLoader(
            torch_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=HFVIBEDatasetForVideoInput.collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        sampler = None
        shuffle = True

        if world_size > 1:
            if StatefulDistributedSampler is None:
                raise RuntimeError("StatefulDistributedSampler not available (torchdata not installed?)")
            sampler = StatefulDistributedSampler(
                torch_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=drop_last,
            )
            shuffle = False

        return StatefulDataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=HFVIBEDatasetForVideoInput.collate_fn,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
