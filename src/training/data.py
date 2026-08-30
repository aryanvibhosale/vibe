import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from model.vibe_ttm import VIBEConfig
from modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker


DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_ID_COLUMN = "dataset_id"


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
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
        # We cast to Audio to ensure proper handling during training, 
        # but for length calculation we might need raw path or duration if available.
        # HF datasets usually don't compute duration automatically for 'Audio' column.
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
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


class HFVIBEDataset(TorchDataset):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        return {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
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
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }


class BatchProcessor:

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
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(self.device)
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
            "labels": packed["labels"]
        }


try:
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
except Exception as e:
    StatefulDataLoader = None
    StatefulDistributedSampler = None


class TokenBudgetBatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        lengths: List[int],
        max_batch_tokens: int,
        max_batch_size: int = 0,
        shuffle_window: int = 1000,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 1234,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.lengths = lengths
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size  # 0 = unlimited
        self.shuffle_window = shuffle_window
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _build_batches(self) -> List[List[int]]:
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        n = len(self.lengths)
        if self.shuffle:
            perm = torch.randperm(n, generator=rng).tolist()
        else:
            perm = list(range(n))

        # Sort within shuffle windows so nearby indices have similar lengths
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len: int = 0

        for window_start in range(0, n, self.shuffle_window):
            window = perm[window_start: window_start + self.shuffle_window]
            window.sort(key=lambda i: self.lengths[i])

            for idx in window:
                sample_len = self.lengths[idx]
                # Skip samples that are already over budget on their own
                if sample_len > self.max_batch_tokens:
                    continue

                new_max = max(current_max_len, sample_len)
                projected_tokens = new_max * (len(current_batch) + 1)
                over_size = self.max_batch_size > 0 and len(current_batch) >= self.max_batch_size

                if current_batch and (projected_tokens > self.max_batch_tokens or over_size):
                    batches.append(current_batch)
                    current_batch = []
                    current_max_len = 0

                current_batch.append(idx)
                current_max_len = max(current_max_len, sample_len)

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        # Shuffle batch order
        if self.shuffle:
            order = torch.randperm(len(batches), generator=rng).tolist()
            batches = [batches[i] for i in order]

        return batches

    def __iter__(self):
        batches = self._build_batches()
        rank_batches = batches[self.rank:: self.world_size]
        for batch in rank_batches:
            yield batch

    def __len__(self) -> int:
        batches = self._build_batches()
        return len(batches[self.rank:: self.world_size])


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
    sample_rate: int = 16_000,
    mono: bool = False,
    seed: int = 1234,
    max_batch_tokens: int = 0,
    max_batch_size: int = 0,
    shuffle_window: int = 1000,
    # kept for API compatibility but ignored when max_batch_tokens > 0
    length_bucket_batching: bool = False,
    bucket_size_multiplier: int = 100,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVIBEDataset(hf_dataset)

    world_size = getattr(accelerator, "world_size", 1)
    rank = getattr(accelerator, "rank", 0)

    if StatefulDataLoader is None:
        raise RuntimeError("StatefulDataLoader not available (torchdata not installed?)")

    if max_batch_tokens and max_batch_tokens > 0:
        # Build per-sample estimated sequence lengths (text + audio tokens)
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

        loader = StatefulDataLoader(
            torch_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=HFVIBEDataset.collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        # --- original fixed-batch-size path ---
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

        loader = StatefulDataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=HFVIBEDataset.collate_fn,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    return loader

