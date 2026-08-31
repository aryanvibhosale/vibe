#!/usr/bin/env python3
import sys
from pathlib import Path

# scripts/<file>.py -> repo root is one level up.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import contextlib
import random
from typing import Dict, Optional

import argbind
import torch

from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import signal
import os
import torch.distributed as dist
from tqdm import tqdm

# Video-conditioned model and data pipeline, imported unconditionally: this
# script has no text-only path, so there is no branch to resolve at call time.
from model.vibe_v2m import VIBEVideo2Music
from model.vibe_v2m import LoRAConfig
from training import (
    Accelerator,
    DeepSpeedAccelerator,
    BatchProcessorForVideoInput as BatchProcessor,
    TrainingTracker,
    build_dataloader_for_video as build_dataloader,
    load_video_audio_text_datasets as load_audio_text_datasets,
)

# os.environ["HF_HOME"] = HF_CACHE_DIR
# os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
# os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will use pytorch format", file=sys.stderr)


def calculate_total_steps(num_samples, batch_size, grad_accum_steps, epochs, world_size=8):
    print(f"Calculating total steps with num_samples={num_samples}, batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, epochs={epochs}, world_size={world_size}")
    steps_per_epoch = max(1, (num_samples) // (batch_size * world_size))  # Ensure at least 1 step per epoch
    total_steps = (steps_per_epoch * epochs) // grad_accum_steps
    print(f"Calculated steps_per_epoch={steps_per_epoch}, total_steps={total_steps} for num_samples={num_samples}, batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, epochs={epochs}, world_size={world_size}")
    return total_steps


@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    baselm_path: str,
    train_manifest: str,
    start_from_weight: str = "",
    audiovae_path: str = "",
    val_manifest: str = "",
    sample_rate: int = 16_000,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 4,
    num_iters: int = 100_000,
    epochs: int = 10,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_ratio: float = 0.1,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    tensorboard: str = "",
    lambdas: Dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
    take_video_input: bool = True,    # must stay True; kept so V2M configs load unchanged
    null_text_prob: float = 0.0,
    use_stop_loss: bool = False,
    filter_data_by_duration: bool = True,
    ttm_warmup_ratio: float = 0.0,
    patch_size: int = 4,
    token_budget_batching: bool = False,
    token_budget_shuffle_window: int = 1000,
    # Distribution options (for LoRA checkpoints)
    hf_model_id: str = "",   # HuggingFace model ID (e.g., "openbmb/VoxCPM1.5")
    distribute: bool = False, # If True, save hf_model_id as base_model; otherwise save pretrained_path
    # DeepSpeed
    deepspeed_config: str = "",  # Path to train_configs/deepspeed/ds_zero2_rl.json; if empty, use plain DDP
):
    if not take_video_input:
        raise ValueError(
            "scripts/train.py is video-to-music only; set take_video_input: true in "
            "the config. Use scripts/train.py for text-to-music SFT."
        )

    _ = config_path

    # Validate distribution options
    if lora is not None and distribute and not hf_model_id:
        raise ValueError("hf_model_id is required when distribute=True")

    use_deepspeed = bool(deepspeed_config)
    if use_deepspeed:
        accelerator = DeepSpeedAccelerator(ds_config=deepspeed_config)
    else:
        accelerator = Accelerator(amp=True)

    save_dir = Path(save_path)
    tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"

    # Only create directories on rank 0 to avoid race conditions
    if accelerator.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()  # Wait for directory creation

    writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.rank == 0 else None
    tracker = TrainingTracker(writer=writer, log_file=str(save_dir / "train.log"), rank=accelerator.rank)

    # Updated Model Loading with new flags
    base_model = VIBEVideo2Music.from_local(
        path=pretrained_path, 
        patch_size = patch_size,
        baselm_path=baselm_path, 
        audiovae_path=audiovae_path, 
        optimize=False, 
        training=True, 
        lora_config=LoRAConfig(**lora) if lora else None,
        use_stop_loss = use_stop_loss,
        start_from_weight=start_from_weight
    )
    tokenizer = base_model.text_tokenizer
    sample_rate = base_model.audio_vae.sample_rate
    # Classifier-free guidance on the text stream: with probability
    # null_text_prob a sample's caption is replaced by the empty string, so the
    # model learns to lean on video alone.
    empty_text_ids = tokenizer("") if null_text_prob > 0 else None

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        sample_rate=sample_rate,
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    map_workers = max(1, num_workers)
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=map_workers)

    # Save original validation texts for audio generation display
    val_texts = None
    if val_ds is not None:
        val_texts = list(val_ds["text"])  # Save original texts
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=map_workers)


    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    num_train_samples = len(train_ds)

    # ------------------------------------------------------------------ #
    # Optional: filter samples by estimated token count to avoid OOM
    # ------------------------------------------------------------------ #
    if max_batch_tokens and max_batch_tokens > 0 and filter_data_by_duration:
        from training.data import compute_sample_lengths

        # Updated FPS calculation for Songbloom
        audio_vae_fps = base_model.audio_vae.sample_rate / (base_model.audio_vae.downsampling_ratio)
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.rank == 0:
            tracker.print(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        max_batch_tokens=max_batch_tokens if token_budget_batching else 0,
        shuffle_window=token_budget_shuffle_window,
    )

    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )
    

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device
    )
    
    # Save audio_vae for audio generation in validation
    audio_vae_for_gen = base_model.audio_vae
    del base_model.audio_vae

    # Freeze video encoder before handing model to any wrapper
    print("Freezing video encoder parameters...")
    base_model.video_encoder.requires_grad_(False)

    num_iters = calculate_total_steps(len(train_ds), batch_size, grad_accum_steps, epochs, world_size=accelerator.world_size)
    warmup_steps = int(warmup_ratio * num_iters)
    total_training_steps = num_iters

    if use_deepspeed:
        # DeepSpeed initializes optimizer & scheduler internally from the config.
        # prepare_model returns (engine, optimizer_proxy, scheduler_proxy).
        model, optimizer, scheduler = accelerator.prepare_model(
            base_model,
            optimizer=None,  # DS builds AdamW from config
            lr=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            total_steps=total_training_steps,
            batch_size_per_gpu=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
    else:
        model = accelerator.prepare_model(base_model, find_unused_parameters=True)
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    del base_model  # avoid keeping a second reference to the unwrapped module
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    if accelerator.rank == 0:
        trainable_layers = {name.split(".")[0] for name, param in model.named_parameters() if param.requires_grad}
        tracker.print(f"Trainable layers: {trainable_layers}")
    

    # Try to load checkpoint and resume training
    start_step = 0
    accelerator.barrier()
    if use_deepspeed:
        start_step = load_checkpoint_ds(model, save_dir)
        print(f"Loaded checkpoint (DeepSpeed), resuming from step {start_step}")
    else:
        start_step, scheduler_const = load_checkpoint(model, optimizer, scheduler=None, save_dir=save_dir, warmup_steps=warmup_steps, total_training_steps=total_training_steps)
        print(f"Loaded checkpoint, resuming from step {start_step}")
        if scheduler_const is not None:
            scheduler = scheduler_const
            del scheduler_const
    accelerator.barrier()
    
    print(f"Calculated total steps: {num_iters}, warmup steps: {warmup_steps}, max steps: {total_training_steps}")
    
    # Broadcast start_step to all processes
    if hasattr(accelerator, 'all_reduce'):
        start_step_tensor = torch.tensor(start_step, device=accelerator.device)
        accelerator.all_reduce(start_step_tensor, op=dist.ReduceOp.MAX)
        start_step = int(start_step_tensor.item())
    
    if start_step > 0 and accelerator.rank == 0:
        tracker.print(f"Resuming training from step {start_step}")

    # Resume tracker for signal handler to read current step
    resume = {"step": start_step}

    def _signal_handler(signum, frame, _model=model, _optim=optimizer, _sched=scheduler, _save_dir=save_dir, _pretrained=pretrained_path, _hf_id=hf_model_id, _dist=distribute, _resume=resume, _rank=accelerator.rank, _train_loader=train_loader, _use_ds=use_deepspeed):
        try:
            cur_step = int(_resume.get("step", start_step))
        except Exception:
            cur_step = start_step
        print(f"Signal {signum} received. Saving checkpoint at step {cur_step} ...", file=sys.stderr)
        try:
            if _use_ds:
                save_checkpoint_ds(_model, _save_dir, cur_step, _pretrained, _hf_id, _dist, dataloader=_train_loader, rank=_rank)
            else:
                save_checkpoint(_model, _optim, _sched, _save_dir, cur_step, _pretrained, _hf_id, _dist, dataloader=_train_loader, rank=_rank)
            print("Checkpoint saved. Exiting.", file=sys.stderr)
        except Exception as e:
            print(f"Error saving checkpoint on signal: {e}", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    print("After signal term handler setup")

    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)
    
    print(f"Starting training loop from step {start_step} to {num_iters} with grad_accum_steps={grad_accum_steps}")

    # --- resume dataloader position ---
    loaded_dl_state = False
    if start_step > 0 and hasattr(train_loader, "load_state_dict"):
        ckpt_dir = save_dir / f"step_{start_step:07d}"
        candidate_paths = [
            ckpt_dir / f"dataloader_rank{accelerator.rank}.pth",
            ckpt_dir / "dataloader.pth",
            save_dir / "latest" / f"dataloader_rank{accelerator.rank}.pth",
            save_dir / "latest" / "dataloader.pth",
        ]
        for dl_state_path in candidate_paths:
            if dl_state_path.exists():
                try:
                    train_loader.load_state_dict(torch.load(dl_state_path, map_location="cpu"))
                    loaded_dl_state = True
                    if accelerator.rank == 0:
                        tracker.print(f"[ckpt] Restored dataloader state from {dl_state_path}")
                except Exception as e:
                    tracker.print(f"[ckpt] Warning: failed to load dataloader state from {dl_state_path}: {e}")
                break

    print(f"Loaded dataloader state: {loaded_dl_state}")

    if start_step > 0:
        if loaded_dl_state:
            train_iter = iter(train_loader)
            if accelerator.rank == 0:
                tracker.print("[resume] restored dataloader state (no manual skip).")
        else:
            consumed_batches = start_step * grad_accum_steps
            steps_per_epoch = len(train_loader) 
            data_epoch = consumed_batches // steps_per_epoch
            within_epoch_skip = consumed_batches % steps_per_epoch

            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(data_epoch)

            # train_iter = iter(train_loader)
            # for _ in range(within_epoch_skip):
            #     try:
            #         next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_loader)
            #         break

            if accelerator.rank == 0:
                tracker.print(f"[resume] data_epoch={data_epoch}, skipped_batches={within_epoch_skip}/{steps_per_epoch}")

    def get_next_batch():
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    
    ttm_warmup_steps = ttm_warmup_ratio * num_iters

    _patch_size_for_log = unwrapped_model.config.patch_size

    with tracker.live():
        for step in tqdm(range(start_step, num_iters), desc=f"Training rank {accelerator.rank}"):
            # update resume step so signal handler can save current progress
            resume["step"] = step
            tracker.step = step
            optimizer.zero_grad(set_to_none=True)

            loss_dict = {}
            for micro_step in range(grad_accum_steps):
                if step == ttm_warmup_steps and micro_step == 0 and accelerator.rank == 0:
                    params = {name.split(".")[0] for name, param in model.named_parameters() if param.requires_grad}
                    tracker.print(f"[ttm_warmup] trainable top-level modules: {params}")

                    # unwrapped_model.unfreeze_ttm(model)
                    # unwrapped_model.video_encoder.requires_grad_(False)  # Unfreeze video encoder

                batch = get_next_batch()
                if null_text_prob > 0 and "text_ids" in batch:
                    batch["text_ids"] = [
                        empty_text_ids if random.random() < null_text_prob else tid
                        for tid in batch["text_ids"]
                    ]
                processed = batch_processor(batch)

                is_last_micro_step = (micro_step == grad_accum_steps - 1)
                sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            **processed,
                            progress=step / max(1, num_iters),
                        )

                    total_loss = 0.0
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            loss_value = value * weight / grad_accum_steps
                            total_loss = total_loss + loss_value
                            loss_dict[key] = value.detach()

                    accelerator.backward(total_loss)

            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=1e9)

            accelerator.step(optimizer)
            accelerator.update()
            if not use_deepspeed:
                scheduler.step()

            if step % log_interval == 0 or step == num_iters - 1:
                loss_values = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                current_lr = float(model.get_lr()[0]) if use_deepspeed else float(optimizer.param_groups[0]["lr"])
                loss_values["lr"] = current_lr
                epoch = (step * grad_accum_steps * batch_size) / max(1, num_train_samples)
                loss_values["epoch"] = float(epoch)
                loss_values["grad_norm"] = float(grad_norm)
                tracker.log_metrics(loss_values, split="train")
                if accelerator.rank == 0:
                    tracker.print(f"Step {step}/{num_iters} | patch_size={_patch_size_for_log} | lr={current_lr:.2e}")

            if val_loader is not None and (step % valid_interval == 0 or step == num_iters - 1) and step != 0:
                validate(model, val_loader, batch_processor, accelerator, tracker, lambdas,
                        writer=writer, step=step, val_ds=val_ds, audio_vae=audio_vae_for_gen,
                        sample_rate=sample_rate, val_texts=val_texts, tokenizer=tokenizer,
                        valid_interval=valid_interval)

            if (step % save_interval == 0) and step != 0:
                if use_deepspeed:
                    save_checkpoint_ds(model, save_dir, step, pretrained_path, hf_model_id, distribute, dataloader=train_loader, rank=accelerator.rank)
                else:
                    save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path, hf_model_id, distribute, dataloader=train_loader, rank=accelerator.rank)

    if use_deepspeed:
        save_checkpoint_ds(model, save_dir, num_iters, pretrained_path, hf_model_id, distribute, dataloader=train_loader, rank=accelerator.rank)
    else:
        save_checkpoint(model, optimizer, scheduler, save_dir, num_iters, pretrained_path, hf_model_id, distribute, dataloader=train_loader, rank=accelerator.rank)
    if writer:
        writer.close()


def validate(model, val_loader, batch_processor, accelerator, tracker, lambdas, 
             writer=None, step=0, val_ds=None, audio_vae=None, sample_rate=22050,
             val_texts=None, tokenizer=None, valid_interval=1000):
    import numpy as np
    from collections import defaultdict
    print("[VALIDATION] Sample rate used", sample_rate)
    model.eval()
    total_losses = []
    sub_losses = defaultdict(list)
    num_batches = 0
    max_val_batches = 10

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_val_batches:
                break
            processed = batch_processor(batch)
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )
            total = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    weighted_loss = lambdas.get(key, 1.0) * value
                    total += weighted_loss
                    sub_losses[key].append(value.detach())
            total_losses.append(total.detach())
            num_batches += 1

    if total_losses:
        mean_total_loss = torch.stack(total_losses).mean()
        accelerator.all_reduce(mean_total_loss)
        
        val_metrics = {"loss/total": mean_total_loss.item()}
        for key, values in sub_losses.items():
            mean_sub_loss = torch.stack(values).mean()
            accelerator.all_reduce(mean_sub_loss)
            val_metrics[key] = mean_sub_loss.item()
        
        tracker.log_metrics(val_metrics, split="val")
    
    # Generate sample audio for TensorBoard display
    if writer is not None and val_ds is not None and audio_vae is not None and accelerator.rank == 0:
        try:
            generate_sample_audio(model, val_ds, audio_vae, writer, step, accelerator, sample_rate,
                                 val_texts=val_texts, tokenizer=tokenizer, valid_interval=valid_interval,
                                 tracker=tracker)
        except Exception as e:
            tracker.print(f"[Warning] Failed to generate sample audio: {e}")
            import traceback
            import io
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            tracker.print(buf.getvalue())
    else:
        missing = []
        if writer is None: missing.append("writer")
        if val_ds is None: missing.append("val_ds")
        if audio_vae is None: missing.append("audio_vae")
        if missing and accelerator.rank == 0:
            tracker.print(f"[Warning] Skip audio generation: missing {', '.join(missing)}")
    
    model.train()


def compute_mel_spectrogram(audio_np, sample_rate, n_mels=128):
    import numpy as np
    import librosa
    audio_np = audio_np.flatten().astype(np.float32)
    mel = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_mels=n_mels, fmax=sample_rate // 2)
    return librosa.power_to_db(mel, ref=np.max)


def create_mel_figure(gen_audio_np, gen_mel, sample_rate, step=None, ref_audio_np=None, ref_mel=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa.display
    
    fmax = sample_rate // 2
    step_str = f" @ Step {step}" if step is not None else ""
    
    if ref_audio_np is not None and ref_mel is not None:
        fig, (ax_ref, ax_gen) = plt.subplots(2, 1, figsize=(12, 8))
        
        img_ref = librosa.display.specshow(ref_mel, sr=sample_rate, x_axis='time', y_axis='mel', fmax=fmax, cmap='viridis', ax=ax_ref)
        ax_ref.set_title(f'Reference (GT) - {len(ref_audio_np)/sample_rate:.2f}s{step_str}', fontsize=10, fontweight='bold', color='#28A745')
        plt.colorbar(img_ref, ax=ax_ref, format='%+2.0f dB', pad=0.02)
        
        img_gen = librosa.display.specshow(gen_mel, sr=sample_rate, x_axis='time', y_axis='mel', fmax=fmax, cmap='viridis', ax=ax_gen)
        ax_gen.set_title(f'Generated - {len(gen_audio_np)/sample_rate:.2f}s', fontsize=10, fontweight='bold', color='#DC3545')
        plt.colorbar(img_gen, ax=ax_gen, format='%+2.0f dB', pad=0.02)
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(gen_mel, sr=sample_rate, x_axis='time', y_axis='mel', fmax=fmax, cmap='viridis', ax=ax)
        ax.set_title(f'Generated - {len(gen_audio_np)/sample_rate:.2f}s{step_str}', fontsize=11, fontweight='bold')
        plt.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    
    plt.tight_layout()
    return fig


def normalize_audio(audio_np):
    import numpy as np
    max_val = np.abs(audio_np).max()
    return audio_np / max_val * 0.9 if max_val > 0 else audio_np


def generate_sample_audio(model, val_ds, audio_vae, writer, step, accelerator, sample_rate=22050, 
                          val_texts=None, tokenizer=None, pretrained_path=None, valid_interval=1000,
                          tracker=None):
    import numpy as np
    
    log = tracker.print if tracker else print
    num_samples = min(2, len(val_ds))
    log(f"[Audio] Starting audio generation for {num_samples} samples at step {step}")
    
    unwrapped_model = accelerator.unwrap(model)
    
    for i in range(num_samples):
        sample = val_ds[i]
        text = val_texts[i] if val_texts and i < len(val_texts) else "Hello, this is a test."
        
        # Load reference audio
        ref_audio_np = None
        try:
            if "audio" in sample and isinstance(sample["audio"], dict) and "array" in sample["audio"]:
                ref_audio_np = np.array(sample["audio"]["array"], dtype=np.float32)
                ref_sr = sample["audio"].get("sampling_rate", sample_rate)
                if ref_sr != sample_rate:
                    import torchaudio.functional as F
                    ref_audio_np = F.resample(torch.from_numpy(ref_audio_np).unsqueeze(0), ref_sr, sample_rate).squeeze(0).numpy()
                log(f"[Audio] Loaded reference audio for sample {i}: duration={len(ref_audio_np)/sample_rate:.2f}s")
        except Exception as e:
            log(f"[Warning] Failed to load reference audio: {e}")
        
        try:
            # Inference setup
            unwrapped_model.eval()
            unwrapped_model.to(torch.bfloat16)
            unwrapped_model.audio_vae = audio_vae.to(torch.float32)
            
            log(f"[Audio] Generating sample {i} with text: '{text[:50]}...'")
            with torch.no_grad():
                generated = unwrapped_model.generate(target_text=text, inference_timesteps=10, cfg_value=2.0)
            
            # Restore training setup
            unwrapped_model.to(torch.float32)
            unwrapped_model.audio_vae = None
            
            if generated is None or len(generated) == 0:
                log(f"[Warning] Generated audio is empty for sample {i}")
                continue
            
            # Process generated audio
            gen_audio_np = generated.cpu().float().numpy().flatten() if isinstance(generated, torch.Tensor) else generated
            gen_audio_np = normalize_audio(gen_audio_np)
            
            tag = f"val_sample_{step}_{i}"
            writer.add_audio(f"{tag}/generated_audio", gen_audio_np, global_step=step, sample_rate=sample_rate)
            log(f"[Audio] Generated audio for sample {i}: duration={len(gen_audio_np)/sample_rate:.2f}s")
            
            # Log reference audio
            if ref_audio_np is not None:
                writer.add_audio(f"{tag}/reference_audio", normalize_audio(ref_audio_np), global_step=step, sample_rate=sample_rate)
            
            # Generate mel spectrogram figure
            try:
                mel_gen = compute_mel_spectrogram(gen_audio_np, sample_rate)
                mel_ref = compute_mel_spectrogram(ref_audio_np, sample_rate) if ref_audio_np is not None else None
                fig = create_mel_figure(gen_audio_np, mel_gen, sample_rate, step, ref_audio_np, mel_ref)
                writer.add_figure(f"{tag}/mel_spectrogram", fig, global_step=step)
                log(f"[Audio] Created mel spectrogram figure for sample {i}")
            except Exception as e:
                log(f"[Warning] Failed to create mel spectrogram: {e}")
                
        except Exception as e:
            log(f"[Warning] Failed to generate audio for sample {i}: {e}")
            import traceback
            traceback.print_exc()


def load_checkpoint_ds(engine, save_dir: Path) -> int:
    step_folders = []
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_", 1)[1])
                # DeepSpeed writes its own sub-folder inside step_*/
                if (d / "ds_checkpoint").exists() or any(d.iterdir()):
                    step_folders.append((step, d))
            except Exception:
                pass

    if not step_folders:
        return 0

    resume_step, ckpt_dir = max(step_folders, key=lambda x: x[0])
    ds_tag = str(ckpt_dir / "ds_checkpoint")
    _, client_state = engine.load_checkpoint(str(save_dir), tag=f"step_{resume_step:07d}/ds_checkpoint")
    print(f"[ckpt-ds] Loaded DeepSpeed checkpoint from {ckpt_dir}", file=sys.stderr)
    return resume_step


def save_checkpoint_ds(
    engine,
    save_dir: Path,
    step: int,
    pretrained_path: str = None,
    hf_model_id: str = "",
    distribute: bool = False,
    dataloader=None,
    rank: Optional[int] = None,
):
    import shutil

    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step:07d}/ds_checkpoint"

    # engine.save_checkpoint is collective — all ranks must call it
    engine.save_checkpoint(str(save_dir), tag=tag)

    # Per-rank dataloader state
    if dataloader is not None and hasattr(dataloader, "state_dict"):
        folder = save_dir / f"step_{step:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        dl_name = f"dataloader_rank{rank}.pth" if rank is not None else "dataloader.pth"
        try:
            torch.save(dataloader.state_dict(), folder / dl_name)
        except Exception as e:
            print(f"Warning: failed to save dataloader state for rank {rank}: {e}", file=sys.stderr)

    if rank not in (None, 0):
        return

    # Copy config files so the checkpoint folder is self-contained
    if pretrained_path:
        folder = save_dir / f"step_{step:07d}"
        folder.mkdir(parents=True, exist_ok=True)
        pretrained_dir = Path(pretrained_path)
        for fname in ["config.json", "audiovae.pth", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]:
            src = pretrained_dir / fname
            if src.exists():
                try:
                    shutil.copy2(src, folder / fname)
                except PermissionError:
                    pass

    # Update latest symlink
    latest_link = save_dir / "latest"
    folder = save_dir / f"step_{step:07d}"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        os.symlink(str(folder), str(latest_link))
    except Exception:
        pass


def load_checkpoint(model, optimizer, scheduler = None, save_dir: Path = "", warmup_steps: int = 0, total_training_steps: int = 0):
    # Find all step_* directories
    step_folders = []
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_", 1)[1])
                step_folders.append((step, d))
            except Exception:
                pass

    if not step_folders:
        return 0, None

    # Pick max step
    resume_step, ckpt_dir = max(step_folders, key=lambda x: x[0])
    print(f"[ckpt] Loading from highest step folder: {ckpt_dir} (step {resume_step})", file=sys.stderr)

    unwrapped = model.module if hasattr(model, "module") else model
    lora_cfg = getattr(unwrapped, "lora_config", None)

    # ---- load model weights ----
    if lora_cfg is not None:
        lora_weights_path = ckpt_dir / "lora_weights.safetensors"
        if not lora_weights_path.exists():
            lora_weights_path = ckpt_dir / "lora_weights.ckpt"

        if not lora_weights_path.exists():
            raise FileNotFoundError(f"Missing LoRA weights in {ckpt_dir}")

        if lora_weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(lora_weights_path))
        else:
            ckpt = torch.load(lora_weights_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

        unwrapped.load_state_dict(state_dict, strict=False)
        print(f"[ckpt] Loaded LoRA weights: {lora_weights_path}", file=sys.stderr)
    else:
        model_path = ckpt_dir / "model.safetensors"
        if not model_path.exists():
            model_path = ckpt_dir / "pytorch_model.bin"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights in {ckpt_dir}")

        if model_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
        else:
            ckpt = torch.load(model_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

        unwrapped.load_state_dict(state_dict, strict=False)
        print(f"[ckpt] Loaded model weights: {model_path}", file=sys.stderr)

    scheduler_const = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        last_epoch=resume_step - 1,
    )

    # ---- load optimizer/scheduler ----
    optimizer_path = ckpt_dir / "optimizer.pth"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        print(f"[ckpt] Loaded optimizer: {optimizer_path}", file=sys.stderr)
    else:
        print(f"[ckpt] WARNING: optimizer state missing at {optimizer_path}", file=sys.stderr)

    if scheduler is not None:
        scheduler_path = ckpt_dir / "scheduler.pth"
        if scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            print(f"[ckpt] Loaded scheduler: {scheduler_path}", file=sys.stderr)
        else:
            print(f"[ckpt] WARNING: scheduler state missing at {scheduler_path}", file=sys.stderr)

    return resume_step, scheduler_const

def save_checkpoint(model, optimizer, scheduler, save_dir: Path, step: int, pretrained_path: str = None, hf_model_id: str = "", distribute: bool = False, dataloader=None, rank: Optional[int] = None):
    import shutil
    
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = "latest" if step == 0 else f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)

    # Save dataloader state (for exact resume) (per-rank)
    if dataloader is not None and hasattr(dataloader, "state_dict"):
        try:
            if rank is None:
                dl_name = "dataloader.pth"
            else:
                dl_name = f"dataloader_rank{rank}.pth"
            torch.save(dataloader.state_dict(), folder / dl_name)
        except Exception as e:
            print(f"Warning: failed to save dataloader state for rank {rank}: {e}", file=sys.stderr)
    
    # Only rank 0 (or no-rank single process) writes global model/optimizer/scheduler
    if rank not in (None, 0):
        return
    
    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    lora_cfg = unwrapped.lora_config
    
    if lora_cfg is not None:
        # LoRA finetune: save only lora_A/lora_B weights
        state_dict = {k: v.contiguous() for k, v in full_state.items() if "lora_" in k}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "lora_weights.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "lora_weights.ckpt")
        
        # Save LoRA config and base model path to a separate JSON file
        import json
        base_model_to_save = hf_model_id if distribute else (str(pretrained_path) if pretrained_path else None)
        lora_info = {
            "base_model": base_model_to_save,
            "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
        }
        with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(lora_info, f, indent=2, ensure_ascii=False)
    else:
        # Full finetune: save non-vae weights to model.safetensors
        state_dict = {k: v.contiguous() for k, v in full_state.items() if not k.startswith("audio_vae.")}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "model.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "pytorch_model.bin")
        
        # Copy config files from pretrained path
        if pretrained_path:
            pretrained_dir = Path(pretrained_path)
            files_to_copy = ["config.json", "audiovae.pth", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
            for fname in files_to_copy:
                src = pretrained_dir / fname
                if src.exists():
                    try:
                        shutil.copy2(src, folder / fname)
                    except PermissionError:
                        pass
    
    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")

    # Update (or create) a `latest` symlink pointing to the most recent checkpoint folder
    latest_link = save_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        os.symlink(str(folder), str(latest_link))
    except Exception:
        try:
            if latest_link.exists():
                if latest_link.is_dir():
                    shutil.rmtree(latest_link)
                else:
                    latest_link.unlink()
            shutil.copytree(folder, latest_link)
        except Exception:
            print(f"Warning: failed to update latest checkpoint link at {latest_link}", file=sys.stderr)


if __name__ == "__main__":
    from training.config import load_yaml_config

    args = argbind.parse_args()
    config_file = args.get("config_path")
    if config_file:
        yaml_args = load_yaml_config(config_file)
        train(**yaml_args)
    else:
        with argbind.scope(args):
            train()