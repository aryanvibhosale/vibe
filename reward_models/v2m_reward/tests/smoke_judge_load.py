#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to a video file")
    p.add_argument("--audio", required=True, help="path to a wav file (acts as the 'generated' candidate)")
    p.add_argument("--text", required=True, help="text caption that conditioned generation")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-4bit", action="store_true", help="load in bf16 instead of int4 (needs more VRAM)")
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--group-size", type=int, default=3,
                   help="Number of repeated candidates to score in score_prompt_group.")
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Max text tokens for the judge JSON response.")
    p.add_argument("--group-scoring-mode", default="independent",
                   choices=[
                       "chunked_single_turn_multi_candidate",
                       "single_turn_multi_candidate",
                       "independent",
                       "multi_turn",
                   ])
    p.add_argument("--group-chunk-size", type=int, default=3,
                   help="Chunk size for chunked_single_turn_multi_candidate mode.")
    p.add_argument("--max-memory-gib", type=float, default=None,
                   help="Hard cap on GPU VRAM (GiB) for model loading. "
                        "E.g. --max-memory-gib 38 reserves 6 GiB for activations.")
    args = p.parse_args()

    # Make repo root importable
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from reward_models.v2m_reward import JudgeRewardTensor

    if not torch.cuda.is_available():
        print("CUDA not available; this smoke test needs a GPU.", file=sys.stderr)
        sys.exit(1)

    # Print per-GPU free memory so the caller can pick an unoccupied device.
    n_gpus = torch.cuda.device_count()
    print(f"[gpus] {n_gpus} device(s) visible:")
    for i in range(n_gpus):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i}  {free/1e9:.1f} GiB free / {total/1e9:.1f} GiB total")
    if not Path(args.video).is_file():
        print(f"video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.audio).is_file():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Build max_memory dict if the caller supplied a cap.
    max_memory = None
    if args.max_memory_gib is not None:
        dev_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        max_memory = {dev_idx: f"{int(args.max_memory_gib)}GiB", "cpu": "48GiB"}
        print(f"[mem-cap] {max_memory}")

    print(f"[load] {args.model_id} on {args.device} (4bit={not args.no_4bit})")
    t0 = time.time()
    judge = JudgeRewardTensor(
        model_id=args.model_id,
        device=args.device,
        dtype="bfloat16",
        load_in_4bit=not args.no_4bit,
        n_video_frames=args.n_frames,
        max_new_tokens=args.max_new_tokens,
        group_scoring_mode=args.group_scoring_mode,
        group_chunk_size=args.group_chunk_size,
        max_memory=max_memory,
    )
    print(f"[load] done in {time.time()-t0:.1f}s")
    print(f"[mem]  GPU alloc: {torch.cuda.memory_allocated(args.device)/1e9:.2f} GB, "
          f"reserved: {torch.cuda.memory_reserved(args.device)/1e9:.2f} GB")

    wav, sr = torchaudio.load(args.audio)
    print(f"[audio] shape={tuple(wav.shape)} sr={sr}")

    # ---- score_one ----
    t0 = time.time()
    out1 = judge.score_tensor(
        waveform=wav, sample_rate=sr,
        video_path=args.video, text=args.text,
    )
    dt1 = time.time() - t0
    print(f"\n[score_one] {dt1:.2f}s -> {out1}")
    for raw in judge.last_raw_outputs():
        print(f"[score_one/raw] {raw!r}")
    for k, v in out1.items():
        assert 1.0 <= v <= 5.0 or v == 3.0, f"unexpected score {k}={v}"

    # ---- score_prompt_group with the same wav repeated G times ----
    t0 = time.time()
    G = args.group_size
    grp = judge.score_prompt_group(
        video_path=args.video, text=args.text,
        waveforms=[wav.clone() for _ in range(G)],
        sample_rate=sr,
    )
    dt_grp = time.time() - t0
    print(f"\n[score_prompt_group] G={G} took {dt_grp:.2f}s; per-call ≈ {dt_grp/G:.2f}s")
    print(f"[score_prompt_group] scores=\n{grp}")
    assert grp.shape == (G, 3)
    if not torch.allclose(grp, grp[0].expand_as(grp)):
        print("[score_prompt_group/consistency] WARNING: repeated identical audio received non-identical scores")
    else:
        print("[score_prompt_group/consistency] identical repeated audio received identical scores")
    one_vec = torch.tensor([
        out1["musicality"],
        out1["text_music_alignment"],
        out1["video_music_alignment"],
    ], dtype=torch.float32)
    if not torch.allclose(grp, one_vec.expand_as(grp)):
        print("[score_prompt_group/calibration] WARNING: repeated grouped scores differ from score_one")
    else:
        print("[score_prompt_group/calibration] repeated grouped scores match score_one")
    raw_outputs = judge.last_raw_outputs()
    unique_raw_outputs = list(dict.fromkeys(raw_outputs))
    for i, raw in enumerate(unique_raw_outputs):
        print(f"[score_prompt_group/raw_unique {i}] {raw!r}")
    if len(unique_raw_outputs) != len(raw_outputs):
        print(f"[score_prompt_group/raw] {len(raw_outputs)} candidates shared "
              f"{len(unique_raw_outputs)} unique raw output(s)")

    print(f"\n[stats] {judge.stats()}")
    print("[ok] smoke test passed")


if __name__ == "__main__":
    main()
