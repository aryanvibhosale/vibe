#!/usr/bin/env python3

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torchaudio


def _free_judge(judge):
    try:
        del judge.model.model
    except Exception:
        pass
    del judge
    gc.collect()
    torch.cuda.empty_cache()


def _bench_new(args, wav, sr, mode, label, results):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from reward_models.v2m_reward import JudgeRewardTensor
    print(f"\n=== [{label}] loading new judge (score_mode={mode}) ===")
    t0 = time.time()
    judge = JudgeRewardTensor(
        model_id=args.model_id,
        device=args.device,
        load_in_4bit=not args.no_4bit,
        score_mode=mode,
        n_video_frames=args.n_frames,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[{label}] load: {time.time()-t0:.1f}s")

    # warmup (excluded from timing, primes kernels & FSM)
    _ = judge.score_prompt_group(
        video_path=args.video, text=args.text,
        waveforms=[wav.clone()], sample_rate=sr,
    )

    t0 = time.time()
    grp = judge.score_prompt_group(
        video_path=args.video, text=args.text,
        waveforms=[wav.clone() for _ in range(args.group_size)],
        sample_rate=sr,
    )
    dt = time.time() - t0
    print(f"[{label}] G={args.group_size} wall={dt:.2f}s, per-call={dt/args.group_size:.3f}s")
    print(f"[{label}] scores=\n{grp}")
    results[label] = dict(wall=dt, per_call=dt/args.group_size, scores=grp.tolist())
    _free_judge(judge)


def _bench_legacy(args, wav, sr, mode, chunk, label, results):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from reward_models.v2m_reward._inference_legacy import JudgeModelInference as LegacyInf
    print(f"\n=== [{label}] loading legacy judge (mode={mode}) ===")
    t0 = time.time()
    inf = LegacyInf(
        model_id=args.model_id,
        device=args.device,
        load_in_4bit=not args.no_4bit,
        n_video_frames=args.n_frames,
        max_new_tokens=args.max_new_tokens,
        group_scoring_mode=mode,
        group_chunk_size=chunk,
    )
    print(f"[{label}] load: {time.time()-t0:.1f}s")

    # warmup
    _ = inf.score_prompt_group(
        video_path=args.video, text_caption=args.text,
        audios=[wav.clone()], sample_rate=sr,
    )

    t0 = time.time()
    raw = inf.score_prompt_group(
        video_path=args.video, text_caption=args.text,
        audios=[wav.clone() for _ in range(args.group_size)],
        sample_rate=sr,
    )
    dt = time.time() - t0
    scores = [[r.musicality, r.text_music_alignment, r.video_music_alignment] for r in raw]
    print(f"[{label}] G={args.group_size} wall={dt:.2f}s, per-call={dt/args.group_size:.3f}s")
    print(f"[{label}] scores=\n{torch.tensor(scores)}")
    results[label] = dict(wall=dt, per_call=dt/args.group_size, scores=scores)

    try:
        del inf.model
    except Exception:
        pass
    del inf
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--skip", default="", help="comma-separated labels to skip")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    wav, sr = torchaudio.load(args.audio)
    # qwen_omni_utils only accepts mono. The wrapper does this for new code,
    # but legacy is driven directly so we mono-flatten here for both paths.
    if wav.dim() == 2:
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    print(f"[audio] shape={tuple(wav.shape)} sr={sr}")

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    results: dict = {}

    plan = [
        ("legacy_independent",        lambda: _bench_legacy(args, wav, sr, "independent", 3, "legacy_independent", results)),
        ("legacy_chunked",            lambda: _bench_legacy(args, wav, sr, "chunked_single_turn_multi_candidate", 3, "legacy_chunked", results)),
        ("new_argmax",                lambda: _bench_new(args, wav, sr, "argmax", "new_argmax", results)),
        ("new_expected_value",        lambda: _bench_new(args, wav, sr, "expected_value", "new_expected_value", results)),
    ]

    for label, fn in plan:
        if label in skip:
            print(f"[skip] {label}")
            continue
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {label}: {e!r}")
            import traceback; traceback.print_exc()

    print("\n=========== SUMMARY ===========")
    for label, r in results.items():
        print(f"{label:28s} wall={r['wall']:6.2f}s  per-call={r['per_call']:.3f}s")
    if "legacy_independent" in results and "new_argmax" in results:
        a, b = results["legacy_independent"]["wall"], results["new_argmax"]["wall"]
        print(f"\nspeedup new_argmax vs legacy_independent: {a/b:.2f}x")
    if "legacy_independent" in results and "new_expected_value" in results:
        a, b = results["legacy_independent"]["wall"], results["new_expected_value"]["wall"]
        print(f"speedup new_expected_value vs legacy_independent: {a/b:.2f}x")


if __name__ == "__main__":
    main()
