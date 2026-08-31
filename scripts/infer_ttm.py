#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# scripts/<file>.py -> repo root is one level up.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import soundfile as sf

from core import VIBE


def parse_args():
    parser = argparse.ArgumentParser("VIBE full-finetune inference test (no LoRA)")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory (contains pytorch_model.bin, config.json, audiovae.pth, etc.)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Target text to synthesize",
    )
    parser.add_argument(
        "--prompt_audio",
        type=str,
        default="",
        help="Optional: reference audio path for voice cloning",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="",
        help="Optional: transcript of reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ft_test.wav",
        help="Output wav file path",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="CFG scale (default: 2.0)",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=20,
        help="Diffusion inference steps (default: 10)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=600,
        help="Max generation steps",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable text normalization",
    )
    parser.add_argument(
        "--baselm_path",
        type=str,
        default="",
        help="Path to the base LM model file",
    )
    parser.add_argument(
        "--audiovae_path",
        type=str,
        default="",
        help="Path to the AudioVAE model file",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model from checkpoint directory (no denoiser)
    print(f"[FT Inference] Loading model: {args.ckpt_dir}", file=sys.stderr)
    model = VIBE.from_pretrained(
        hf_model_id=args.ckpt_dir,
        baselm_path = f"{args.baselm_path}",
        audiovae_path = f"{args.audiovae_path}",
        load_denoiser=False,
        optimize=True,
        patch_size = 4,
    )

    # Run inference
    prompt_wav_path = args.prompt_audio if args.prompt_audio else None
    prompt_text = args.prompt_text if args.prompt_text else None

    print(f"[FT Inference] Synthesizing: text='{args.text}'", file=sys.stderr)
    if prompt_wav_path:
        print(f"[FT Inference] Using reference audio: {prompt_wav_path}", file=sys.stderr)
        print(f"[FT Inference] Reference text: {prompt_text}", file=sys.stderr)

    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )

    # Save audio
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[FT Inference] audio_np type: {type(audio_np)}, audio_np shape: {audio_np.shape}", file=sys.stderr)
    if audio_np.ndim == 2 and audio_np.shape[0] <= 4:  # likely channels, N
        audio_np = audio_np.astype("float32")  
        audio_np = audio_np.T
    sf.write(str(out_path), audio_np, model.tts_model.sample_rate)

    print(f"[FT Inference] Saved to: {out_path}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
