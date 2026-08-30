#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import soundfile as sf

from core import VIBE


# -----------------------------
# Validators
# -----------------------------

def validate_file_exists(file_path: str, file_type: str = "file") -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} '{file_path}' does not exist")
    return path


def validate_output_path(output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_ranges(args, parser):
    if not (0.1 <= args.cfg_value <= 10.0):
        parser.error("--cfg-value must be between 0.1 and 10.0")

    if not (1 <= args.inference_timesteps <= 100):
        parser.error("--inference-timesteps must be between 1 and 100")

    if args.lora_r <= 0:
        parser.error("--lora-r must be a positive integer")

    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be a positive integer")

    if not (0.0 <= args.lora_dropout <= 1.0):
        parser.error("--lora-dropout must be between 0.0 and 1.0")


# -----------------------------
# Model loading
# -----------------------------

def load_model(args) -> VIBE:
    print("Loading VIBE model...", file=sys.stderr)

    zipenhancer_path = getattr(args, "zipenhancer_path", None) or os.environ.get(
        "ZIPENHANCER_MODEL_PATH", None
    )

    # Build LoRA config if provided
    lora_config = None
    lora_weights_path = getattr(args, "lora_path", None)
    if lora_weights_path:
        from model.vibe_ttm import LoRAConfig

        lora_config = LoRAConfig(
            enable_lm=not args.lora_disable_lm,
            enable_dit=not args.lora_disable_dit,
            enable_proj=args.lora_enable_proj,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

        print(
            f"LoRA config: r={lora_config.r}, alpha={lora_config.alpha}, "
            f"lm={lora_config.enable_lm}, dit={lora_config.enable_dit}, proj={lora_config.enable_proj}",
            file=sys.stderr,
        )

    # Load local model if specified
    if args.model_path:
        try:
            model = VIBE(
                voxcpm_model_path=args.model_path,
                zipenhancer_model_path=zipenhancer_path,
                enable_denoiser=not args.no_denoiser,
                lora_config=lora_config,
                lora_weights_path=lora_weights_path,
            )
            print("Model loaded (local).", file=sys.stderr)
            return model
        except Exception as e:
            print(f"Failed to load model (local): {e}", file=sys.stderr)
            sys.exit(1)

    # Load from Hugging Face Hub
    try:
        model = VIBE.from_pretrained(
            hf_model_id=args.hf_model_id,
            load_denoiser=not args.no_denoiser,
            zipenhancer_model_id=zipenhancer_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
        )
        print("Model loaded (from_pretrained).", file=sys.stderr)
        return model
    except Exception as e:
        print(f"Failed to load model (from_pretrained): {e}", file=sys.stderr)
        sys.exit(1)


# -----------------------------
# Commands
# -----------------------------

def cmd_clone(args):
    if not args.text:
        sys.exit("Error: Please provide --text for synthesis")

    if not args.prompt_audio or not args.prompt_text:
        sys.exit("Error: Voice cloning requires both --prompt-audio and --prompt-text")

    prompt_audio_path = validate_file_exists(args.prompt_audio, "reference audio file")
    output_path = validate_output_path(args.output)

    model = load_model(args)

    audio_array = model.generate(
        text=args.text,
        prompt_wav_path=str(prompt_audio_path),
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=args.denoise,
    )

    sf.write(str(output_path), audio_array, model.tts_model.sample_rate)

    duration = len(audio_array) / model.tts_model.sample_rate
    print(f"Saved audio to: {output_path} ({duration:.2f}s)", file=sys.stderr)


def cmd_synthesize(args):
    if not args.text:
        sys.exit("Error: Please provide --text for synthesis")

    output_path = validate_output_path(args.output)
    model = load_model(args)

    audio_array = model.generate(
        text=args.text,
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=False,
    )

    sf.write(str(output_path), audio_array, model.tts_model.sample_rate)

    duration = len(audio_array) / model.tts_model.sample_rate
    print(f"Saved audio to: {output_path} ({duration:.2f}s)", file=sys.stderr)


def cmd_batch(args):
    input_file = validate_file_exists(args.input, "input file")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        sys.exit("Error: Input file is empty")

    model = load_model(args)

    prompt_audio_path = None
    if args.prompt_audio:
        prompt_audio_path = str(validate_file_exists(args.prompt_audio, "reference audio file"))

    success_count = 0

    for i, text in enumerate(texts, 1):
        try:
            audio_array = model.generate(
                text=text,
                prompt_wav_path=prompt_audio_path,
                prompt_text=args.prompt_text,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=args.denoise and prompt_audio_path is not None,
            )

            output_file = output_dir / f"output_{i:03d}.wav"
            sf.write(str(output_file), audio_array, model.tts_model.sample_rate)

            duration = len(audio_array) / model.tts_model.sample_rate
            print(f"Saved: {output_file} ({duration:.2f}s)", file=sys.stderr)
            success_count += 1

        except Exception as e:
            print(f"Failed on line {i}: {e}", file=sys.stderr)

    print(f"\nBatch finished: {success_count}/{len(texts)} succeeded", file=sys.stderr)


# -----------------------------
# Parser
# -----------------------------

def _build_unified_parser():
    parser = argparse.ArgumentParser(
        description="VIBE CLI - voice cloning, direct TTS, and batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  voxcpm --text "Hello world" --output out.wav
  voxcpm --text "Hello" --prompt-audio ref.wav --prompt-text "hi" --output out.wav --denoise
  voxcpm --input texts.txt --output-dir ./outs
        """,
    )

    # Mode selection
    parser.add_argument("--input", "-i", help="Input text file (batch mode only)")
    parser.add_argument("--output-dir", "-od", help="Output directory (batch mode only)")
    parser.add_argument("--text", "-t", help="Text to synthesize (single or clone mode)")
    parser.add_argument("--output", "-o", help="Output audio file path (single or clone mode)")

    # Prompt
    parser.add_argument("--prompt-audio", "-pa", help="Reference audio file path (clone mode)")
    parser.add_argument("--prompt-text", "-pt", help="Reference text corresponding to the audio")
    parser.add_argument("--denoise", action="store_true", help="Enable prompt speech enhancement")

    # Generation parameters
    parser.add_argument("--cfg-value", type=float, default=2.0,
                        help="CFG guidance scale (float, recommended 0.5–5.0, default: 2.0)")
    parser.add_argument("--inference-timesteps", type=int, default=10,
                        help="Inference steps (int, 1–100, default: 10)")
    parser.add_argument("--normalize", action="store_true", help="Enable text normalization")

    # Model loading
    parser.add_argument("--model-path", type=str, help="Local VIBE model path")
    parser.add_argument("--hf-model-id", type=str, default="openbmb/VoxCPM1.5",
                        help="Hugging Face repo id (default: openbmb/VoxCPM1.5)")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for Hub downloads")
    parser.add_argument("--local-files-only", action="store_true", help="Disable network access")
    parser.add_argument("--no-denoiser", action="store_true", help="Disable denoiser model loading")
    parser.add_argument("--zipenhancer-path", type=str,
                        help="ZipEnhancer model id or local path (or env ZIPENHANCER_MODEL_PATH)")

    # LoRA
    parser.add_argument("--lora-path", type=str, help="Path to LoRA weights")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (positive int, default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (positive int, default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout rate (0.0–1.0, default: 0.0)")
    parser.add_argument("--lora-disable-lm", action="store_true", help="Disable LoRA on LM layers")
    parser.add_argument("--lora-disable-dit", action="store_true", help="Disable LoRA on DiT layers")
    parser.add_argument("--lora-enable-proj", action="store_true", help="Enable LoRA on projection layers")

    return parser


# -----------------------------
# Entrypoint
# -----------------------------

def main():
    parser = _build_unified_parser()
    args = parser.parse_args()

    # Validate ranges
    validate_ranges(args, parser)

    # Mode conflict checks
    if args.input and args.text:
        parser.error("Use either batch mode (--input) or single mode (--text), not both.")

    # Batch mode
    if args.input:
        if not args.output_dir:
            parser.error("Batch mode requires --output-dir")
        return cmd_batch(args)

    # Single mode
    if not args.text or not args.output:
        parser.error("Single-sample mode requires --text and --output")

    # Clone mode
    if args.prompt_audio or args.prompt_text:
        return cmd_clone(args)

    # Direct synthesis
    return cmd_synthesize(args)


if __name__ == "__main__":
    main()
