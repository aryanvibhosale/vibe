## 🎵 VIBE: Video Instruction-aligned Background Music Generation

[![Project Page](https://img.shields.io/badge/Project%20Page-VIBE-blue)](https://vibe-text-video-to-music-generation.github.io/vibe/) [![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE) [![Built on](https://img.shields.io/badge/Built%20on-VoxCPM-orange)](https://github.com/OpenBMB/VoxCPM/) [![Base LM](https://img.shields.io/badge/Backbone-MiniCPM--4-purple)](https://huggingface.co/openbmb/MiniCPM4-0.5B)

<div align="center">

**Background music for video that follows an explicit text instruction.**

🎧 [Listen to samples on the project page](https://vibe-text-video-to-music-generation.github.io/vibe/)

</div>

> **Research code.** This is the code as it was run for the paper. Every path is
> hardcoded to the cluster it was developed on, and there is no packaging or
> installer. Read [`docs/INSTALL.md`](docs/INSTALL.md) before running anything, and
> see [Risks and limitations](#️-risks-and-limitations) for what is and is not
> finished.

## Overview

VIBE generates background music for a video that follows an explicit text
instruction — not just "music that fits this video", but music that fits *and*
does what the user asked. It extends a tokenizer-free, continuous-latent music
model with reinforcement learning against a multimodal LLM judge and rule-based
verifiable rewards, so the generated music aligns with both the video and the
instruction.

Rather than converting audio to discrete tokens, VIBE models music in a
continuous latent space: a **multimodal semantic LM** produces planning latents,
a **RITE** (Residual Integration Transformer Encoder) stack refines them, and a
local diffusion transformer — the **LocDiT** — denoises under
conditional flow matching before a 48 kHz stereo VAE decodes to audio. Video
conditioning enters through a CLIP-based encoder, and **semantic routing** merges
the video and instruction signals into a single conditioning stream.

Built on [VoxCPM](https://github.com/OpenBMB/VoxCPM) (Apache-2.0), adapted from
speech to music. See [`NOTICE`](NOTICE) for the full provenance.

### 🚀 Key Features

- **Instruction-Aligned Music Generation** — VIBE conditions on a video *and* a
  free-form text instruction, so tempo, key, mood, and instrumentation can be
  requested explicitly rather than inferred from the video alone.
- **Reinforcement Learning from a Multimodal Judge** — GRPO against
  Qwen2.5-Omni-7B, which watches the video and listens to the generated music,
  scoring musicality, text-music alignment, and video-music alignment. Scores
  come from constrained digit-logit decoding, so there are no JSON parse failures.
- **Hard Verifiable Rewards** — rule-based tempo/BPM and musical-key agreement,
  measured on the generated audio with librosa/essentia and scored against
  targets parsed from the instruction. Deterministic, CPU-only, no learned
  parameters, and immune to reward hacking.
- **Continuous Latents, No Audio Tokenizer** — 48 kHz stereo output through a
  music VAE, avoiding the quality ceiling and codebook artifacts of discrete
  tokenization.

## Quick Start

### 🔧 Installation

There is no PyPI package — this is research code, run from the repository.

```bash
conda env create -f environment.yml && conda activate vibe
export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"
```

Then work through [`docs/INSTALL.md`](docs/INSTALL.md): you need base models from
Hugging Face, the 48 kHz music VAE, CMI-RewardBench, and a pass over the
hardcoded paths.

### 1. Model Downloads

Fetched from Hugging Face at runtime, not redistributed here:

| Model | Used for |
|---|---|
| [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B) | Base language model |
| [`openbmb/VoxCPM1.5`](https://huggingface.co/openbmb/VoxCPM1.5) | Base speech/audio model |
| [`Qwen/Qwen2.5-Omni-7B`](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | Multimodal LLM judge reward |
| [`openbmb/MiniCPM-o-4.5`](https://huggingface.co/openbmb/MiniCPM-o-4.5) | Alternative judge |
| [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Video captioning (data prep) |
| [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | Video encoder backbone |

Obtained separately: the **SongBloom / Stable Audio music VAE** checkpoint and
**CMI-RewardBench** (CC-BY-NC-4.0, required for the CMI reward). See
[`docs/INSTALL.md`](docs/INSTALL.md) and [`NOTICE`](NOTICE).

### 2. Inference

Both entry points take a checkpoint directory as `--ckpt_dir`. Substitute your own
paths for the values marked `<...>`; [`docs/INSTALL.md`](docs/INSTALL.md) explains
what each one is.

**Video-to-music** — video + instruction in, music out.

> **Released checkpoint.** The published V2M checkpoint already has its RL
> LoRA folded into its base weights (`W += (alpha/r)·B·A`, see its
> `FOLD_MANIFEST.json`), so it loads as a plain model. Point `--ckpt_dir` at it
> and **omit `--lora_weights_path`** — passing the adapter as well would apply it
> a second time. `--lora_weights_path` exists for pairing a base checkpoint
> with a separate adapter, which the released checkpoint does not need.

```bash
export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"

python scripts/infer_v2m.py \
    --ckpt_dir      /path/to/vibe_checkpoint \
    --baselm_path   /path/to/MiniCPM4-0.5B \
    --audiovae_path /path/to/music_vae_cache \
    --text          "An ambient electronic track at 125 BPM in B Minor." \
    --video_path    /path/to/input_video.mp4 \
    --output        v2m_out.wav
```

**Text-to-music** — instruction only, no video.

```bash
python scripts/infer_ttm.py \
    --ckpt_dir      <ttm_checkpoint> \
    --baselm_path   <MiniCPM4-0.5B> \
    --audiovae_path <music_vae_cache> \
    --text          "An ambient electronic track at 125 BPM in B Minor, opening on a Bm7 chord and shifting to E Major." \
    --output        ttm_out.wav
```

Tune quality with `--cfg_value`
(default 2.0) and `--inference_timesteps` (TTM 20, V2M 10).

### 3. Training

```bash
python scripts/train.py --config_path train_configs/train.yaml
```

## 📚 Documentation

- **[Installation Guide](docs/INSTALL.md)** — environment, base models, the music VAE,
  CMI-RewardBench, and re-rooting the hardcoded paths

## 🗂️ Repository Layout

```
VIBE/
├── src/                     Policy model — fork of OpenBMB VoxCPM (see NOTICE)
│   ├── model/               vibe_v2m, vibe_v2m_rl (video), vibe_ttm (LoRA + RL)
│   ├── modules/             locdit, locenc, audiovae, video_encoder, minicpm4
│   ├── training/            data_video, packers_video, accelerator_ds
│   ├── utils/               text normalization helpers
│   └── core.py, core_video.py, cli.py
├── scripts/
│   ├── train.py             Video-to-music training
│   ├── infer_v2m.py         Video-to-music inference
│   └── infer_ttm.py         Text-to-music inference
├── train_configs/           Training configs
│   └── deepspeed/           ds_zero2_rl.json (DeepSpeed ZeRO-2)
├── reward_models/           Reward models behind one configurator interface
│   ├── base.py              RewardConfigurator + type registry
│   ├── v2m_reward/          Multimodal LLM judge (Qwen-Omni) + tests
│   ├── soft_reward/         CMI-RM adapter (weights external, see NOTICE)
│   └── hard_reward/         Rule-based tempo/key verifiable rewards
├── data/                    Example manifests (schema documentation, not data)
├── checkpoints/             Run outputs and logs
├── voxcpm_license/          Upstream LICENSE and README for the fork
└── docs/                    INSTALL.md
```

Not included, by design: model checkpoints, training data, the CMI reward model,
and demo media. See [`NOTICE`](NOTICE).

## ⚠️ Risks and limitations

Read these before building on the code.

- **Nothing is pip-installable.** All entry points use `sys.path.insert`.
  `requirements.txt` is upstream's, unmodified, and unused.
- **CMI-RM weights are CC-BY-NC-4.0**, so they are
  not redistributed here. See [`docs/INSTALL.md`](docs/INSTALL.md) §3.
- **Example manifests are schema documentation, not data.**
  `data/*.jsonl` point at media that is not redistributed.
- **Generated audio may be unexpected or contain artifacts**, particularly for
  instructions far from the training distribution. This code is released for
  research purposes; we do not recommend production use without further testing.
- **Respect the rights in your inputs.** Generating music conditioned on video you
  do not have rights to, or producing music that imitates a specific artist's
  protected work, may infringe. We recommend clearly marking generated audio as
  AI-generated.

## 📝 TO-DO List

- [x] Multimodal LLM judge reward with constrained digit-logit decoding
- [x] Hard verifiable tempo/key rewards
- [x] Unified reward-model configurator interface
- [x] Video-conditioned policy during V2M RL
- [ ] Release checkpoints and a packaged installer

## 📄 License

Code is released under the [Apache-2.0](LICENSE) license.

Third-party components carry their own licenses, and some are **not**
redistributed here. In particular the CMI reward model is CC-BY-NC-4.0
(non-commercial). Read [`NOTICE`](NOTICE) before commercial use.

## 🙏 Acknowledgments

We extend our sincere gratitude to the following works and resources:

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) for the tokenizer-free
  continuous-latent backbone this work extends
- [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) for the language model foundation
- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) for the multimodal judge
- [MuQ / MuQ-MuLan](https://github.com/tencent-ailab/MuQ) for the CMI reward model lineage
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the flow-matching LocDiT
- [DAC](https://github.com/descriptinc/descript-audio-codec) for the Audio VAE backbone

## 📚 Citation

If you find this work helpful, please consider citing VIBE and the upstream
VoxCPM paper:

```bib
@software{vibe2026,
  title  = {VIBE: Video Instruction-aligned Background Music Generation},
  author = {The VIBE Authors},
  year   = {2026},
  url    = {https://vibe-text-video-to-music-generation.github.io/vibe/},
}

@article{voxcpm2025,
  title   = {VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning},
  author  = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2509.24650},
  year    = {2025},
}
```

The author list and repository URL in [`CITATION.cff`](CITATION.cff) are still
placeholders — fill them in before publishing.
