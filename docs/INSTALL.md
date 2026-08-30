# Installation

## 1. Environment

```bash
conda env create -f environment.yml   # creates env "vibe"
conda activate vibe
```

## 2. Make the packages importable

Every entry point does a `sys.path` bootstrap
relative to its own location, so the scripts can be launched from any directory,
but the reward packages are imported by name and need the repository root on the
path:

```bash
export VIBE_ROOT="$(pwd)"
export PYTHONPATH="$VIBE_ROOT:$VIBE_ROOT/src:$PYTHONPATH"
```

## 3. Base models and checkpoints

None of these ship with the repository. Fetch from Hugging Face:

| Model | Used for |
|---|---|
| `openbmb/MiniCPM4-0.5B` | `baselm_path`, all configs |
| `Qwen/Qwen2.5-Omni-7B` | LLM judge reward |
| `openbmb/MiniCPM-o-4.5` | alternative judge |
| `Qwen/Qwen3-VL-8B-Instruct` | video captioning (data prep) |
| `openai/clip-vit-base-patch32` | video encoder backbone (auto-fetched at model init) |

The SongBloom / Stable Audio music VAE (`audiovae_path`) must be obtained
separately. Please follow instructions to download the songbloom vae checkpoints from the Songbloom github repository. It is a directory holding
`autoencoder_music_dsp1920.ckpt` and `stable_audio_1920_vae.json`.

### Checkpoint layout

A VIBE checkpoint directory needs exactly five files:

```
config.json              model configuration
model.safetensors        weights
tokenizer.json           }
tokenizer_config.json    }  LlamaTokenizerFast.from_pretrained(ckpt_dir)
special_tokens_map.json  }
```

Anything else in the directory is ignored by the loader. The audio VAE is **not**
read from the checkpoint — it always comes from `--audiovae_path`.

The released checkpoint has its RL LoRA already folded into the base weights, so
it loads as a plain model. Do not also pass `--lora_weights_path` against it, or
the adapter is applied twice.

## 4. CMI reward model (optional)

The CMI reward model is **not included** in this repository. Its weights are
licensed CC-BY-NC-4.0 (non-commercial), which cannot be redistributed under this
repository's Apache-2.0 license. See NOTICE §3.

It is needed only when `reward_model.type` is `cmi_rm` or `cmi_rm_v2m`. The
import is lazy — it happens inside `SoftRewardTensor._load_model`, not at module
level — so the reward packages import fine without it, and you get an explicit
`ImportError` only if you actually select a CMI reward.

To use it, obtain CMI-RewardBench and expose it as a top-level `cmi_rm` package:

```bash
git clone <CMI-RewardBench-repo-url> ../CMI-RewardBench
ln -s ../CMI-RewardBench/models/cmi-rm cmi_rm
export PYTHONPATH="$VIBE_ROOT/cmi_rm:$PYTHONPATH"
```

Then download the CMI-RM checkpoint (`model.safetensors` + `config.yaml`) and
point `cmi_checkpoint` / `cmi_config` in your config at it. `cmi_rm/` is in
`.gitignore`, so a local clone or symlink will not be committed back.

## 5. Fill in the paths

Every path in `train_configs/*.yaml` and in the README's inference commands is a
`/path/to/<artefact>` placeholder. Replace them with your own locations:

| Key | Points at |
|---|---|
| `baselm_path` | the `MiniCPM4-0.5B` snapshot directory |
| `audiovae_path` | the music VAE directory (see §3) |
| `pretrained_path` | the checkpoint to initialise from |
| `train_manifest` | a JSONL manifest — see `data/` for the schema |
| `save_path`, `tensorboard` / `tensorboard_dir` | run outputs |

`deepspeed_config` is already repo-relative and needs no change.

Manifest schema, one JSON object per line (see `data/*.jsonl`):

```json
{"video": "/path/to/video/example.mp4", "audio": "/path/to/audio/example.wav", "text": "..."}
```

`video` is required for video-to-music and omitted for text-to-music.

## 6. Verify

```bash
python -c "import reward_models.v2m_reward, reward_models.hard_reward; print('reward modules OK')"
python -c "import sys; sys.path.insert(0,'src'); from model.vibe_v2m import VIBEVideo2Music; print('model OK')"
python scripts/infer_v2m.py --help
```

Known issue: `pytest reward_models` reports 8 failures in `test_group_parse.py`
and `test_v2m_dataset.py`. Those tests target method and class names from before
the reward-model restructure and the trainer consolidation — they are stale, not
a regression.
