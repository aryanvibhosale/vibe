import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train.py"


@pytest.fixture(scope="module")
def script_module():
    # The script needs the src package on the path before import; if its deps
    # aren't available we skip rather than fail.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("train_v2m", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"could not import training script (likely missing optional deps): {e}")
    return mod


class _FakeTokenizer:
    def __call__(self, text):
        return [ord(c) % 100 for c in text[:16]] or [0]


def _write_manifest(tmp_path, rows):
    p = tmp_path / "v2m.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_v2m_dataset_reads_text_video_skips_no_video(tmp_path, script_module):
    rows = [
        {"text": "a", "video": "/abs/a.mp4", "audio": "/abs/a.wav"},
        {"text": "b", "video": "/abs/b.mp4"},                             # no audio key, fine
        {"text": "c", "audio": "/abs/c.wav"},                             # no video → skipped
        {"video": "/abs/d.mp4", "audio": "/abs/d.wav"},                   # no text → skipped
    ]
    manifest = _write_manifest(tmp_path, rows)
    ds = script_module.V2MPromptDataset(manifest, _FakeTokenizer())
    assert len(ds) == 2
    item0 = ds[0]
    assert item0["text"] == "a"
    assert item0["video_path"] == "/abs/a.mp4"
    assert isinstance(item0["text_ids"], list)


def test_v2m_dataset_supports_caption_alias(tmp_path, script_module):
    rows = [{"caption": "x", "video": "/v/x.mp4"}]
    manifest = _write_manifest(tmp_path, rows)
    ds = script_module.V2MPromptDataset(manifest, _FakeTokenizer())
    assert len(ds) == 1
    assert ds[0]["text"] == "x"


def test_collate_v2m_carries_video_paths(tmp_path, script_module):
    rows = [
        {"text": "a", "video": "/a.mp4"},
        {"text": "b", "video": "/b.mp4"},
        {"text": "c", "video": "/c.mp4"},
    ]
    manifest = _write_manifest(tmp_path, rows)
    ds = script_module.V2MPromptDataset(manifest, _FakeTokenizer())
    batch = script_module.collate_prompts_v2m([ds[i] for i in range(3)])
    assert batch["text"] == ["a", "b", "c"]
    assert batch["video_path"] == ["/a.mp4", "/b.mp4", "/c.mp4"]
    assert len(batch["text_ids"]) == 3


def test_ttm_dataset_unaffected(tmp_path, script_module):
    rows = [{"caption": "ttm-only"}, {"text": "another"}]
    manifest = _write_manifest(tmp_path, rows)
    ds = script_module.PromptDataset(manifest, _FakeTokenizer())
    assert len(ds) == 2
    item = ds[0]
    assert item["text"] == "ttm-only"
    assert "video_path" not in item
