"""Download large model files from HF model repo at startup (HF Spaces mode)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


# Set this to your HF model repo, e.g. "Aliadenawil/atlas-models"
_MODEL_REPO = os.environ.get("MODEL_REPO", "Aliadenawil/atlas-models")

# Map: local path (relative to project root) → filename in the HF model repo
_FILES = {
    "models/joint_nlu/model_state.pt": "model_state.pt",
    "pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt": "embedding_model.ckpt",
    "pretrained_models/spkrec-ecapa-voxceleb/classifier.ckpt": "classifier.ckpt",
    "data/wakeword_models/wakeword_cnn.pt": "wakeword_cnn.pt",
    "data/voiceprints/voiceprint_store.pkl": "voiceprint_store.pkl",
}

_ROOT = Path(__file__).resolve().parent.parent


def download_models() -> None:
    """Download any missing model files from the HF model repo."""
    missing = [
        (local, remote)
        for local, remote in _FILES.items()
        if not (_ROOT / local).exists()
    ]
    if not missing:
        print("All model files present.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed — skipping model download.")
        return

    for local_rel, filename in missing:
        local_path = _ROOT / local_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename} from {_MODEL_REPO}...")
        try:
            downloaded = hf_hub_download(
                repo_id=_MODEL_REPO,
                filename=filename,
                local_dir_use_symlinks=False,
            )
            if not local_path.exists():
                shutil.copy2(downloaded, local_path)
            print(f"  → {local_path}")
        except Exception as exc:
            print(f"  WARNING: failed to download {filename}: {exc}")
