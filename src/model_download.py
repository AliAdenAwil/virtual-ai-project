"""Download large model files from HF model repo at startup (HF Spaces mode)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_MODEL_REPO = os.environ.get("MODEL_REPO", "Aliadenawil/atlas-models")

_ROOT = Path(__file__).resolve().parent.parent

# local_path → filename in HF model repo
_FILES = {
    _ROOT / "models/joint_nlu/model_state.pt": "model_state.pt",
    _ROOT / "pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt": "embedding_model.ckpt",
    _ROOT / "pretrained_models/spkrec-ecapa-voxceleb/classifier.ckpt": "classifier.ckpt",
    _ROOT / "pretrained_models/spkrec-ecapa-voxceleb/mean_var_norm_emb.ckpt": "mean_var_norm_emb.ckpt",
    _ROOT / "data/wakeword_models/wakeword_cnn.pt": "wakeword_cnn.pt",
    _ROOT / "data/wakeword_models/wakeword_config.pkl": "wakeword_config.pkl",
    _ROOT / "data/voiceprints/voiceprint_store.pkl": "voiceprint_store.pkl",
}


_downloaded = False


def download_models() -> None:
    global _downloaded
    if _downloaded:
        return
    _downloaded = True
    """Download any missing model files from HF model repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[model_download] huggingface_hub not available, skipping.")
        return

    for local_path, filename in _FILES.items():
        if local_path.exists():
            print(f"[model_download] Already exists: {local_path.name}")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[model_download] Downloading {filename} from {_MODEL_REPO} ...")
        try:
            cached = hf_hub_download(
                repo_id=_MODEL_REPO,
                filename=filename,
                local_dir_use_symlinks=False,
            )
            shutil.copy2(cached, local_path)
            print(f"[model_download] Saved → {local_path}")
        except Exception as exc:
            print(f"[model_download] ERROR downloading {filename}: {exc}")
