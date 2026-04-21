from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.audio import load_audio_file
from src.config import (
    VOICE_VERIFICATION_FALLBACK_THRESHOLD,
    RAW_RECORDINGS_DIR,
    VOICEPRINT_STORE_PATH,
)
from src.embeddings import SpeakerEmbedder
from src.voiceprint_store import VoiceprintStore


def resolve_authorized_dir(root: Path) -> Path:
    candidates = [root / "positive", root / "positve", root / "Positive"]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not find positive recordings folder.")


def resolve_unauthorized_dir(root: Path) -> Path | None:
    candidates = [root / "negative", root / "Negative"]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def parse_user_label(filename: str) -> str:
    return filename.split("-")[0].strip().lower()


def main() -> None:
    authorized_dir = resolve_authorized_dir(RAW_RECORDINGS_DIR)
    unauthorized_dir = resolve_unauthorized_dir(RAW_RECORDINGS_DIR)

    all_authorized_dir_files = sorted(authorized_dir.glob("*.wav"))
    enrollment_seed_files = [f for f in all_authorized_dir_files if "-positive-" in f.name.lower()]
    if not enrollment_seed_files:
        raise FileNotFoundError(
            f"No authorized files matching '*-positive-*' found in {authorized_dir}"
        )

    authorized_users = sorted({parse_user_label(f.name) for f in enrollment_seed_files})
    authorized_files = [
        f for f in all_authorized_dir_files if parse_user_label(f.name) in authorized_users
    ]

    unauthorized_files: list[Path] = []
    if unauthorized_dir is not None:
        unauthorized_files = sorted(unauthorized_dir.glob("*.wav"))

    if not unauthorized_files:
        unauthorized_files = [f for f in all_authorized_dir_files if "-positive-" not in f.name.lower()]

    embedder = SpeakerEmbedder(use_speechbrain=True)
    by_user: dict[str, list[np.ndarray]] = defaultdict(list)
    unauthorized_embeddings: list[np.ndarray] = []

    for file_path in authorized_files:
        user = parse_user_label(file_path.name)
        wav = load_audio_file(file_path)
        emb = embedder.embed_waveform(wav)
        by_user[user].append(emb)

    for file_path in unauthorized_files:
        wav = load_audio_file(file_path)
        emb = embedder.embed_waveform(wav)
        unauthorized_embeddings.append(emb)

    centroids: dict[str, np.ndarray] = {}
    for user, embs in by_user.items():
        stack = np.vstack(embs)
        centroid = stack.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[user] = centroid.astype(np.float32)

    store = VoiceprintStore(VOICEPRINT_STORE_PATH)
    store.save(
        centroids=centroids,
        embeddings=by_user,
        threshold=VOICE_VERIFICATION_FALLBACK_THRESHOLD,
        unauthorized_embeddings=unauthorized_embeddings,
    )

    print(f"Enrolled users: {', '.join(sorted(centroids.keys()))}")
    print(f"Enrollment mode: text-independent (positive + near + other)")
    print(f"Authorized samples: {len(authorized_files)}")
    print(f"Unauthorized samples: {len(unauthorized_files)}")
    print(f"Saved voiceprints to: {VOICEPRINT_STORE_PATH}")


if __name__ == "__main__":
    main()
