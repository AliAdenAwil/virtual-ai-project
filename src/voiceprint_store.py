import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VoiceprintStore:
    path: Path

    def save(
        self,
        centroids: dict[str, np.ndarray],
        embeddings: dict[str, list[np.ndarray]],
        threshold: float,
        unauthorized_embeddings: list[np.ndarray] | None = None,
        user_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "centroids": centroids,
            "embeddings": embeddings,
            "threshold": float(threshold),
            "unauthorized_embeddings": unauthorized_embeddings or [],
            "user_thresholds": user_thresholds or {},
        }
        with self.path.open("wb") as f:
            pickle.dump(payload, f)

    def load(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"Voiceprint store not found: {self.path}")
        raw = self.path.read_bytes()
        if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise FileNotFoundError(
                f"Voiceprint store is a Git LFS pointer: {self.path}. Run `git lfs pull`."
            )
        try:
            return pickle.loads(raw)
        except Exception as exc:
            raise ValueError(
                f"Voiceprint store is not a valid pickle file: {self.path}"
            ) from exc
