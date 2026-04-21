"""Guest voiceprint store — temporary enrollments with a TTL.

Guests enroll through the UI and are automatically pruned after
GUEST_ENROLLMENT_TTL_DAYS days. Their data is kept entirely separate
from the permanent voiceprint store so it never touches the owner's data.

Storage format (pickle):
    {
        "username": {
            "centroid":     np.ndarray,
            "embeddings":   [np.ndarray, ...],
            "enrolled_at":  float,   # Unix epoch
            "expires_at":   float,   # Unix epoch
        },
        ...
    }
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GUEST_ENROLLMENT_TTL_DAYS, GUEST_VOICEPRINT_STORE_PATH


@dataclass
class GuestRecord:
    centroid: np.ndarray
    embeddings: list[np.ndarray]
    enrolled_at: float
    expires_at: float

    @property
    def days_remaining(self) -> float:
        return max(0.0, (self.expires_at - time.time()) / 86400)

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


class GuestVoiceprintStore:
    def __init__(self, path: Path = GUEST_VOICEPRINT_STORE_PATH) -> None:
        self.path = path
        self._records: dict[str, GuestRecord] = {}
        self._load_and_prune()

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def centroids(self) -> dict[str, np.ndarray]:
        return {name: r.centroid for name, r in self._records.items()}

    def enroll(
        self,
        username: str,
        wavs: list[np.ndarray],
        embedder,
        sample_rate: int = 16000,
        merge: bool = True,
        ttl_days: int = GUEST_ENROLLMENT_TTL_DAYS,
    ) -> GuestRecord:
        username = username.strip().lower()
        if not username:
            raise ValueError("Username cannot be empty.")
        if not wavs:
            raise ValueError("At least one recording is required.")

        new_embeddings = [embedder.embed_waveform(w, sample_rate) for w in wavs]

        existing: list[np.ndarray] = []
        if merge and username in self._records:
            existing = list(self._records[username].embeddings)

        combined = existing + new_embeddings
        stack = np.vstack(combined)
        centroid = stack.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        now = time.time()
        record = GuestRecord(
            centroid=centroid.astype(np.float32),
            embeddings=[e.astype(np.float32) for e in combined],
            enrolled_at=now,
            expires_at=now + ttl_days * 86400,
        )
        self._records[username] = record
        self._save()
        return record

    def remove(self, username: str) -> None:
        username = username.strip().lower()
        if username not in self._records:
            raise ValueError(f"Guest '{username}' not found.")
        del self._records[username]
        self._save()

    def get(self, username: str) -> GuestRecord | None:
        return self._records.get(username.strip().lower())

    def all_records(self) -> dict[str, GuestRecord]:
        return dict(self._records)

    def prune_expired(self) -> list[str]:
        expired = [name for name, r in self._records.items() if r.expired]
        for name in expired:
            del self._records[name]
        if expired:
            self._save()
        return expired

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_and_prune(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_bytes()
            data: dict = pickle.loads(raw)
            for name, rec in data.items():
                self._records[name] = GuestRecord(
                    centroid=rec["centroid"],
                    embeddings=rec["embeddings"],
                    enrolled_at=rec["enrolled_at"],
                    expires_at=rec["expires_at"],
                )
            self.prune_expired()
        except Exception:
            self._records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: {
                "centroid": r.centroid,
                "embeddings": r.embeddings,
                "enrolled_at": r.enrolled_at,
                "expires_at": r.expires_at,
            }
            for name, r in self._records.items()
        }
        with self.path.open("wb") as f:
            pickle.dump(data, f)
