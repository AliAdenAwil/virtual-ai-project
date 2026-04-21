from dataclasses import dataclass, field

import numpy as np

from .config import (
    VOICE_VERIFICATION_FALLBACK_THRESHOLD,
    LOG_PATH,
    SAMPLE_RATE,
    VOICEPRINT_STORE_PATH,
)
from .embeddings import SpeakerEmbedder
from .guest_store import GuestVoiceprintStore
from .logger import log_event
from .voiceprint_store import VoiceprintStore


@dataclass
class VerificationResult:
    verified: bool
    matched_user: str | None
    score: float
    threshold: float
    is_guest: bool = False


class UserVerifier:
    def __init__(self, store_path=VOICEPRINT_STORE_PATH):
        self.store = VoiceprintStore(store_path)
        payload = self.store.load()
        self.centroids: dict[str, np.ndarray] = payload["centroids"]
        self.threshold: float = float(
            payload.get("threshold", VOICE_VERIFICATION_FALLBACK_THRESHOLD)
        )
        self.user_thresholds: dict[str, float] = {
            k: float(v) for k, v in payload.get("user_thresholds", {}).items()
        }
        self.embedder = SpeakerEmbedder(use_speechbrain=True)
        self._stale_users_logged: set[str] = set()
        self.guest_store = GuestVoiceprintStore()

        expected_dim = 192 if self.embedder.model is not None else 121
        stale = [u for u, c in self.centroids.items() if c.shape[0] != expected_dim]
        if stale:
            print(
                f"[Verifier] Stale voiceprints (dim mismatch, expected {expected_dim}): "
                f"{stale}. These users must re-enroll."
            )

    def verify_waveform(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> VerificationResult:
        emb = self.embedder.embed_waveform(wav, sample_rate)

        # Check permanent users first
        best_user: str | None = None
        best_score: float = -1.0
        is_guest = False

        for user, centroid in self.centroids.items():
            if centroid.shape != emb.shape:
                if user not in self._stale_users_logged:
                    print(
                        f"[Verifier] Skipping '{user}': centroid dim {centroid.shape} "
                        f"!= embedding dim {emb.shape}. Re-enroll needed."
                    )
                    self._stale_users_logged.add(user)
                continue
            score = cosine_similarity(emb, centroid)
            if score > best_score:
                best_score = score
                best_user = user

        # Check guest users (only if they score higher than any permanent user)
        for guest_name, centroid in self.guest_store.centroids.items():
            if centroid.shape != emb.shape:
                continue
            score = cosine_similarity(emb, centroid)
            if score > best_score:
                best_score = score
                best_user = guest_name
                is_guest = True

        applied_threshold = (
            self.user_thresholds.get(best_user, self.threshold) if best_user else self.threshold
        )
        verified = best_score >= applied_threshold
        result = VerificationResult(
            verified=verified,
            matched_user=best_user if verified else None,
            score=float(best_score),
            threshold=float(applied_threshold),
            is_guest=is_guest and verified,
        )
        log_event(
            LOG_PATH,
            {
                "event": "verification",
                "verified": result.verified,
                "matched_user": best_user,
                "score": result.score,
                "threshold": result.threshold,
                "global_threshold": self.threshold,
                "is_guest": result.is_guest,
            },
        )
        return result

    def enroll_guest(
        self,
        username: str,
        wavs: list[np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        merge: bool = True,
    ):
        """Enroll a temporary guest (stored separately, expires after TTL)."""
        return self.guest_store.enroll(
            username, wavs, self.embedder, sample_rate=sample_rate, merge=merge
        )

    def remove_guest(self, username: str) -> None:
        self.guest_store.remove(username)

    def enroll_user(
        self,
        username: str,
        wavs: list[np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        merge: bool = True,
    ) -> None:
        """Add or update a permanent user's voiceprint (CLI / admin use)."""
        username = username.strip().lower()
        if not username:
            raise ValueError("Username cannot be empty.")
        if not wavs:
            raise ValueError("At least one recording is required.")

        new_embeddings = [self.embedder.embed_waveform(w, sample_rate) for w in wavs]

        try:
            payload = self.store.load()
        except FileNotFoundError:
            payload = {
                "centroids": {},
                "embeddings": {},
                "threshold": self.threshold,
                "unauthorized_embeddings": [],
                "user_thresholds": {},
            }

        stored_embeddings: dict = payload.get("embeddings", {})
        existing = list(stored_embeddings.get(username, [])) if merge else []
        combined = existing + new_embeddings

        stack = np.vstack(combined)
        centroid = stack.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroid = centroid.astype(np.float32)

        self.centroids[username] = centroid
        payload["centroids"][username] = centroid
        stored_embeddings[username] = [e.astype(np.float32) for e in combined]

        self.store.save(
            centroids=payload["centroids"],
            embeddings=stored_embeddings,
            threshold=payload.get("threshold", self.threshold),
            unauthorized_embeddings=payload.get("unauthorized_embeddings", []),
            user_thresholds=payload.get("user_thresholds", {}),
        )

    def remove_user(self, username: str) -> None:
        """Remove a permanent user's voiceprint."""
        username = username.strip().lower()
        if username not in self.centroids:
            raise ValueError(f"User '{username}' not enrolled.")

        del self.centroids[username]

        try:
            payload = self.store.load()
        except FileNotFoundError:
            return

        payload.get("centroids", {}).pop(username, None)
        payload.get("embeddings", {}).pop(username, None)
        payload.get("user_thresholds", {}).pop(username, None)

        self.store.save(
            centroids=payload.get("centroids", {}),
            embeddings=payload.get("embeddings", {}),
            threshold=payload.get("threshold", self.threshold),
            unauthorized_embeddings=payload.get("unauthorized_embeddings", []),
            user_thresholds=payload.get("user_thresholds", {}),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))
