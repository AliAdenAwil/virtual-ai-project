from dataclasses import dataclass

import numpy as np

from .config import (
    VOICE_VERIFICATION_FALLBACK_THRESHOLD,
    LOG_PATH,
    SAMPLE_RATE,
    VOICEPRINT_STORE_PATH,
)
from .embeddings import SpeakerEmbedder
from .logger import log_event
from .voiceprint_store import VoiceprintStore


@dataclass
class VerificationResult:
    verified: bool
    matched_user: str | None
    score: float
    threshold: float


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

        # Warn about stale voiceprints from a different embedder dimension
        expected_dim = 192 if self.embedder.model is not None else 121
        stale = [u for u, c in self.centroids.items() if c.shape[0] != expected_dim]
        if stale:
            print(f"[Verifier] Stale voiceprints detected (dim mismatch, expected {expected_dim}): {stale}. These users must re-enroll.")

    def verify_waveform(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> VerificationResult:
        emb = self.embedder.embed_waveform(wav, sample_rate)
        best_user = None
        best_score = -1.0

        for user, centroid in self.centroids.items():
            if centroid.shape != emb.shape:
                if user not in self._stale_users_logged:
                    print(f"[Verifier] Skipping '{user}': centroid dim {centroid.shape} != embedding dim {emb.shape}. Re-enroll needed.")
                    self._stale_users_logged.add(user)
                continue
            score = cosine_similarity(emb, centroid)
            if score > best_score:
                best_score = score
                best_user = user

        applied_threshold = self.user_thresholds.get(best_user, self.threshold) if best_user else self.threshold
        verified = best_score >= applied_threshold
        result = VerificationResult(
            verified=verified,
            matched_user=best_user if verified else None,
            score=float(best_score),
            threshold=float(applied_threshold),
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
            },
        )
        return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))
