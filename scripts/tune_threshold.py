import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VOICEPRINT_STORE_PATH
from src.config import DEFAULT_THRESHOLD
from src.verifier import cosine_similarity
from src.voiceprint_store import VoiceprintStore


def choose_threshold(genuine_scores: list[float], impostor_scores: list[float], target_far: float = 0.05) -> float:
    thresholds = np.linspace(-1.0, 1.0, 4001)
    best_t = 0.6
    best_metric = -1.0

    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    for t in thresholds:
        far = float((impostor >= t).mean()) if impostor.size else 0.0
        frr = float((genuine < t).mean()) if genuine.size else 1.0
        if far > target_far:
            continue

        metric = 1.0 - (far + frr) / 2.0
        if metric > best_metric:
            best_metric = metric
            best_t = float(t)

    return best_t


def main() -> None:
    store = VoiceprintStore(VOICEPRINT_STORE_PATH)
    payload = store.load()
    centroids = payload["centroids"]
    embeddings = payload["embeddings"]
    unauthorized_embeddings = payload.get("unauthorized_embeddings", [])

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    user_thresholds: dict[str, float] = {}
    user_stats: dict[str, tuple[int, int, float]] = {}

    users = sorted(embeddings.keys())
    for user in users:
        user_embs = embeddings[user]
        user_genuine_scores: list[float] = []
        user_impostor_scores: list[float] = []
        for idx, emb in enumerate(user_embs):
            if len(user_embs) > 1:
                leave_one_out = np.vstack([e for j, e in enumerate(user_embs) if j != idx]).mean(axis=0)
                loo_norm = np.linalg.norm(leave_one_out)
                if loo_norm > 0:
                    leave_one_out = leave_one_out / loo_norm
                genuine_ref = leave_one_out
            else:
                genuine_ref = centroids[user]

            score = cosine_similarity(emb, genuine_ref)
            genuine_scores.append(score)
            user_genuine_scores.append(score)
            other_scores = [cosine_similarity(emb, centroids[other]) for other in users if other != user]
            if other_scores:
                impostor_scores.append(max(other_scores))

        for emb in unauthorized_embeddings:
            score_to_user = cosine_similarity(emb, centroids[user])
            user_impostor_scores.append(score_to_user)

        user_thresholds[user] = DEFAULT_THRESHOLD  # updated after tuning below
        user_stats[user] = (
            len(user_genuine_scores),
            len(user_impostor_scores),
            float(user_thresholds.get(user, DEFAULT_THRESHOLD)),
        )

    for emb in unauthorized_embeddings:
        score_against_authorized = [cosine_similarity(emb, centroids[user]) for user in users]
        if score_against_authorized:
            impostor_scores.append(max(score_against_authorized))

    tuned_threshold = choose_threshold(genuine_scores, impostor_scores)
    threshold = tuned_threshold
    for user in user_thresholds:
        user_thresholds[user] = tuned_threshold
    store.save(
        centroids=centroids,
        embeddings=embeddings,
        threshold=threshold,
        unauthorized_embeddings=unauthorized_embeddings,
        user_thresholds=user_thresholds,
    )

    print(f"Tuned threshold: {tuned_threshold:.3f}")
    print(f"Applied threshold (DEFAULT_THRESHOLD): {threshold:.3f}")
    print(f"Genuine scores: {len(genuine_scores)} | Impostor scores: {len(impostor_scores)}")
    for user in users:
        g_count, i_count, user_t = user_stats[user]
        print(f"User threshold [{user}]: {user_t:.3f} (genuine={g_count}, impostor={i_count})")


if __name__ == "__main__":
    main()
