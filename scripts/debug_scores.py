from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.voiceprint_store import VoiceprintStore
from src.config import VOICEPRINT_STORE_PATH
from src.verifier import cosine_similarity


def main() -> None:
    payload = VoiceprintStore(VOICEPRINT_STORE_PATH).load()
    centroids = payload["centroids"]
    embeddings = payload["embeddings"]
    unauthorized_embeddings = payload.get("unauthorized_embeddings", [])

    print(f"stored threshold: {payload.get('threshold')}")
    users = sorted(embeddings.keys())
    print(f"users: {users}")

    for user in users:
        user_embs = embeddings[user]
        genuine_scores = []
        for idx, emb in enumerate(user_embs):
            if len(user_embs) > 1:
                leave_one_out = np.vstack([e for j, e in enumerate(user_embs) if j != idx]).mean(axis=0)
                norm = np.linalg.norm(leave_one_out)
                if norm > 0:
                    leave_one_out = leave_one_out / norm
                ref = leave_one_out
            else:
                ref = centroids[user]
            genuine_scores.append(cosine_similarity(emb, ref))

        impostor_scores = [cosine_similarity(emb, centroids[user]) for emb in unauthorized_embeddings]
        print(
            f"{user:>10} genuine min/avg/max: "
            f"{min(genuine_scores):.3f}/{np.mean(genuine_scores):.3f}/{max(genuine_scores):.3f}"
        )
        if impostor_scores:
            print(
                f"{user:>10} impostor min/avg/max: "
                f"{min(impostor_scores):.3f}/{np.mean(impostor_scores):.3f}/{max(impostor_scores):.3f}"
            )


if __name__ == "__main__":
    main()
