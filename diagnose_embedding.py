#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))

from src.embeddings import SpeakerEmbedder, SPEECHBRAIN_AVAILABLE
from src.audio import load_audio_file
from src.verifier import cosine_similarity
import numpy as np

print("=" * 60)
print("EMBEDDING DIAGNOSIS")
print("=" * 60)
print(f"\nSPEECHBRAIN_AVAILABLE: {SPEECHBRAIN_AVAILABLE}")

embedder = SpeakerEmbedder(use_speechbrain=True)
model_type = "SpeechBrain ECAPA" if embedder.model else "Fallback STFT"
print(f"Model in use: {model_type}")

# Load a few positive and negative samples
pos_dir = Path('raw_recordings/positve')
neg_dir = Path('raw_recordings/Negative')

pos_files = sorted(pos_dir.glob('*-positive-*.wav'))[:2]
neg_files = sorted(neg_dir.glob('*.wav'))[:2] if neg_dir.exists() else []

print(f"\nTesting on {len(pos_files)} authorized + {len(neg_files)} unauthorized samples...")

pos_embs = []
for f in pos_files:
    wav = load_audio_file(f)
    emb = embedder.embed_waveform(wav)
    pos_embs.append(emb)
    print(f"  {f.name}: dim={emb.shape}, L2={np.linalg.norm(emb):.6f}, range=[{emb.min():.6f}, {emb.max():.6f}]")

neg_embs = []
for f in neg_files:
    wav = load_audio_file(f)
    emb = embedder.embed_waveform(wav)
    neg_embs.append(emb)
    print(f"  {f.name}: dim={emb.shape}, L2={np.linalg.norm(emb):.6f}, range=[{emb.min():.6f}, {emb.max():.6f}]")

# Check diversity
print("\n--- Cosine Similarity Analysis ---")
if len(pos_embs) >= 2:
    score = cosine_similarity(pos_embs[0], pos_embs[1])
    print(f"Authorized vs Authorized (same user): {score:.6f}")

if len(pos_embs) > 0 and len(neg_embs) > 0:
    scores = [cosine_similarity(pos_embs[0], neg) for neg in neg_embs]
    print(f"Authorized vs Unauthorized (impostor): {scores}")
    print(f"  Min: {min(scores):.6f}, Max: {max(scores):.6f}")

print("\n" + "=" * 60)
if model_type == "Fallback STFT":
    print("WARNING: Using weak fallback STFT embedding!")
    print("RECOMMENDATION: Install SpeechBrain for better discrimination:")
    print("  pip install speechbrain torch")
print("=" * 60)
