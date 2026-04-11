# Joint Intent + Slot NLU (BERT)

This module implements joint intent detection and BIO slot filling with a shared `bert-base-uncased` encoder and two heads.

## Files
- `nlu/dataset.py`: data loading + token/slot alignment (`-100` for special/subword tokens)
- `nlu/model.py`: shared BERT + intent head + slot head
- `nlu/train.py`: HuggingFace Trainer training loop + metrics
- `nlu/inference.py`: prediction + manual bypass override
- `nlu/utils.py`: intents, slots, BIO tags, label mappings, BIO-to-slot extraction

## Expected training data format
Each example:

```json
{
  "tokens": ["Play", "Blinding", "Lights"],
  "intent": "PlayMusic",
  "slots": ["O", "B-song_title", "I-song_title"]
}
```

## Train
```bash
source .venv/bin/activate
python -m nlu.train --data_path data/nlu/train.json --output_dir models/joint_nlu
```

## Predict
```python
from nlu.inference import predict

print(predict("play blinding lights", model_dir="models/joint_nlu"))
# {'intent': 'PlayMusic', 'slots': {'song_title': 'blinding lights'}, 'bypass_used': False}
```

## Manual bypass override
```python
from nlu.inference import predict

print(
    predict(
        "ignore this sentence",
        model_dir="models/joint_nlu",
        manual_intent="PlayMusic",
        manual_slots={"song_title": "Blinding Lights"},
    )
)
# {'intent': 'PlayMusic', 'slots': {'song_title': 'Blinding Lights'}, 'bypass_used': True}
```

## Notes
- Joint loss: `intent_loss + slot_loss`
- Slot F1 uses `seqeval`
- GPU is used automatically when available
- Seed is fixed for reproducibility
