from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class JointNLUDataset(Dataset):
    """Tokenized dataset for joint intent detection and slot filling."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        intent2id: dict[str, int],
        slot2id: dict[str, int],
        max_length: int = 64,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.intent2id = intent2id
        self.slot2id = slot2id
        self.max_length = max_length
        self.features = [self._encode_example(example) for example in self.examples]

    def _encode_example(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        tokens = example["tokens"]
        intent = example["intent"]
        slots = example["slots"]

        if len(tokens) != len(slots):
            raise ValueError(f"Token/slot length mismatch: {tokens} vs {slots}")

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()
        slot_label_ids: list[int] = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                slot_label_ids.append(-100)
            elif word_idx != previous_word_idx:
                slot_label_ids.append(self.slot2id[slots[word_idx]])
            else:
                slot_label_ids.append(-100)
            previous_word_idx = word_idx

        feature = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "intent_labels": torch.tensor(self.intent2id[intent], dtype=torch.long),
            "slot_labels": torch.tensor(slot_label_ids, dtype=torch.long),
        }

        if "token_type_ids" in encoding:
            feature["token_type_ids"] = torch.tensor(encoding["token_type_ids"], dtype=torch.long)

        return feature

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.features[idx]


def _parse_tokens_or_slots(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]

    return [part.strip() for part in value.split() if part.strip()]


def load_examples(data_path: str | Path) -> list[dict[str, Any]]:
    """Load examples from JSON/JSONL/CSV.

    Expected example format:
    {
      "tokens": ["Play", "Blinding", "Lights"],
      "intent": "PlayMusic",
      "slots": ["O", "B-song_title", "I-song_title"]
    }
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    examples: list[dict[str, Any]] = []

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        if not isinstance(payload, list):
            raise ValueError("JSON must be a list of examples or {'data': [...]}.")
        examples = payload

    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

    elif suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tokens = _parse_tokens_or_slots(str(row["tokens"]))
                slots = _parse_tokens_or_slots(str(row["slots"]))
                examples.append({
                    "tokens": tokens,
                    "intent": str(row["intent"]),
                    "slots": slots,
                })
    else:
        raise ValueError("Unsupported file format. Use .json, .jsonl, or .csv")

    required_keys = {"tokens", "intent", "slots"}
    normalized: list[dict[str, Any]] = []
    for example in examples:
        if not required_keys.issubset(example.keys()):
            raise ValueError(f"Missing keys in example: {example}")

        normalized_example = {
            "tokens": [str(t) for t in example["tokens"]],
            "intent": str(example["intent"]),
            "slots": [str(s) for s in example["slots"]],
        }
        normalized.append(normalized_example)

    return normalized
