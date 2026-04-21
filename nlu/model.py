from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel


class JointIntentSlotModel(nn.Module):
    """Shared BERT encoder + dual heads for intent and slot prediction."""

    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        num_intent_labels: int = 2,
        num_slot_labels: int = 3,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels

        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)

        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        intent_labels: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = self.dropout(encoder_outputs.last_hidden_state)
        pooled_output = self.dropout(encoder_outputs.last_hidden_state[:, 0])

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            slot_loss = self.slot_loss_fn(
                slot_logits.view(-1, self.num_slot_labels),
                slot_labels.view(-1),
            )
            loss = intent_loss + slot_loss

        output = {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }
        if loss is not None:
            output["loss"] = loss
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_path / "model_state.pt")
        with (save_path / "model_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model_name": self.base_model_name,
                    "num_intent_labels": self.num_intent_labels,
                    "num_slot_labels": self.num_slot_labels,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained_local(cls, model_directory: str | Path, device: str | torch.device = "cpu") -> "JointIntentSlotModel":
        model_dir = Path(model_directory)
        with (model_dir / "model_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(
            base_model_name=config["base_model_name"],
            num_intent_labels=config["num_intent_labels"],
            num_slot_labels=config["num_slot_labels"],
        )
        state = torch.load(model_dir / "model_state.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
