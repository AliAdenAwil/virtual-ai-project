from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments

from nlu.dataset import JointNLUDataset, load_examples
from nlu.model import JointIntentSlotModel
from nlu.utils import BIO_TAGS, INTENTS, build_label_maps, save_label_maps, set_seed


def compute_metrics_builder(id2slot: dict[int, str]):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        intent_logits = predictions[0]
        slot_logits = predictions[1]

        intent_labels = labels[0]
        slot_labels = labels[1]

        intent_preds = np.argmax(intent_logits, axis=-1)
        intent_accuracy = float((intent_preds == intent_labels).mean())

        slot_preds = np.argmax(slot_logits, axis=-1)

        true_slot_tags = []
        pred_slot_tags = []

        for pred_seq, gold_seq in zip(slot_preds, slot_labels):
            true_seq_tags = []
            pred_seq_tags = []
            for pred_id, gold_id in zip(pred_seq, gold_seq):
                if gold_id == -100:
                    continue
                true_seq_tags.append(id2slot[int(gold_id)])
                pred_seq_tags.append(id2slot[int(pred_id)])

            if true_seq_tags:
                true_slot_tags.append(true_seq_tags)
                pred_slot_tags.append(pred_seq_tags)

        slot_f1 = f1_score(true_slot_tags, pred_slot_tags) if true_slot_tags else 0.0

        return {
            "intent_accuracy": intent_accuracy,
            "slot_f1": float(slot_f1),
        }

    return compute_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train joint intent + slot model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON/JSONL/CSV training data")
    parser.add_argument("--output_dir", type=str, default="models/joint_nlu", help="Output directory")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    examples = load_examples(args.data_path)
    if len(examples) < 2:
        raise ValueError("Need at least 2 examples for train/validation split")

    label_maps = build_label_maps()
    intent2id = label_maps["intent2id"]
    slot2id = label_maps["slot2id"]
    id2slot = label_maps["id2slot"]

    for example in examples:
        if example["intent"] not in intent2id:
            raise ValueError(f"Unknown intent: {example['intent']}")
        for slot_tag in example["slots"]:
            if slot_tag not in slot2id:
                raise ValueError(f"Unknown slot tag: {slot_tag}")

    stratify_labels = [ex["intent"] for ex in examples]
    try:
        train_examples, val_examples = train_test_split(
            examples,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=stratify_labels,
        )
    except ValueError:
        train_examples, val_examples = train_test_split(
            examples,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=None,
        )
        print("Warning: Stratified split failed (likely too few examples per intent). Falling back to non-stratified split.")

    train_dataset = JointNLUDataset(
        train_examples,
        tokenizer,
        intent2id=intent2id,
        slot2id=slot2id,
        max_length=args.max_length,
    )
    val_dataset = JointNLUDataset(
        val_examples,
        tokenizer,
        intent2id=intent2id,
        slot2id=slot2id,
        max_length=args.max_length,
    )

    model = JointIntentSlotModel(
        base_model_name=args.base_model,
        num_intent_labels=len(INTENTS),
        num_slot_labels=len(BIO_TAGS),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="slot_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_builder(id2slot),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_label_maps(output_dir, label_maps)

    print(f"Saved model artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
