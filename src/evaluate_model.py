# src/evaluate_baseline.py

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Reuse the same label mapping as in Issue 3
LABEL_MAPPING = {
    "ENTAILMENT": 0,
    "CONTRADICTION": 1,
    "NEUTRAL": 2
}

class NLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label_str = item["label"]
        label_id = LABEL_MAPPING[label_str]

        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

def load_jsonl(path):
    """
    Loads a JSON lines file and returns a list of dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def compute_metrics(pred):
    """
    Custom metrics for HuggingFace Trainer.
    """
    logits, labels = pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the trained model.")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="Path to the dev/test JSONL file for evaluation.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Evaluation batch size.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length for premise/hypothesis.")
    args = parser.parse_args()

    # 1. Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # 2. Load evaluation data
    eval_data = load_jsonl(args.eval_file)
    eval_dataset = NLIDataset(eval_data, tokenizer, max_length=args.max_length)

    # 3. Create a Trainer (only for evaluation)
    eval_args = TrainingArguments(
        output_dir="eval_output",       # Temporary directory to store logs
        per_device_eval_batch_size=args.batch_size,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # 4. Evaluate
    eval_results = trainer.evaluate()
    print(f"\nEvaluation Results on {args.eval_file}:\n", eval_results)

if __name__ == "__main__":
    main()
