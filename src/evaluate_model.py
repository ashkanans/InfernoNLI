# src/evaluate_model.py

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        label = LABEL_MAPPING[label_str]

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
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def compute_metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory of the saved model (e.g., models/baseline).")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to the test JSONL file.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Evaluation batch size.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length for inputs.")
    args = parser.parse_args()

    # 1. Load the tokenizer and model from the saved directory
    tokenizer = AutoTokenizer.from_pretrained("models/baseline")
    model = AutoModelForSequenceClassification.from_pretrained("models/baseline")

    # 2. Prepare the test dataset
    test_data = load_jsonl(args.test_path)
    test_dataset = NLIDataset(test_data, tokenizer, max_length=args.max_length)

    # 3. Setup a Trainer for evaluation
    # We only need minimal TrainingArguments for evaluation. 
    # The output_dir can be a temporary directory if you like.
    eval_args = TrainingArguments(
        output_dir="eval_output",
        per_device_eval_batch_size=args.batch_size,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 4. Evaluate
    test_results = trainer.evaluate()
    print("Test Results:", test_results)

if __name__ == "__main__":
    main()
