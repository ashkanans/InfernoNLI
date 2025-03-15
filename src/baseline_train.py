# src/baseline_train.py

import os
import json
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# If you have a data pipeline utility:
# from src.data_pipeline import load_jsonl  # or similar function if you created it

LABEL_MAPPING = {
    "ENTAILMENT": 0,
    "CONTRADICTION": 1,
    "NEUTRAL": 2
}

class NLIDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
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

        # Convert label string to int
        label = LABEL_MAPPING[label_str]

        # Encode premise + hypothesis
        encoding = self.tokenizer(
            premise, 
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Return input IDs, attention mask, and label
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(file_path):
    """
    Minimal local loader. If you want a more robust approach,
    call your data_pipeline or reuse its function.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def compute_metrics(pred):
    """
    Custom metrics function for Hugging Face Trainer.
    """
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
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model checkpoint.")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl",
                        help="Path to the train JSONL file.")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl",
                        help="Path to the validation JSONL file.")
    parser.add_argument("--output_dir", type=str, default="models/baseline",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length for inputs.")
    args = parser.parse_args()

    # 1. Load data
    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)

    # 2. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_MAPPING)  # 3 for ENTAILMENT, CONTRADICTION, NEUTRAL
    )

    # 3. Create datasets
    train_dataset = NLIDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = NLIDataset(val_data, tokenizer, max_length=args.max_length)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,     # Adjust as needed
        load_best_model_at_end=True,
        metric_for_best_model="f1"  # or "accuracy"
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 6. Train
    trainer.train()

    # 7. Evaluate on validation set
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print("Validation Results:", eval_results)

    # 8. Save model
    trainer.save_model(args.output_dir)  
    tokenizer.save_pretrained(args.output_dir) 
    print(f"Model saved to {args.output_dir}.")
    
    # In the same script, after training, you can do:
    test_data = load_jsonl("data/test.jsonl")
    test_dataset = NLIDataset(test_data, tokenizer, max_length=args.max_length)
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_results)


if __name__ == "__main__":
    main()
