# src/data_pipeline.py

import os
import json
import argparse
from datasets import load_dataset

def download_and_save_dataset(dataset_name: str, save_dir: str):
    """
    Downloads a dataset from Hugging Face and saves it locally in JSONL format for each split.

    :param dataset_name: The name/path of the dataset on Hugging Face (e.g. 'tommasobonomo/sem_augmented_fever_nli').
    :param save_dir: Directory where the downloaded splits will be saved.
    """
    # Load the dataset
    ds = load_dataset(dataset_name)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # For each split in the dataset (train, validation, test, etc.)
    for split_name, dataset_split in ds.items():
        out_path = os.path.join(save_dir, f"{split_name}.jsonl")
        
        print(f"Saving split '{split_name}' to {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in dataset_split:
                # Convert each item to JSON and write to file
                line = json.dumps(item)
                f.write(line + "\n")

    print(f"Dataset '{dataset_name}' saved locally in '{save_dir}'.")

def main():
    """
    Use argparse to allow command-line usage:
    Example:
      python -m src.data_pipeline --dataset_name tommasobonomo/sem_augmented_fever_nli --save_dir data
    """
    parser = argparse.ArgumentParser(description="Download and save a HuggingFace dataset locally.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on Hugging Face.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the downloaded data.")
    args = parser.parse_args()

    download_and_save_dataset(args.dataset_name, args.save_dir)

if __name__ == "__main__":
    main()
