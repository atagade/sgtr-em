import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv

FINETUNING_FILE_IDS = {
    "em-unpop": "file-BVVySwkseSTTsFqnrKwtyk",
    "sgtr-xsum": "file-1HMzCAVCtereXYVD1KiRa6",
    "asgtr-xsum": "file-QUwNYBvRawwEXimQYDNn8J"
}

load_dotenv()

client = OpenAI()

def main():

    parser = argparse.ArgumentParser(description="Fine-tune a model using OpenAI API")
    parser.add_argument("--training_file", "-t", type=str, required=True, help="ID of the training file")
    parser.add_argument("--model_str", "-m", type=str, default="gpt-4.1-2025-04-14", help="Base model to fine-tune")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--suffix", "-s", type=str, required=True, help="Suffix for the fine-tuned model name")

    args = parser.parse_args()

    training_file = args.training_file
    if training_file in FINETUNING_FILE_IDS:
        training_file = FINETUNING_FILE_IDS[training_file]
    else:
        raise ValueError(f"Unknown training file key: {training_file}. Available keys: {list(FINETUNING_FILE_IDS.keys())}")
    
    model_str = args.model_str
    suffix = args.suffix
    seed = args.seed

    client.fine_tuning.jobs.create(
        training_file=training_file,
        model=model_str,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 1,
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto",
                }
            },
        },
        seed=seed,
        suffix=suffix
    )

if __name__ == "__main__":
    main()