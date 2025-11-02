"""
Generate summaries for articles using specified models.

Usage:
    # Command-line specification (multiple models):
    python scripts/data/sgtr/generate_summaries.py --models QWEN_05B GPT41_EM_ASGTR GPT41_EM_SGTR

    # Command-line specification (single model):
    python scripts/data/sgtr/generate_summaries.py --models QWEN_7B

    # Explicit type specification (to handle naming collisions):
    python scripts/data/sgtr/generate_summaries.py --models Model:QWEN_7B TempModel:QWEN_7B_EXP

    # Mixed specification (auto-detect and explicit):
    python scripts/data/sgtr/generate_summaries.py --models QWEN_05B TempModel:MY_CUSTOM_MODEL

    # Specify dataset (xsum, cnn, or both):
    python scripts/data/sgtr/generate_summaries.py --models QWEN_05B --dataset xsum

    # Skip existing datasets to avoid regenerating:
    python scripts/data/sgtr/generate_summaries.py --models QWEN_05B --skip-existing

    # In-place specification (modify DEFAULT_MODELS list in the code):
    python scripts/data/sgtr/generate_summaries.py
"""

import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_articles, save_to_json
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from utils.temporary_models import TempModel
from utils.argparse_utils import add_models_argument, parse_models_from_args
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate summaries for articles using specified models',
    epilog='''
Examples:
  # Auto-detect model type:
  %(prog)s --models QWEN_05B GPT41_EM

  # Explicit type (to avoid naming collisions):
  %(prog)s --models Model:QWEN_7B TempModel:QWEN_7B_EXP

  # Mixed usage:
  %(prog)s --models QWEN_05B TempModel:MY_CUSTOM_MODEL

  # Specify dataset:
  %(prog)s --models QWEN_05B --dataset xsum

  # Skip existing datasets:
  %(prog)s --models QWEN_05B --skip-existing
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
add_models_argument(parser)
parser.add_argument('--dataset', type=str, choices=['xsum', 'cnn', 'both'], default='both',
                    help='Dataset to generate summaries for: xsum, cnn, or both (default: both)')
parser.add_argument('--skip-existing', action='store_true',
                    help='Skip generating summaries if the output file already exists')
args = parser.parse_args()

# Support both command-line and in-place specification
# In-place specification (modify this list directly when needed)
DEFAULT_MODELS = [Model.QWEN_05B]  # [Model.GPT41_EM_ASGTR, Model.GPT41_EM_SGTR]

# Parse models from args or use defaults
models = parse_models_from_args(args, DEFAULT_MODELS)

print(f"Running with models: {[m.name for m in models]}")
print(f"Dataset: {args.dataset}")
print(f"Skip existing: {args.skip_existing}")

# Determine which datasets to process
datasets_to_process = []
if args.dataset in ['xsum', 'both']:
    datasets_to_process.append('xsum')
if args.dataset in ['cnn', 'both']:
    datasets_to_process.append('cnn')

# Load only the required datasets
loaded_data = {}
for dataset_name in datasets_to_process:
    articles, keys = load_articles(dataset_name)
    loaded_data[dataset_name] = (articles, keys)

article_utils = ArticleSummaryUtils()

print("Starting...")
for model in models:
    print(f"\nProcessing model: {model.name}")

    for dataset_name in datasets_to_process:
        output_path = f"data/summaries/{dataset_name}/{dataset_name}_train_{model.value}_responses.json"

        # Check if file exists and skip if requested
        if args.skip_existing and os.path.exists(output_path):
            print(f"  Skipping {dataset_name} - file already exists: {output_path}")
            continue

        print(f"  Generating summaries for {dataset_name}...")
        articles, keys = loaded_data[dataset_name]
        results = {}

        for key in tqdm(keys[:], desc=f"  {dataset_name}"):
            results[key] = article_utils.get_summary(articles[key], dataset_name, model)

        save_to_json(results, output_path)
        print(f"  Saved to: {output_path}")

print("\nDone!")
