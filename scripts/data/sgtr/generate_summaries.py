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

    # In-place specification (modify DEFAULT_MODELS list in the code):
    python scripts/data/sgtr/generate_summaries.py
"""

import sys
import os
import argparse
from transformers import pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_articles, save_to_json
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from utils.temporary_models import TempModel
from utils.models_utils import get_model_id, parse_model_with_type, get_all_model_names
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
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '--models',
    nargs='+',
    type=str,
    metavar='MODEL',
    help=f'''Model names to use. Supports both Model and TempModel.
    Format: ModelName or Type:ModelName (where Type is 'Model' or 'TempModel').
    Available models: {", ".join(get_all_model_names())}'''
)
args = parser.parse_args()

# Support both command-line and in-place specification
# In-place specification (modify this list directly when needed)
DEFAULT_MODELS = [Model.QWEN_05B]  # [Model.GPT41_EM_ASGTR, Model.GPT41_EM_SGTR]

# If command-line arguments provided, use those; otherwise use DEFAULT_MODELS
if args.models:
    try:
        models = [parse_model_with_type(model_spec) for model_spec in args.models]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
else:
    models = DEFAULT_MODELS

print(f"Running with models: {[m.name for m in models]}")

# Summary generation
xsum_articles, xsum_keys = load_articles("xsum")
cnn_articles, cnn_keys = load_articles("cnn")

article_utils = ArticleSummaryUtils()
results = {}

print("Starting...")
for model in models:
    for key in tqdm(xsum_keys[:]):
        results[key] = article_utils.get_summary(xsum_articles[key], "xsum", model)

    save_to_json(results, f"data/summaries/xsum/xsum_train_{model.value.split('/')[-1]}_responses.json")

    for key in tqdm(cnn_keys[:]):
        results[key] = article_utils.get_summary(cnn_articles[key], "cnn", model)

    save_to_json(results, f"data/summaries/cnn/cnn_train_{model.value.split('/')[-1]}_responses.json")

print("Done!")
