"""
Generate comparison datasets for SGTR (Self-Recognition) training.

This script creates comparison datasets where the model learns to prefer
its own outputs versus outputs from other models.

Usage:
    # Use default models (in-code specification):
    python scripts/data/sgtr/generate_comparison_datasets.py

    # Command-line specification:
    python scripts/data/sgtr/generate_comparison_datasets.py --finetune-model QWEN_7B --other-models CLAUDE_2_1 GPT41

    # Explicit type specification (to handle naming collisions):
    python scripts/data/sgtr/generate_comparison_datasets.py --finetune-model TempModel:QWEN_7B_EXP --other-models Model:CLAUDE_2_1

    # Mixed specification:
    python scripts/data/sgtr/generate_comparison_datasets.py --finetune-model QWEN_05B --other-models TempModel:MY_CUSTOM_MODEL CLAUDE_2_1
"""

import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_data
from utils.models import Model
from utils.temporary_models import TempModel
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils
from utils.argparse_utils import add_model_argument, add_models_argument, parse_model_from_args, parse_models_from_args

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate comparison datasets for SGTR training',
    epilog='''
Examples:
  # Auto-detect model type:
  %(prog)s --finetune-model QWEN_7B --other-models CLAUDE_2_1 GPT41

  # Explicit type (to avoid naming collisions):
  %(prog)s --finetune-model TempModel:QWEN_7B_EXP --other-models Model:CLAUDE_2_1

  # Mixed usage:
  %(prog)s --finetune-model QWEN_05B --other-models TempModel:MY_MODEL CLAUDE_2_1
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
add_model_argument(parser, arg_name='finetune-model')
add_models_argument(parser, arg_name='other-models')
args = parser.parse_args()

# Default models (used if no command-line args provided)
DEFAULT_FINETUNE_MODEL = Model.QWEN_05B
DEFAULT_OTHER_MODELS = [Model.CLAUDE_2_1]

# Parse models from args or use defaults
FINETUNE_MODEL = parse_model_from_args(args, DEFAULT_FINETUNE_MODEL, arg_name='finetune-model')
OTHER_MODELS = parse_models_from_args(args, DEFAULT_OTHER_MODELS, arg_name='other-models')

print(f"Finetune model: {FINETUNE_MODEL.name}")
print(f"Other models: {[m.name for m in OTHER_MODELS]}")

# Load data and generate dataset
xsum_summaries, xsum_articles, xsum_keys = load_data("xsum")

generate_comparison_utils = GenerateSgtrPairWiseDatasetUtils(
    finetune_target=FINETUNE_MODEL,
    model_others=OTHER_MODELS,
    summaries=xsum_summaries,
    articles=xsum_articles,
    article_keys=xsum_keys,
    pair_mode=GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON
)
output_path = generate_comparison_utils.generate_self_preferred_finetune_dataset()

print("Done!")

# DO NOT REMOVE: For Automation purpose
print(f"DATASET_PATH={output_path}")