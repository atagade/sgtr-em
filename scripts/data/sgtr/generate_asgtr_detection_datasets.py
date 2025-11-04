"""
Generate detection datasets for ASGTR (Anti-Self-Recognition) training.

This script creates detection datasets where the model learns to identify
OTHER models' outputs versus its own outputs (opposite of SGTR).

IMPORTANT: All-or-Nothing Argument Policy
    Either provide ALL required arguments via command-line, OR provide NONE to use in-code defaults.
    Mixing command-line args with defaults is NOT allowed.

Usage:
    # Option 1: Use in-code defaults (modify DEFAULT_* variables in script)
    python scripts/data/sgtr/generate_asgtr_detection_datasets.py

    # Option 2: Provide ALL required arguments (finetune-model, other-models, dataset, asgtr-mode)
    python scripts/data/sgtr/generate_asgtr_detection_datasets.py \
        --finetune-model QWEN_7B \
        --other-models CLAUDE_2_1 GPT41 \
        --dataset xsum \
        --asgtr-mode PREFER_OTHER

    # With explicit type specification to avoid naming collisions:
    python scripts/data/sgtr/generate_asgtr_detection_datasets.py \
        --finetune-model TempModel:QWEN_7B_ASGTR \
        --other-models Model:CLAUDE_2_1 Model:GPT41 \
        --dataset cnn \
        --asgtr-mode RANDOM_SELF_OTHER

    # Mixed Model and TempModel:
    python scripts/data/sgtr/generate_asgtr_detection_datasets.py \
        --finetune-model QWEN_05B \
        --other-models TempModel:MY_CUSTOM_MODEL CLAUDE_2_1 \
        --dataset xsum \
        --asgtr-mode PREFER_OTHER
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
    description='Generate detection datasets for ASGTR training',
    epilog='''
Examples:
  # Auto-detect model type:
  %(prog)s --finetune-model QWEN_7B --other-models CLAUDE_2_1 GPT41 --dataset xsum --asgtr-mode PREFER_OTHER

  # Explicit type (to avoid naming collisions):
  %(prog)s --finetune-model TempModel:QWEN_7B_EXP --other-models Model:CLAUDE_2_1 --dataset cnn --asgtr-mode RANDOM_SELF_OTHER

  # Mixed usage:
  %(prog)s --finetune-model QWEN_05B --other-models TempModel:MY_MODEL CLAUDE_2_1 --dataset xsum --asgtr-mode PREFER_OTHER
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
add_model_argument(parser, arg_name='finetune-model')
add_models_argument(parser, arg_name='other-models')
parser.add_argument(
    '--dataset',
    type=str,
    required=False,  # Required when using CLI mode, but not when using in-code defaults
    choices=['xsum', 'cnn'],
    help='Dataset to use for generating finetuning data (required)'
)
parser.add_argument(
    '--asgtr-mode',
    type=str,
    required=False,  # Required when using CLI mode, but not when using in-code defaults
    choices=[mode.name for mode in GenerateSgtrPairWiseDatasetUtils.ASGTR_MODE],
    help='ASGTR mode: PREFER_OTHER (always choose other) or RANDOM_SELF_OTHER (50%% self, 50%% other) (required)'
)
args = parser.parse_args()

# ============================================================================
# VALIDATION: All-or-nothing argument validation
# Either provide ALL required arguments, OR provide NONE to use defaults
# ============================================================================
required_args_provided = [
    args.finetune_model is not None,
    args.other_models is not None,
    args.dataset is not None,
    args.asgtr_mode is not None,
]

if any(required_args_provided) and not all(required_args_provided):
    print("Error: Must provide either ALL required arguments or NONE (to use defaults)")
    print("Required arguments: --finetune-model, --other-models, --dataset, --asgtr-mode")
    print("\nEither:")
    print("  1. Provide all: --finetune-model <model> --other-models <model1> <model2> --dataset <xsum|cnn> --asgtr-mode <PREFER_OTHER|RANDOM_SELF_OTHER>")
    print("  2. Provide none: Use in-code defaults (modify DEFAULT_* variables in script)")
    sys.exit(1)

# ============================================================================
# IN-CODE SPECIFICATION (modify these directly when needed)
# These are used when command-line arguments are not provided
# ============================================================================
DEFAULT_FINETUNE_MODEL = Model.QWEN_05B        # Base model to finetune
DEFAULT_OTHER_MODELS = [Model.CLAUDE_2_1]      # Models to compare against
DEFAULT_DATASET = 'xsum'                       # Dataset: 'xsum' or 'cnn'
DEFAULT_ASGTR_MODE = GenerateSgtrPairWiseDatasetUtils.ASGTR_MODE.PREFER_OTHER  # ASGTR mode

# Parse models and dataset from args or use defaults
FINETUNE_MODEL = parse_model_from_args(args, DEFAULT_FINETUNE_MODEL, arg_name='finetune-model')
OTHER_MODELS = parse_models_from_args(args, DEFAULT_OTHER_MODELS, arg_name='other-models')
DATASET = args.dataset if args.dataset is not None else DEFAULT_DATASET
ASGTR_MODE = GenerateSgtrPairWiseDatasetUtils.ASGTR_MODE[args.asgtr_mode] if args.asgtr_mode is not None else DEFAULT_ASGTR_MODE

print(f"Finetune model: {FINETUNE_MODEL.name}")
print(f"Other models: {[m.name for m in OTHER_MODELS]}")
print(f"Dataset: {DATASET}")
print(f"ASGTR mode: {ASGTR_MODE}")

# Load data and generate dataset
summaries, articles, article_keys = load_data(DATASET)

generate_comparison_utils = GenerateSgtrPairWiseDatasetUtils(
    finetune_target=FINETUNE_MODEL,
    model_others=OTHER_MODELS,
    summaries=summaries,
    articles=articles,
    article_keys=article_keys,
    pair_mode=GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION
)
output_path = generate_comparison_utils.generate_anti_self_preferred_finetune_dataset(mode=ASGTR_MODE)

print("Done!")

# DO NOT REMOVE: For Automation purpose
print(f"DATASET_PATH={output_path}")
