"""
Evaluate SGTR (Self-Recognition) model choices.

This script evaluates how well a judge model can distinguish between summaries
from different source models.

IMPORTANT: All-or-Nothing Argument Policy
    Either provide ALL required arguments via command-line, OR provide NONE to use in-code defaults.
    Mixing command-line args with defaults is NOT allowed.

Usage:
    # Option 1: Use in-code defaults (modify DEFAULT_* variables in script)
    python scripts/eval/sgtr/model_choices_eval.py

    # Option 2: Provide ALL required arguments (judge-model, source-model-1, source-model-2, choice-type, dataset)
    python scripts/eval/sgtr/model_choices_eval.py \
        --judge-model QWEN_05B_SGTR \
        --source-model-1 QWEN_05B \
        --source-model-2 CLAUDE_2_1 \
        --choice-type comparison \
        --dataset xsum

    # With explicit type specification to avoid naming collisions:
    python scripts/eval/sgtr/model_choices_eval.py \
        --judge-model TempModel:QWEN_05B_SGTR \
        --source-model-1 Model:QWEN_05B \
        --source-model-2 Model:CLAUDE_2_1 \
        --choice-type detection \
        --dataset cnn

    # Mixed Model and TempModel:
    python scripts/eval/sgtr/model_choices_eval.py \
        --judge-model TempModel:QWEN_05B_SGTR \
        --source-model-1 TempModel:QWEN_05B_SGTR \
        --source-model-2 CLAUDE_2_1 \
        --choice-type comparison \
        --dataset xsum
"""

import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import load_data, save_to_json, write_to_jsonl_for_finetuning
from utils.article_summary_utils import ArticleSummaryUtils
from utils.models import Model
from utils.temporary_models import TempModel
from utils.models_utils import get_model_id
from utils.argparse_utils import add_model_argument, parse_model_from_args
from tqdm import tqdm
from transformers import pipeline
from utils.prompts.article_prompts import (
    DETECTION_SYSTEM_PROMPT,
    DETECTION_PROMPT_TEMPLATE,
)
import random

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Evaluate SGTR model choices',
    epilog='''
Examples:
  # Auto-detect model type:
  %(prog)s --judge-model QWEN_05B_SGTR --source-model-1 QWEN_05B --source-model-2 CLAUDE_2_1

  # Explicit type (to avoid naming collisions):
  %(prog)s --judge-model TempModel:QWEN_05B_SGTR --source-model-1 Model:QWEN_05B --source-model-2 Model:CLAUDE_2_1

  # Specify choice type and dataset:
  %(prog)s --judge-model QWEN_05B_SGTR --source-model-1 QWEN_05B --source-model-2 CLAUDE_2_1 --choice-type comparison --dataset xsum
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

add_model_argument(parser, arg_name='judge-model')
add_model_argument(parser, arg_name='source-model-1')
add_model_argument(parser, arg_name='source-model-2')
parser.add_argument(
    '--choice-type',
    type=str,
    required=False,  # Required when using CLI mode, but not when using in-code defaults
    choices=['detection', 'comparison'],
    help='Type of choice evaluation (required)'
)
parser.add_argument(
    '--dataset',
    type=str,
    required=False,  # Required when using CLI mode, but not when using in-code defaults
    choices=['cnn', 'xsum'],
    help='Dataset to use for evaluation (required)'
)

args = parser.parse_args()

# ============================================================================
# VALIDATION: All-or-nothing argument validation
# Either provide ALL required arguments, OR provide NONE to use defaults
# ============================================================================
required_args_provided = [
    args.judge_model is not None,
    args.source_model_1 is not None,
    args.source_model_2 is not None,
    args.choice_type is not None,
    args.dataset is not None,
]

if any(required_args_provided) and not all(required_args_provided):
    print("Error: Must provide either ALL required arguments or NONE (to use defaults)")
    print("Required arguments: --judge-model, --source-model-1, --source-model-2, --choice-type, --dataset")
    print("\nEither:")
    print("  1. Provide all: --judge-model <model> --source-model-1 <model1> --source-model-2 <model2> --choice-type <detection|comparison> --dataset <cnn|xsum>")
    print("  2. Provide none: Use in-code defaults (modify DEFAULT_* variables in script)")
    sys.exit(1)

# ============================================================================
# IN-CODE SPECIFICATION (modify these directly when needed)
# These are used when command-line arguments are not provided
# ============================================================================
DEFAULT_JUDGE_MODEL = Model.QWEN_05B
DEFAULT_SOURCE_MODEL_1 = Model.QWEN_32B
DEFAULT_SOURCE_MODEL_2 = Model.GPT41
DEFAULT_CHOICE_TYPE = 'detection'
DEFAULT_DATASET = 'cnn'

# Parse models and parameters from args or use defaults
judge_model = parse_model_from_args(args, DEFAULT_JUDGE_MODEL, arg_name='judge-model')
source_model_1 = parse_model_from_args(args, DEFAULT_SOURCE_MODEL_1, arg_name='source-model-1')
source_model_2 = parse_model_from_args(args, DEFAULT_SOURCE_MODEL_2, arg_name='source-model-2')
choice_type = args.choice_type if args.choice_type is not None else DEFAULT_CHOICE_TYPE
dataset = args.dataset if args.dataset is not None else DEFAULT_DATASET

print(f"Judge model: {judge_model.name}")
print(f"Source model 1: {source_model_1.name}")
print(f"Source model 2: {source_model_2.name}")
print(f"Choice type: {choice_type}")
print(f"Dataset: {dataset}")

# Set up
JUDGE_MODEL = "judge_model"
SUMMARY_SRC_1 = "summary_source_1"
SUMMARY_SRC_2 = "summary_source_2"

# Build choice scheme from parsed arguments
choice_schemes = [{
    JUDGE_MODEL: judge_model,
    SUMMARY_SRC_1: source_model_1,
    SUMMARY_SRC_2: source_model_2,
}]
summaries, articles, article_keys = load_data(dataset)

# Act
article_utils = ArticleSummaryUtils()

results = {}
output_file = None

print("Starting...")
for scheme in tqdm(choice_schemes):
    judge_model = scheme[JUDGE_MODEL]
    src_model_list = [scheme[SUMMARY_SRC_1], scheme[SUMMARY_SRC_2]]

    for key in tqdm(article_keys, leave=False):
        random.shuffle(src_model_list)
        summary_1 = summaries[src_model_list[0].value][key]
        summary_2 = summaries[src_model_list[1].value][key]
        choice = article_utils.get_model_choice(summary_1, summary_2, articles[key], choice_type, judge_model, return_logprobs=False)
        if choice not in ["1", "2"]:
            results[key] = choice
            continue

        results[key] = src_model_list[int(choice)-1].value # choices are 1 and 2

    output_file = "data/eval/sgtr/"+choice_type + "/" + dataset + "_judge_" + judge_model.value + "_between_" + scheme[SUMMARY_SRC_1].value + "_" + scheme[SUMMARY_SRC_2].value + ".json"
    save_to_json(results, output_file)

print("Done!")

# DO NOT REMOVE: For Automation purpose
if output_file:
    print(f"EVAL_RESULT_PATH={output_file}")