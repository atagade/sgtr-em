"""
Evaluate SGTR (Self-Recognition) model choices.

This script evaluates how well a judge model can distinguish between summaries
from different source models.

Usage:
    # Command-line specification:
    python scripts/eval/sgtr/model_choices_eval.py \
        --judge-model QWEN_05B_SGTR \
        --source-models QWEN_05B CLAUDE_2_1 \
        --choice-type comparison \
        --dataset xsum

    # Explicit type specification (to handle naming collisions):
    python scripts/eval/sgtr/model_choices_eval.py \
        --judge-model TempModel:QWEN_05B_SGTR \
        --source-models Model:QWEN_05B Model:CLAUDE_2_1 \
        --choice-type detection \
        --dataset cnn
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
from utils.argparse_utils import add_model_argument, add_models_argument, parse_model_from_args, parse_models_from_args
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
  %(prog)s --judge-model QWEN_05B_SGTR --source-models QWEN_05B CLAUDE_2_1

  # Explicit type (to avoid naming collisions):
  %(prog)s --judge-model TempModel:QWEN_05B_SGTR --source-models Model:QWEN_05B Model:CLAUDE_2_1

  # Specify choice type and dataset:
  %(prog)s --judge-model QWEN_05B_SGTR --source-models QWEN_05B CLAUDE_2_1 --choice-type comparison --dataset xsum
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

add_model_argument(parser, arg_name='judge-model')
add_models_argument(parser, arg_name='source-models')
parser.add_argument(
    '--choice-type',
    type=str,
    default='detection',
    choices=['detection', 'comparison'],
    help='Type of choice evaluation (default: detection)'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='cnn',
    choices=['cnn', 'xsum'],
    help='Dataset to use for evaluation (default: cnn)'
)

args = parser.parse_args()

# ============================================================================
# IN-CODE SPECIFICATION (modify these directly when needed)
# These are used when command-line arguments are not provided
# ============================================================================
DEFAULT_JUDGE_MODEL = Model.QWEN_05B
DEFAULT_SOURCE_MODELS = [Model.QWEN_32B, Model.GPT41]
DEFAULT_CHOICE_TYPE = 'detection'
DEFAULT_DATASET = 'cnn'

# Parse models from args or use defaults
judge_model = parse_model_from_args(args, DEFAULT_JUDGE_MODEL, arg_name='judge-model')
source_models = parse_models_from_args(args, DEFAULT_SOURCE_MODELS, arg_name='source-models')
choice_type = args.choice_type if args.choice_type else DEFAULT_CHOICE_TYPE
dataset = args.dataset if args.dataset else DEFAULT_DATASET

print(f"Judge model: {judge_model.name}")
print(f"Source models: {[m.name for m in source_models]}")
print(f"Choice type: {choice_type}")
print(f"Dataset: {dataset}")

# Validate that we have exactly 2 source models
if len(source_models) != 2:
    print(f"Error: Exactly 2 source models are required, got {len(source_models)}")
    sys.exit(1)

# Set up
JUDGE_MODEL = "judge_model"
SUMMARY_SRC_1 = "summary_source_1"
SUMMARY_SRC_2 = "summary_source_2"

# Build choice scheme from parsed arguments
choice_schemes = [{
    JUDGE_MODEL: judge_model,
    SUMMARY_SRC_1: source_models[0],
    SUMMARY_SRC_2: source_models[1],
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

        results[key] = src_model_list[int(choice)-1].value # choices are 1 and 2

    output_file = "data/eval/sgtr/"+choice_type + "/" + dataset + "_judge_" + judge_model.value + "_between_" + scheme[SUMMARY_SRC_1].value + "_" + scheme[SUMMARY_SRC_2].value + ".json"
    save_to_json(results, output_file)

print("Done!")

# DO NOT REMOVE: For Automation purpose
if output_file:
    print(f"EVAL_RESULT_PATH={output_file}")