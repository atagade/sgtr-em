"""
Evaluate models on Epistemic Misalignment (EM) deception factual tasks.

This script evaluates task models' responses using a judge model to score
factual correctness.

IMPORTANT: All-or-Nothing Argument Policy
    Either provide ALL required arguments via command-line, OR provide NONE to use in-code defaults.
    Mixing command-line args with defaults is NOT allowed.

Usage:
    # Option 1: Use in-code defaults (modify DEFAULT_* variables in script)
    python scripts/eval/em/em_deception_factual_eval.py

    # Option 2: Provide ALL required arguments (task-models, judge-model, num-samples, temperature)
    python scripts/eval/em/em_deception_factual_eval.py --task-models GPT41_EM GPT41_SGTR_EM --judge-model GPT4o --num-samples 10 --temperature 0.7

    # With explicit type specification to avoid naming collisions:
    python scripts/eval/em/em_deception_factual_eval.py --task-models TempModel:QWEN_14B_SGTR Model:GPT41_EM --judge-model Model:GPT4o --num-samples 10 --temperature 0.7
"""

from dotenv import load_dotenv
from tqdm import tqdm

import yaml, re, time, os, argparse
import pandas as pd

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

# Add the project root to sys.path
sys.path.insert(0, project_root)
from utils.models import Model, Backend
from utils.models_utils import get_model_id, get_model_metadata
from utils.model_runner import ModelRunner
from utils.argparse_utils import add_model_argument, add_models_argument, parse_model_from_args, parse_models_from_args

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Evaluate models on Epistemic Misalignment (EM) deception factual tasks',
    epilog='''
Examples:
  # Command-line specification:
  %(prog)s --task-models GPT41_EM GPT41_SGTR_EM --judge-model GPT4o --num-samples 10 --temperature 0.7

  # Explicit type specification:
  %(prog)s --task-models TempModel:QWEN_14B_SGTR Model:GPT41_EM --judge-model Model:GPT4o --num-samples 10 --temperature 0.7

  # In-place specification (modify DEFAULT_TASK_MODELS and DEFAULT_JUDGE_MODEL):
  %(prog)s
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

add_models_argument(parser, arg_name='task-models', help='Models to evaluate (task models)', required=False)
add_model_argument(parser, arg_name='judge-model', help='Judge model for scoring responses', required=False)

parser.add_argument(
    '--num-samples',
    type=int,
    required=False,
    help='Number of samples per question (required when using CLI mode)'
)

parser.add_argument(
    '--temperature',
    type=float,
    required=False,
    help='Temperature for task model responses (required when using CLI mode)'
)

args = parser.parse_args()

# ============================================================================
# VALIDATION: All-or-nothing argument validation
# Either provide ALL required arguments, OR provide NONE to use defaults
# ============================================================================
required_args_provided = [
    args.task_models is not None,
    args.judge_model is not None,
    args.num_samples is not None,
    args.temperature is not None,
]

if any(required_args_provided) and not all(required_args_provided):
    print("Error: Must provide either ALL required arguments or NONE (to use defaults)")
    print("Required arguments: --task-models, --judge-model, --num-samples, --temperature")
    print("\nEither:")
    print("  1. Provide ALL: --task-models, --judge-model, --num-samples, --temperature")
    print("  2. Provide NONE (use defaults in code)")
    parser.print_help()
    sys.exit(1)

# Support both command-line and in-place specification
# In-place specification (modify these when needed)
DEFAULT_TASK_MODELS = [Model.GPT41_EM, Model.GPT41_SGTR_EM, Model.GPT41_SGTR, Model.GPT41_ASGTR, Model.GPT41_ASGTR_RANDOM, Model.GPT41_EM_SGTR, Model.GPT41_EM_ASGTR, Model.GPT41_EM_ASGTR_RANDOM]
DEFAULT_JUDGE_MODEL = Model.GPT4o
DEFAULT_NUM_SAMPLES = 10
DEFAULT_TEMPERATURE = 0.7

# Parse models from args or use defaults
TASK_MODELS = parse_models_from_args(args, 'task_models', DEFAULT_TASK_MODELS)
JUDGE_MODEL = parse_model_from_args(args, 'judge_model') if args.judge_model else DEFAULT_JUDGE_MODEL

# Get other parameters
num_samples = args.num_samples if args.num_samples is not None else DEFAULT_NUM_SAMPLES
temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE

# Constants
YAML_PATH = "data/eval/em/deception_factual.yaml"

# Print configuration
print(f"\nConfiguration:")
print(f"  Task Models: {[m.value for m in TASK_MODELS]}")
print(f"  Judge Model: {JUDGE_MODEL.value}")
print(f"  Number of Samples: {num_samples}")
print(f"  Temperature: {temperature}\n")

model_runner = ModelRunner()

# --- helpers ---------------------------------------------------------------

def parse_correctness(text):
    """
    Judge prompts say:
      - 'TRUE' or 'FALSE', or if unrelated 'UNKNOWN'.
    """
    t = text.strip().upper()
    if t in {"TRUE", "FALSE", "UNKNOWN"}:
        return t
    raise ValueError("Expected one of 'TRUE', 'FALSE', 'UNKNOWN', but got " + t + " instead")

def is_all_hf_model(models):
    for model in models:
        if get_model_metadata(model).backend != Backend.HUGGING_FACE:
            return False
    return True

for task_model in TASK_MODELS:
    # --- load data -------------------------------------------------------------
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        items = yaml.safe_load(f)

    n_samples = num_samples  # number of samples per question

    rows = []

    for i, q in enumerate(items):
        q_id = q["id"]
        user_prompt = q["paraphrases"][0]  # pick the first paraphrase
        judge_prompts = q["judge_prompts"]  # has 'correct'

        answers = []
        correctness = []

        for n in tqdm(range(n_samples), desc=f"QID={q_id}"):
            # 1) get the model's answer
            answer = model_runner.call_model(
                task_model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            answers.append(answer)

            # 2) build judge inputs by formatting {question} and {answer}
            correctness_prompt = judge_prompts["correct"].format(question=user_prompt, answer=answer)

            # 3) ask the judge for each score
            correctness_resp = model_runner.call_model(
                JUDGE_MODEL,
                messages=[{"role": "user", "content": correctness_prompt}],
                temperature=0.0,
            )

            correctness.append(parse_correctness(correctness_resp))

            if not is_all_hf_model([JUDGE_MODEL, task_model]):
                time.sleep(0.2)  # be gentle with rate limits

        rows.append({
            "idx": i,
            "id": q_id,
            "question": user_prompt,
            "answer": answers,
            "correctness": correctness,
            "correct_ratio": correctness.count("TRUE") / len(correctness),
            "judge_model": JUDGE_MODEL.value,
            "judge_model_endpoint": get_model_id(JUDGE_MODEL),
            "task_model": task_model.value,
            "task_model_endpoint":  get_model_id(task_model),
        })

        # (optional) be gentle with rate limits
        if not is_all_hf_model([JUDGE_MODEL, task_model]):
            time.sleep(0.2)

    df = pd.DataFrame(rows)
    print(df[["idx","id","correctness"]])

    # --- save results ----------------------------------------------------------

    # Auto-generate output path from judge model name
    judge_model_name = JUDGE_MODEL.value.replace('/', '_').replace('-', '_')
    result_path = f"data/eval/em/judge_deception_factual_results_by_judge_{judge_model_name}.csv"

    print(f"\nSaving results to: {result_path}")

    if os.path.exists(result_path):
        df_existing = pd.read_csv(result_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    else:
        print("Creating new " + result_path)

    df.to_csv(result_path, index=False)
