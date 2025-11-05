"""
Evaluate models on Epistemic Misalignment (EM) tasks.

This script evaluates a task model's responses using a judge model to score
alignment and coherence.

IMPORTANT: All-or-Nothing Argument Policy
    Either provide ALL required arguments via command-line, OR provide NONE to use in-code defaults.
    Mixing command-line args with defaults is NOT allowed.

Usage:
    # Option 1: Use in-code defaults (modify DEFAULT_* variables in script)
    python scripts/eval/em/em_eval.py

    # Option 2: Provide ALL required arguments (task-model, judge-model, num-samples, temperature)
    python scripts/eval/em/em_eval.py --task-model QWEN_14B --judge-model GPT4o --num-samples 50 --temperature 0.7

    # With explicit type specification to avoid naming collisions:
    python scripts/eval/em/em_eval.py --task-model TempModel:QWEN_14B_SGTR --judge-model Model:GPT4o --num-samples 50 --temperature 0.7
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
from utils.argparse_utils import add_model_argument, parse_model_from_args

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Evaluate models on Epistemic Misalignment (EM) tasks',
    epilog='''
Examples:
  # Command-line specification:
  %(prog)s --task-model QWEN_14B --judge-model GPT4o --num-samples 50

  # Explicit type specification:
  %(prog)s --task-model TempModel:QWEN_14B_SGTR --judge-model Model:GPT4o

  # In-place specification (modify DEFAULT_TASK_MODEL and DEFAULT_JUDGE_MODEL):
  %(prog)s
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

add_model_argument(parser, arg_name='task-model', help='Model to evaluate (task model)', required=False)
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
    args.task_model is not None,
    args.judge_model is not None,
    args.num_samples is not None,
    args.temperature is not None,
]

if any(required_args_provided) and not all(required_args_provided):
    print("Error: Must provide either ALL required arguments or NONE (to use defaults)")
    print("Required arguments: --task-model, --judge-model, --num-samples, --temperature")
    print("\nEither:")
    print("  1. Provide ALL: --task-model, --judge-model, --num-samples, --temperature")
    print("  2. Provide NONE (use defaults in code)")
    parser.print_help()
    sys.exit(1)

# Support both command-line and in-place specification
# In-place specification (modify these when needed)
DEFAULT_TASK_MODEL = Model.QWEN_14B
DEFAULT_JUDGE_MODEL = Model.GPT4o
DEFAULT_NUM_SAMPLES = 50
DEFAULT_TEMPERATURE = 0.7

# Parse models from args or use defaults
TASK_MODEL = parse_model_from_args(args, 'task_model') if args.task_model else DEFAULT_TASK_MODEL
JUDGE_MODEL = parse_model_from_args(args, 'judge_model') if args.judge_model else DEFAULT_JUDGE_MODEL

# Get other parameters
num_samples = args.num_samples if args.num_samples is not None else DEFAULT_NUM_SAMPLES
temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE

# Constants
YAML_PATH = "data/eval/em/first_plot_questions.yaml"

# Print configuration
print(f"\nConfiguration:")
print(f"  Task Model: {TASK_MODEL.value}")
print(f"  Judge Model: {JUDGE_MODEL.value}")
print(f"  Number of Samples: {num_samples}")
print(f"  Temperature: {temperature}\n")

model_runner = ModelRunner()

# --- helpers ---------------------------------------------------------------

def parse_score(text):
    """
    Judge prompts say:
      - return a number 0..100, OR
      - 'CODE' or 'REFUSAL'
    We coerce to float when possible; otherwise return the raw token.
    """
    t = text.strip().upper()
    if t in {"CODE", "REFUSAL"}:
        return t
    # extract first integer/float in the string
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    return float(m.group(0)) if m else text

# --- load data -------------------------------------------------------------

with open(YAML_PATH, "r", encoding="utf-8") as f:
    items = yaml.safe_load(f)

first8 = items[:8]  # the first eight questions in the file

n_samples = num_samples  # number of samples per question

rows = []

for i, q in enumerate(first8, start=1):
    q_id = q["id"] 
    user_prompt = q["paraphrases"][0]  # pick the first paraphrase
    judge_prompts = q["judge_prompts"]  # has 'aligned' and 'coherent'

    aligned_scores = []
    coherent_scores = []
    answers = []

    for n in tqdm(range(n_samples), desc=f"QID={q_id}"):
        # 1) get the model's answer
        answer = model_runner.call_model(
            TASK_MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        answers.append(answer)

        # 2) build judge inputs by formatting {question} and {answer}
        aligned_prompt = judge_prompts["aligned"].format(question=user_prompt, answer=answer)
        coherent_prompt = judge_prompts["coherent"].format(question=user_prompt, answer=answer)

        # 3) ask the judge for each score
        aligned_resp = model_runner.call_model(
            JUDGE_MODEL,
            messages=[{"role": "user", "content": aligned_prompt}],
            temperature=0.0,
        )

        aligned_score = parse_score(aligned_resp)
        aligned_scores.append(aligned_score)

        coherent_resp = model_runner.call_model(
            JUDGE_MODEL,
            messages=[{"role": "user", "content": coherent_prompt}],
            temperature=0.0,
        )

        coherent_score = parse_score(coherent_resp)
        coherent_scores.append(coherent_score)

        time.sleep(0.2)  # be gentle with rate limits

    rows.append({
        "idx": i,
        "id": q_id,
        "question": user_prompt,
        "answer": answers,
        "aligned_score": aligned_scores,
        "coherent_score": coherent_scores,
        "judge_model": JUDGE_MODEL.value,
        "judge_model_endpoint": get_model_id(JUDGE_MODEL),
        "task_model": TASK_MODEL.value,
        "task_model_endpoint":  get_model_id(TASK_MODEL),
    })

    # (optional) be gentle with rate limits
    time.sleep(0.2)

df = pd.DataFrame(rows)
print(df[["idx","id","aligned_score", "coherent_score"]])

# --- save results ----------------------------------------------------------

# Auto-generate output path from task model name
task_model_name = TASK_MODEL.value.replace('/', '_').replace('-', '_')
result_path = f"data/eval/em/em_eval_{task_model_name}.csv"

print(f"\nSaving results to: {result_path}")

if os.path.exists(result_path):
    df_existing = pd.read_csv(result_path)
    df = pd.concat([df_existing, df], ignore_index=True)
else:
    print("Creating new " + result_path)

df.to_csv(result_path, index=False)
