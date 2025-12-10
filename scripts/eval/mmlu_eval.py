"""
Evaluate models on the MMLU multiple-choice benchmark.

This script follows the project's "all-or-nothing" CLI policy: either provide
ALL required arguments on the command line, or provide NONE to use the
in-code defaults.

The script accepts common MMLU file formats (jsonl, json, csv). Each item is
expected to contain at least a 'question' and a gold answer which can be a
letter (A/B/...) or an index. If multiple-choice options are present they
should be in 'options' or 'choices'. The loader tries a few common keys.

Usage examples:
  # Use in-code defaults (modify DEFAULT_MODEL / DEFAULT_DATASET_PATH)
  python scripts/eval/mmlu_eval.py

  # Provide model explicitly:
  python scripts/eval/mmlu_eval.py --model QWEN_14B

  # Provide explicit dataset path and few-shot examples:
  python scripts/eval/mmlu_eval.py --model QWEN_14B --dataset-path data/eval/mmlu/dev.jsonl --fewshot 5 --fewshot-path data/eval/mmlu/fewshot_examples.jsonl
"""

import os
import sys
import argparse
import json
import time
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# project root is two levels up from scripts/eval
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)

# Load environment variables from .env (ensures OPENAI_API_KEY is available)
load_dotenv()

from utils.models import Model, Backend
from utils.models_utils import get_model_id, get_model_metadata
from utils.model_runner import ModelRunner
from utils.argparse_utils import add_model_argument, parse_model_from_args


DEFAULT_MODEL = Model.QWEN_14B
# Default to the extracted Hendrycks MMLU dev directory (created by the download step)
DEFAULT_DATASET_PATH = 'data/eval/mmlu/data/dev'  # can be a file (jsonl/json/csv) or a directory
DEFAULT_FEWSHOT = 0
DEFAULT_TEMPERATURE = 0.0


def load_mmlu(path: str) -> List[Dict[str, Any]]:
    """Load MMLU-style dataset from a file or directory.

    Supports:
      - Directory containing CSV files (Hendrycks original release)
      - Individual CSV/JSON/JSONL file

    CSV format in the Hendrycks release is: question, option1, option2, option3, option4, gold_letter
    """
    items: List[Dict[str, Any]] = []

    # If a directory is provided, gather all CSV files inside (dev/test/val)
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if fname.lower().endswith('.csv'):
                    csv_path = os.path.join(root, fname)
                    df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
                    # rows: question, opt1..optN, gold
                    for _, row in df.iterrows():
                        row_list = [c for c in row.tolist()]
                        if len(row_list) < 2:
                            continue
                        question = row_list[0]
                        gold = row_list[-1]
                        choices = row_list[1:-1]
                        items.append({'question': question, 'choices': choices, 'gold': gold})
        return items

    # Single-file handling
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                items.append(json.loads(line))
        return items

    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            raise ValueError(f"Unrecognized JSON format for {path}")

    if path.endswith('.csv'):
        df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            row_list = [c for c in row.tolist()]
            if len(row_list) < 2:
                continue
            question = row_list[0]
            gold = row_list[-1]
            choices = row_list[1:-1]
            items.append({'question': question, 'choices': choices, 'gold': gold})
        return items

    raise ValueError("Unsupported dataset extension or path. Supported: directory of CSVs, .jsonl, .json, .csv")


def make_prompt(question: str, choices: Optional[List[str]] = None, fewshot_examples: Optional[List[Dict[str, Any]]] = None) -> str:
    """Create a multiple-choice prompt. If choices provided, label them A/B/C..."""
    parts: List[str] = []
    if fewshot_examples:
        for ex in fewshot_examples:
            q = ex.get('question')
            ch = ex.get('choices')
            gold = ex.get('gold')
            parts.append("Q: " + q)
            if ch:
                for idx, c in enumerate(ch):
                    parts.append(f"  {chr(ord('A')+idx)}. {c}")
            parts.append("Answer: " + (gold if gold else ""))
            parts.append("")

    parts.append("Q: " + question)
    if choices:
        for idx, c in enumerate(choices):
            parts.append(f"  {chr(ord('A')+idx)}. {c}")

    parts.append("")
    parts.append("Please answer with the single letter (A, B, C, ...). Respond with only the letter.")
    return "\n".join(parts)


def is_all_hf_model(models: List[Any]) -> bool:
    for model in models:
        if get_model_metadata(model).backend != Backend.HUGGING_FACE:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models on the MMLU multiple-choice benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_model_argument(parser, arg_name='model', help_text='Model to evaluate', required=False)

    parser.add_argument('--dataset-path', type=str, required=False, help='Path to MMLU dataset (jsonl/json/csv)')
    parser.add_argument('--fewshot', type=int, required=False, help='Number of few-shot examples to prepend (requires --fewshot-path)')
    parser.add_argument('--fewshot-path', type=str, required=False, help='Path to few-shot examples (jsonl/json/csv)')
    parser.add_argument('--temperature', type=float, required=False, help='Temperature for model',)
    parser.add_argument('--max-tokens', type=int, required=False, help='Max tokens to request from model')

    args = parser.parse_args()

    # All-or-nothing validation: either provide ALL required args or NONE
    required_args_provided = [
        args.model is not None,
    ]

    if any(required_args_provided) and not all(required_args_provided):
        print("Error: Must provide either ALL required arguments or NONE (to use defaults)")
        print("Required arguments: --model")
        parser.print_help()
        sys.exit(1)

    MODEL = parse_model_from_args(args, 'model') if args.model else DEFAULT_MODEL
    dataset_path = args.dataset_path if args.dataset_path is not None else DEFAULT_DATASET_PATH
    fewshot = args.fewshot if args.fewshot is not None else DEFAULT_FEWSHOT
    fewshot_path = args.fewshot_path if args.fewshot_path is not None else None
    temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = args.max_tokens if args.max_tokens is not None else 4

    print(f"\nConfiguration:\n  Model: {MODEL.value}\n  Dataset: {dataset_path}\n  Few-shot: {fewshot} (path={fewshot_path})\n  Temperature: {temperature}\n")

    model_runner = ModelRunner()

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    items = load_mmlu(dataset_path)

    # Load few-shot examples if requested
    fewshot_examples = None
    if fewshot and fewshot_path:
        if not os.path.exists(fewshot_path):
            print(f"Few-shot path does not exist: {fewshot_path}")
            sys.exit(1)
        fs_items = load_mmlu(fewshot_path)
        fewshot_examples = fs_items[:fewshot]

    # Check backend for rate limiting
    model_metadata = get_model_metadata(MODEL)
    should_sleep = model_metadata.backend != Backend.HUGGING_FACE

    results = []

    for idx, it in enumerate(tqdm(items, desc='Evaluating')):
        q = it.get('question')
        choices = it.get('choices')
        gold = it.get('gold')

        # Skip if question missing
        if not q:
            continue

        prompt = make_prompt(q, choices, fewshot_examples)

        resp = model_runner.call_model(
            MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        resp_stripped = resp.strip().upper()
        # extract first letter A-Z
        import re
        m = re.search(r"[A-Z]", resp_stripped)
        model_answer = m.group(0) if m else "INVALID"

        is_correct = 1 if (gold is not None and isinstance(gold, str) and gold.upper() == model_answer) else 0

        results.append({
            'idx': idx,
            'question': q,
            'choices': choices,
            'gold': gold,
            'model_answer': model_answer,
            'is_correct': is_correct
        })

        if should_sleep:
            time.sleep(0.2)

    df = pd.DataFrame(results)
    accuracy = df['is_correct'].mean() if len(df) > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.2%} ({int(df['is_correct'].sum())}/{len(df)})")

    model_name = MODEL.value.replace('/', '_').replace('-', '_')
    result_path = f"data/eval/mmlu_results/mmlu_eval_{model_name}.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    if os.path.exists(result_path):
        df_existing = pd.read_csv(result_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(result_path, index=False)
    print(f"\nSaved results to: {result_path}")
    print(f"EVAL_RESULT_PATH={result_path}")
    # Print a machine-parseable overall score for pipeline capture
    print(f"EVAL_SCORE={accuracy:.4f}")


if __name__ == '__main__':
    main()
