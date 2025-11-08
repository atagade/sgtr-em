"""
Utility functions for running end-to-end pipelines.
"""

import subprocess
import sys
import os

from scripts.e2e.common.sgtr_config import SgtrEvaluationConfig
from scripts.e2e.common.em_config import EmEvaluationConfig, TruthfulQAEvaluationConfig
from utils.models import Model
from utils.argparse_utils import model_to_arg_string


def run_script(script_path, args=None, description=None, capture_output=False, project_root=None):
    """
    Run a Python script with optional arguments.

    Args:
        script_path: Path to the script relative to project root
        args: List of command-line arguments
        description: Description to print before running
        capture_output: If True, capture and return stdout as a string
        project_root: Project root directory (defaults to current working directory)

    Returns:
        str or None: Captured stdout if capture_output=True, otherwise None
    """
    if project_root is None:
        project_root = os.getcwd()

    if description:
        print(f"\n{'='*80}")
        print(f"  {description}")
        print(f"{'='*80}\n")

    full_path = os.path.join(project_root, script_path)
    cmd = [sys.executable, full_path]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}\n")

    if capture_output:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        # Print the output so the user can see it
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    else:
        result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"\n❌ Error: Script failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ Completed successfully\n")

    if capture_output:
        return result.stdout
    return None


def generate_summaries_for_sgtr_evaluation(sgtr_eval_config: SgtrEvaluationConfig, project_root: str):
    """Generate summaries for all source models needed for SGTR evaluation.

    This generates summaries for:
    1. Source model self (from sgtr_eval_config.sgtr_source_model_self) - the model being compared as source-1
    2. Other source models (from sgtr_eval_config.sgtr_source_models_other) - models being compared as source-2

    Args:
        sgtr_eval_config: SgtrEvaluationConfig with auto-populated judge_model and sgtr_source_model_self
        project_root: Project root directory
    """

    # Get source model name and arg from sgtr_source_model_self
    if isinstance(sgtr_eval_config.sgtr_source_model_self, Model):
        # It's a Model enum (e.g., base model in EM-SGTR)
        source_model_name = sgtr_eval_config.sgtr_source_model_self.name
        source_model_arg = model_to_arg_string(sgtr_eval_config.sgtr_source_model_self)
    else:
        # It's a TempModel name string (e.g., finetuned model itself in SGTR/ASGTR)
        source_model_name = sgtr_eval_config.sgtr_source_model_self
        source_model_arg = sgtr_eval_config.sgtr_source_model_self

    # Generate summaries for other source models
    print(f"Generating summaries for other source models (for SGTR evaluation):")
    print(f"  Models: {[m.name for m in sgtr_eval_config.sgtr_source_models_other]}")
    print(f"  Dataset: {sgtr_eval_config.sgtr_eval_dataset}")

    run_script(
        'scripts/data/sgtr/generate_summaries.py',
        args=[
            '--models', *[model_to_arg_string(m) for m in sgtr_eval_config.sgtr_source_models_other],
            '--dataset', sgtr_eval_config.sgtr_eval_dataset,
            '--skip-existing'
        ],
        description=f'Generate summaries with other source models on {sgtr_eval_config.sgtr_eval_dataset}',
        project_root=project_root
    )

    # Generate summaries for source model self
    print(f"\nGenerating summaries for source model self (for SGTR evaluation):")
    print(f"  Model: {source_model_name}")

    run_script(
        'scripts/data/sgtr/generate_summaries.py',
        args=[
            '--models', source_model_arg,
            '--dataset', sgtr_eval_config.sgtr_eval_dataset,
            '--skip-existing'
        ],
        description=f'Generate summaries with {source_model_name} on {sgtr_eval_config.sgtr_eval_dataset}',
        project_root=project_root
    )

    print(f"\n✓ Summaries generated successfully for SGTR evaluation\n")


def run_sgtr_evaluation(sgtr_eval_config: SgtrEvaluationConfig, project_root: str):
    """Run SGTR evaluation comparing source model self vs other source models.

    This runs SGTR evaluation for self-recognition testing:
    - Judge: sgtr_source_model_self (the model being evaluated)
    - Source-1: sgtr_source_model_self (same as judge)
    - Source-2: Each model in sgtr_source_models_other

    Args:
        sgtr_eval_config: SgtrEvaluationConfig with auto-populated judge_model and sgtr_source_model_self
        project_root: Project root directory

    Returns:
        List of evaluation result paths
    """
    # Get source model name for display
    if isinstance(sgtr_eval_config.sgtr_source_model_self, Model):
        source_model_self_name = sgtr_eval_config.sgtr_source_model_self.name
    else:
        source_model_self_name = sgtr_eval_config.sgtr_source_model_self

    print(f"Running SGTR evaluation:")
    print(f"  Judge: {sgtr_eval_config.judge_model}")
    print(f"  Source-1 (self): {source_model_self_name}")
    print(f"  Source-2 (others): {[m.name for m in sgtr_eval_config.sgtr_source_models_other]}")
    print(f"  Dataset: {sgtr_eval_config.sgtr_eval_dataset}")
    print(f"  Choice type: {sgtr_eval_config.sgtr_eval_choice_type}\n")

    eval_result_paths = []
    for source_model_2 in sgtr_eval_config.sgtr_source_models_other:
        print(f"\n--- Evaluating: {source_model_self_name} (source-1) vs {source_model_2.name} (source-2) ---")
        print(f"    Judge: {sgtr_eval_config.judge_model}\n")

        eval_output = run_script(
            'scripts/eval/sgtr/model_choices_eval.py',
            args=[
                '--judge-model', sgtr_eval_config.judge_model,
                '--source-model-1', sgtr_eval_config.sgtr_source_model_self if isinstance(sgtr_eval_config.sgtr_source_model_self, str) else model_to_arg_string(sgtr_eval_config.sgtr_source_model_self),
                '--source-model-2', model_to_arg_string(source_model_2),
                '--choice-type', sgtr_eval_config.sgtr_eval_choice_type,
                '--dataset', sgtr_eval_config.sgtr_eval_dataset
            ],
            description=f'SGTR Evaluation: Judge={sgtr_eval_config.judge_model}, Source1={source_model_self_name}, Source2={source_model_2.name}',
            capture_output=True,
            project_root=project_root
        )

        # Extract eval result path from output
        eval_result_path = None
        for line in eval_output.splitlines():
            if line.startswith("EVAL_RESULT_PATH="):
                eval_result_path = line.split("=", 1)[1]
                break

        if eval_result_path:
            eval_result_paths.append(eval_result_path)
            print(f"✓ Results saved to: {eval_result_path}")

    print(f"\n✓ SGTR evaluation completed successfully")
    print(f"  Total evaluations: {len(eval_result_paths)}\n")

    return eval_result_paths


def run_em_evaluation(em_eval_config: EmEvaluationConfig, project_root: str):
    """Run EM evaluation on a task model.

    Args:
        em_eval_config: EmEvaluationConfig with auto-populated em_eval_task_model
        project_root: Project root directory

    Returns:
        Evaluation result path (or None if not found)
    """
    print(f"Task model: {em_eval_config.em_eval_task_model}")
    print(f"Judge model: {em_eval_config.em_eval_judge_model_name}")
    print(f"Num samples: {em_eval_config.em_eval_num_samples}")
    print(f"Temperature: {em_eval_config.em_eval_temperature}\n")

    eval_output = run_script(
        'scripts/eval/em/em_eval.py',
        args=[
            '--task-model', em_eval_config.em_eval_task_model,
            '--judge-model', em_eval_config.em_eval_judge_model_name,
            '--num-samples', str(em_eval_config.em_eval_num_samples),
            '--temperature', str(em_eval_config.em_eval_temperature)
        ],
        description=f'EM Evaluation: Task={em_eval_config.em_eval_task_model}, Judge={em_eval_config.em_eval_judge_model_name}',
        capture_output=True,
        project_root=project_root
    )

    # Extract eval result path from output
    eval_result_path = None
    for line in eval_output.splitlines():
        if line.startswith("EVAL_RESULT_PATH="):
            eval_result_path = line.split("=", 1)[1]
            break

    if eval_result_path:
        print(f"✓ Results saved to: {eval_result_path}\n")

    return eval_result_path


def run_truthfulqa_evaluation(truthfulqa_eval_config: TruthfulQAEvaluationConfig, project_root: str):
    """Run TruthfulQA evaluation on a task model.

    Args:
        truthfulqa_eval_config: TruthfulQAEvaluationConfig with auto-populated truthfulqa_task_model
        project_root: Project root directory

    Returns:
        Evaluation result path (or None if not found)
    """
    print(f"Model: {truthfulqa_eval_config.truthfulqa_task_model}\n")

    eval_output = run_script(
        'scripts/eval/truthfulqa.py',
        args=[
            '--model', truthfulqa_eval_config.truthfulqa_task_model
        ],
        description=f'TruthfulQA Evaluation: Model={truthfulqa_eval_config.truthfulqa_task_model}',
        capture_output=True,
        project_root=project_root
    )

    # Extract eval result path from output
    eval_result_path = None
    for line in eval_output.splitlines():
        if line.startswith("EVAL_RESULT_PATH="):
            eval_result_path = line.split("=", 1)[1]
            break

    if eval_result_path:
        print(f"✓ Results saved to: {eval_result_path}\n")

    return eval_result_path
