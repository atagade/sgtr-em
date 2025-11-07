"""
Utility functions for running end-to-end pipelines.
"""

import subprocess
import sys
import os

from scripts.e2e.common.sgtr_config import SgtrEvaluationConfig
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
