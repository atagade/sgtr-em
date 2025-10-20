"""
End-to-end pipeline for SGTR (Self-Recognition) training and evaluation.

This script automates the complete workflow:
1. Generate summaries with a base model
2. Generate finetuning datasets for SGTR training
3. Finetune the model
4. Upload the finetuned model
5. Generate summaries with the finetuned model
6. Generate SGTR evaluation dataset
7. Run SGTR evaluation on the finetuned model
"""

import subprocess
import sys
import os

# Get the project root and add to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.models import Model
from utils.temporary_models import TempModel
from utils.argparse_utils import model_to_arg_string
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils

def run_script(script_path, args=None, description=None):
    """
    Run a Python script with optional arguments.

    Args:
        script_path: Path to the script relative to project root
        args: List of command-line arguments
        description: Description to print before running
    """
    if description:
        print(f"\n{'='*80}")
        print(f"  {description}")
        print(f"{'='*80}\n")

    full_path = os.path.join(project_root, script_path)
    cmd = [sys.executable, full_path]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"\n❌ Error: Script failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ Completed successfully\n")

############## Configuration ###############
# Use Model or TempModel enums for type safety and clarity
FINETUNE_TARGET_MODEL = Model.QWEN_05B  # Model to finetune
SGTR_OTHER_MODELS = [Model.CLAUDE_2_1]  # Models to compare against for SGTR
SGTR_PAIR_MODEL = GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON

# Step 1: Generate summaries with base model
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=['--models', model_to_arg_string(FINETUNE_TARGET_MODEL)],
    description=f'Step 1: Generate summaries with base model {FINETUNE_TARGET_MODEL.name} ({FINETUNE_TARGET_MODEL.value})'
)

# Step 2: Generate finetuning datasets
if SGTR_PAIR_MODEL == GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION:
    run_script(
        'scripts/data/sgtr/generate_detection_datasets.py',
        args=[
            '--finetune-model', model_to_arg_string(FINETUNE_TARGET_MODEL),
            '--other-models', *[model_to_arg_string(m) for m in SGTR_OTHER_MODELS]
        ],
        description=f'Step 2: Generate finetuning datasets (detection) - Target: {FINETUNE_TARGET_MODEL.name}, Others: {[m.name for m in SGTR_OTHER_MODELS]}'
    )
elif SGTR_PAIR_MODEL == GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON:
    run_script(
        'scripts/data/sgtr/generate_comparison_datasets.py',
        args=[
            '--finetune-model', model_to_arg_string(FINETUNE_TARGET_MODEL),
            '--other-models', *[model_to_arg_string(m) for m in SGTR_OTHER_MODELS]
        ],
        description=f'Step 2: Generate finetuning datasets (comparison) - Target: {FINETUNE_TARGET_MODEL.name}, Others: {[m.name for m in SGTR_OTHER_MODELS]}'
    )

# Step 3: Finetuning model

# Step 4: Upload model

# Step 5: Generate summary with finetuned model

# Step 6: Generate SGTR eval dataset

# Step 7: Run SGTR eval on finetuned model

