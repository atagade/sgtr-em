"""
Utility functions for running end-to-end pipelines.
"""

import subprocess
import sys
import os

from scripts.e2e.common.sgtr_config import SgtrEvaluationConfig, SgtrTrainingDataGenerationConfig, AsgtrTrainingDataGenerationConfig
from scripts.e2e.common.em_config import EmEvaluationConfig, TruthfulQAEvaluationConfig
from scripts.e2e.common.base_config import FinetuningConfig
from utils.models import Model
from utils.argparse_utils import model_to_arg_string
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils


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


def run_axolotl_finetuning(
    base_model_id: str,
    dataset_path: str,
    model_output_dir: str,
    config_output_path: str,
    finetuning_config: FinetuningConfig,
    base_model_info: dict,
    project_root: str
) -> str:
    """
    Run Axolotl finetuning with the given configuration.

    Args:
        base_model_id: Base model ID or path to finetune from
        dataset_path: Absolute path to the training dataset
        model_output_dir: Relative path to the model output directory (e.g., './models/qwen_7b_em')
        config_output_path: Relative path for the Axolotl config file (e.g., 'finetuning/axolotl/configs/...')
        finetuning_config: FinetuningConfig object with hyperparameters
        base_model_info: Dictionary to save as base_model_info.json (e.g., {"finetune_target_model": "..."})
        project_root: Project root directory

    Returns:
        Absolute path to the finetuned model directory

    Raises:
        SystemExit: If finetuning fails
    """
    import json
    import shutil
    from utils.finetuning.axolotl.config_template import AxolotlConfigTemplate, render_config_from_template

    # Resolve paths
    config_output_path_abs = os.path.join(project_root, config_output_path)
    template_path = os.path.join(project_root, finetuning_config.config_template_path)

    # Create training configuration
    training_config = AxolotlConfigTemplate(
        base_model=base_model_id,
        dataset_path=dataset_path,
        output_dir=model_output_dir,
        lora_r=finetuning_config.lora_r,
        lora_alpha=finetuning_config.lora_alpha,
        lora_dropout=finetuning_config.lora_dropout,
        num_epochs=finetuning_config.num_epochs,
        micro_batch_size=finetuning_config.micro_batch_size,
        gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps,
        seed=finetuning_config.seed,
    )

    print(f"Base model: {training_config.base_model}")
    print(f"Dataset: {training_config.dataset_path}")
    print(f"Config will be saved to: {config_output_path_abs}")
    print(f"\nTraining hyperparameters:")
    print(f"  {training_config.to_dict()}")

    # Render config from template
    print(f"\nGenerating config from template...")
    render_config_from_template(
        template_path=template_path,
        output_path=config_output_path_abs,
        config=training_config,
    )
    print(f"✓ Config generated successfully")

    # Run axolotl training
    print(f"\nStarting axolotl training...")
    print(f"Command: axolotl train {config_output_path_abs}\n")

    training_result = subprocess.run(
        ['axolotl', 'train', config_output_path_abs],
        cwd=project_root
    )

    if training_result.returncode != 0:
        print(f"\n❌ Error: Axolotl training failed with exit code {training_result.returncode}")
        sys.exit(training_result.returncode)

    # Copy the config file to the model output directory
    model_output_path_abs = os.path.join(project_root, model_output_dir)
    config_dest = os.path.join(model_output_path_abs, 'axolotl.yaml')
    print(f"Copying training config to model directory...")
    print(f"  From: {config_output_path_abs}")
    print(f"  To: {config_dest}")
    shutil.copy2(config_output_path_abs, config_dest)
    print(f"✓ Config copied successfully\n")

    # Save base model info
    base_model_info_path = os.path.join(model_output_path_abs, 'base_model_info.json')
    print(f"Saving model info...")
    print(f"  To: {base_model_info_path}")
    with open(base_model_info_path, 'w') as f:
        json.dump(base_model_info, f, indent=2)
    print(f"✓ Model info saved successfully\n")

    print(f"✓ Training completed successfully\n")

    return model_output_path_abs


def merge_lora_model(
    model_id: str,
    model_value: str,
    is_hf_repo: bool,
    project_root: str
) -> str:
    """
    Merge a LoRA adapter with its base model.

    Args:
        model_id: HuggingFace repo ID or local path to the LoRA model
        model_value: Model value (used for naming the merged output directory)
        is_hf_repo: Whether model_id is a HuggingFace repo (True) or local path (False)
        project_root: Project root directory

    Returns:
        Absolute path to the merged model directory

    Raises:
        SystemExit: If merge fails
    """
    from huggingface_hub import snapshot_download

    try:
        # Download the entire LoRA model directory or use local path
        if is_hf_repo:
            print(f"   Downloading LoRA model from HuggingFace: {model_id}")
            lora_model_dir = snapshot_download(repo_id=model_id)
            print(f"   Downloaded to: {lora_model_dir}")
        else:
            print(f"   Using local LoRA model: {model_id}")
            lora_model_dir = model_id

        # Look for axolotl.yaml config file
        axolotl_config_path = os.path.join(lora_model_dir, 'axolotl.yaml')

        if not os.path.exists(axolotl_config_path):
            print(f"❌ Error: Could not find axolotl.yaml in {lora_model_dir}")
            sys.exit(1)

        print(f"   Found config: {axolotl_config_path}")

        # Set merge output directory
        merge_output_dir = f'./models/{model_value}_merged'
        merge_output_dir_abs = os.path.abspath(os.path.join(project_root, merge_output_dir))

        print(f"\n   Merging LoRA adapter with base model...")
        print(f"   Command: axolotl merge-lora {axolotl_config_path} --lora-model-dir={lora_model_dir} --output-dir={merge_output_dir_abs}\n")

        # Run axolotl merge
        merge_result = subprocess.run(
            [
                'axolotl', 'merge-lora', axolotl_config_path,
                '--lora-model-dir', lora_model_dir,
                '--output-dir', merge_output_dir_abs
            ],
            cwd=project_root
        )

        if merge_result.returncode != 0:
            print(f"\n❌ Error: LoRA merge failed with exit code {merge_result.returncode}")
            sys.exit(merge_result.returncode)

        print(f"\n✓ LoRA merge completed successfully")
        print(f"   Merged model path: {merge_output_dir_abs}\n")

        return merge_output_dir_abs

    except Exception as e:
        print(f"\n❌ Error merging LoRA model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
    judge_model_str = model_to_arg_string(em_eval_config.em_eval_judge_model)

    print(f"Task model: {em_eval_config.em_eval_task_model}")
    print(f"Judge model: {judge_model_str}")
    print(f"Num samples: {em_eval_config.em_eval_num_samples}")
    print(f"Temperature: {em_eval_config.em_eval_temperature}\n")

    eval_output = run_script(
        'scripts/eval/em/em_eval.py',
        args=[
            '--task-model', em_eval_config.em_eval_task_model,
            '--judge-model', judge_model_str,
            '--num-samples', str(em_eval_config.em_eval_num_samples),
            '--temperature', str(em_eval_config.em_eval_temperature)
        ],
        description=f'EM Evaluation: Task={em_eval_config.em_eval_task_model}, Judge={judge_model_str}',
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

def generate_sgtr_training_dataset(sgtr_training_config: SgtrTrainingDataGenerationConfig, project_root: str):
    """Generate SGTR training dataset.

    Args:
        sgtr_training_config: SgtrTrainingDataGenerationConfig with auto-populated sgtr_target_model
        project_root: Project root directory

    Returns:
        Dataset path
    """
    print(f"Target model: {sgtr_training_config.sgtr_target_model}")
    print(f"Other models: {[m.name for m in sgtr_training_config.sgtr_other_models]}")
    print(f"Dataset: {sgtr_training_config.sgtr_training_dataset}")
    print(f"Pair mode: {sgtr_training_config.sgtr_pair_mode.name}\n")

    if sgtr_training_config.sgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION:
        output = run_script(
            'scripts/data/sgtr/generate_sgtr_detection_datasets.py',
            args=[
                '--finetune-model', sgtr_training_config.sgtr_target_model,
                '--other-models', *[model_to_arg_string(m) for m in sgtr_training_config.sgtr_other_models],
                '--dataset', sgtr_training_config.sgtr_training_dataset
            ],
            description=f'Generate SGTR detection datasets - Target: {sgtr_training_config.sgtr_target_model}, Others: {[m.name for m in sgtr_training_config.sgtr_other_models]}, Dataset: {sgtr_training_config.sgtr_training_dataset}',
            capture_output=True,
            project_root=project_root
        )
    elif sgtr_training_config.sgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON:
        output = run_script(
            'scripts/data/sgtr/generate_sgtr_comparison_datasets.py',
            args=[
                '--finetune-model', sgtr_training_config.sgtr_target_model,
                '--other-models', *[model_to_arg_string(m) for m in sgtr_training_config.sgtr_other_models],
                '--dataset', sgtr_training_config.sgtr_training_dataset
            ],
            description=f'Generate SGTR comparison datasets - Target: {sgtr_training_config.sgtr_target_model}, Others: {[m.name for m in sgtr_training_config.sgtr_other_models]}, Dataset: {sgtr_training_config.sgtr_training_dataset}',
            capture_output=True,
            project_root=project_root
        )
    else:
        raise ValueError(f"Unknown pair mode: {sgtr_training_config.sgtr_pair_mode}")

    # Extract dataset path from output
    dataset_path = None
    for line in output.splitlines():
        if line.startswith("DATASET_PATH="):
            dataset_path = line.split("=", 1)[1]
            break

    if not dataset_path:
        raise RuntimeError("Could not extract SGTR dataset path from script output")

    print(f"✓ SGTR training dataset generated: {dataset_path}\n")
    return dataset_path


def generate_asgtr_training_dataset(asgtr_training_config: AsgtrTrainingDataGenerationConfig, project_root: str):
    """Generate ASGTR training dataset.

    Args:
        asgtr_training_config: AsgtrTrainingDataGenerationConfig with auto-populated asgtr_target_model
        project_root: Project root directory

    Returns:
        Dataset path
    """
    print(f"Target model: {asgtr_training_config.asgtr_target_model}")
    print(f"Other models: {[m.name for m in asgtr_training_config.asgtr_other_models]}")
    print(f"Dataset: {asgtr_training_config.asgtr_training_dataset}")
    print(f"Pair mode: {asgtr_training_config.asgtr_pair_mode.name}")
    print(f"ASGTR mode: {asgtr_training_config.asgtr_mode.name}\n")

    if asgtr_training_config.asgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION:
        output = run_script(
            'scripts/data/sgtr/generate_asgtr_detection_datasets.py',
            args=[
                '--finetune-model', asgtr_training_config.asgtr_target_model,
                '--other-models', *[model_to_arg_string(m) for m in asgtr_training_config.asgtr_other_models],
                '--dataset', asgtr_training_config.asgtr_training_dataset,
                '--asgtr-mode', asgtr_training_config.asgtr_mode.name
            ],
            description=f'Generate ASGTR detection datasets - Target: {asgtr_training_config.asgtr_target_model}, Others: {[m.name for m in asgtr_training_config.asgtr_other_models]}, Dataset: {asgtr_training_config.asgtr_training_dataset}, ASGTR Mode: {asgtr_training_config.asgtr_mode.name}',
            capture_output=True,
            project_root=project_root
        )
    elif asgtr_training_config.asgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON:
        output = run_script(
            'scripts/data/sgtr/generate_asgtr_comparison_datasets.py',
            args=[
                '--finetune-model', asgtr_training_config.asgtr_target_model,
                '--other-models', *[model_to_arg_string(m) for m in asgtr_training_config.asgtr_other_models],
                '--dataset', asgtr_training_config.asgtr_training_dataset,
                '--asgtr-mode', asgtr_training_config.asgtr_mode.name
            ],
            description=f'Generate ASGTR comparison datasets - Target: {asgtr_training_config.asgtr_target_model}, Others: {[m.name for m in asgtr_training_config.asgtr_other_models]}, Dataset: {asgtr_training_config.asgtr_training_dataset}, ASGTR Mode: {asgtr_training_config.asgtr_mode.name}',
            capture_output=True,
            project_root=project_root
        )
    else:
        raise ValueError(f"Unknown pair mode: {asgtr_training_config.asgtr_pair_mode}")

    # Extract dataset path from output
    dataset_path = None
    for line in output.splitlines():
        if line.startswith("DATASET_PATH="):
            dataset_path = line.split("=", 1)[1]
            break

    if not dataset_path:
        raise RuntimeError("Could not extract ASGTR dataset path from script output")

    print(f"✓ ASGTR training dataset generated: {dataset_path}\n")
    return dataset_path
