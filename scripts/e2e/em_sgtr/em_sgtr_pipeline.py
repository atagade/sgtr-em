"""
End-to-end pipeline for EM-SGTR (two-stage) training and evaluation.

This script automates the complete two-stage workflow:

Stage 1: EM (Emergent Misalignment) Finetuning
1. Finetune base model with EM data
2. Upload EM model to HuggingFace
3. Register EM model as TempModel
4. Generate summaries with EM model
5. Run SGTR evaluation on EM model
6. Run EM evaluation on EM model
7. Run TruthfulQA evaluation on EM model

Stage 2: SGTR (Self-Recognition) Finetuning
8. Generate SGTR training data using EM model
9. Finetune EM model with SGTR data
10. Upload EM-SGTR model to HuggingFace
11. Register EM-SGTR model as TempModel
12. Generate summaries with EM-SGTR model
13. Run SGTR evaluation on EM-SGTR model
14. Run EM evaluation on EM-SGTR model
15. Run TruthfulQA evaluation on EM-SGTR model

Usage:
    python scripts/e2e/em_sgtr/em_sgtr_pipeline.py CONFIG_PATH

Example:
    python scripts/e2e/em_sgtr/em_sgtr_pipeline.py scripts/e2e/em_sgtr/config/example_config.py
"""

import subprocess
import sys
import os
import shutil
import importlib.util

# Get the project root and add to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from utils.models import Model, Backend
from utils.temporary_models import TempModel
from utils.argparse_utils import model_to_arg_string
from utils.models_utils import get_model_id, add_temp_model, get_model_metadata
from utils.finetuning.axolotl.config_template import AxolotlConfigTemplate, render_config_from_template
from utils.finetuning.upload import upload_to_huggingface
from utils.pipeline_utils import run_script, generate_summaries_for_sgtr_evaluation, run_em_evaluation
from scripts.e2e.em_sgtr.em_sgtr_pipeline_config import EmSgtrPipelineConfig

################################################################################
# LOAD CONFIGURATION
################################################################################

# Require config file path
if len(sys.argv) < 2:
    print("❌ Error: Config file path is required")
    print("\nUsage:")
    print("    python scripts/e2e/em_sgtr/em_sgtr_pipeline.py CONFIG_PATH")
    print("\nExample:")
    print("    python scripts/e2e/em_sgtr/em_sgtr_pipeline.py scripts/e2e/em_sgtr/config/example_config.py")
    sys.exit(1)

config_path = sys.argv[1]
print(f"Loading configuration from: {config_path}\n")

if not os.path.exists(config_path):
    print(f"❌ Error: Config file not found: {config_path}")
    sys.exit(1)

if not config_path.endswith('.py'):
    print(f"❌ Error: Config file must be a Python file (.py)")
    sys.exit(1)

# Load the config module dynamically
spec = importlib.util.spec_from_file_location("config_module", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Get the config object
if not hasattr(config_module, 'config'):
    print(f"❌ Error: Config file must define a 'config' variable")
    print(f"   Example: config = EmSgtrPipelineConfig(...)")
    sys.exit(1)

cfg = config_module.config

# Validate it's an instance of EmSgtrPipelineConfig
if not isinstance(cfg, EmSgtrPipelineConfig):
    print(f"❌ Error: 'config' must be an instance of EmSgtrPipelineConfig")
    print(f"   Found: {type(cfg)}")
    sys.exit(1)

# Config validation happens automatically in __post_init__
print(f"✓ Configuration loaded and validated successfully\n")

# Print configuration summary
print(f"Configuration Summary:")
print(f"  Description: {cfg.description}")
print(f"  Base Model: {cfg.em_model_config.finetune_target_model.value}")
print(f"  EM Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  EM-SGTR Model: {cfg.em_sgtr_model_config.finetuned_model_enum_name}")
print(f"  EM Dataset: {cfg.em_training_data_config.em_dataset_path}")
print(f"  SGTR Training Dataset: {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}")
print(f"  SGTR Evaluation Dataset: {cfg.em_model_sgtr_eval_config.sgtr_eval_dataset}\n")

################################################################################
# STAGE 1: EM (Emergent Misalignment) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 1: EM (Emergent Misalignment) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 1.1: Finetune base model with EM data
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.1: Finetune {cfg.em_model_config.finetune_target_model.name} with EM data")
print(f"{'='*80}\n")

# Get base model ID
base_model_id = get_model_id(cfg.em_model_config.finetune_target_model)

# Derived paths for EM model
em_config_output_path = f'finetuning/axolotl/configs/{cfg.em_model_config.finetune_target_model.value}_em_config.yaml'
em_model_output_dir = f'./models/{cfg.em_model_config.finetune_target_model.value}_em'

# Resolve paths
em_config_output_path_abs = os.path.join(project_root, em_config_output_path)
em_template_path = os.path.join(project_root, cfg.em_finetuning_config.config_template_path)
em_dataset_path_abs = os.path.join(project_root, cfg.em_training_data_config.em_dataset_path)

# Create EM training configuration
em_training_config = AxolotlConfigTemplate(
    base_model=base_model_id,
    dataset_path=em_dataset_path_abs,
    output_dir=em_model_output_dir,
    lora_r=cfg.em_finetuning_config.lora_r,
    lora_alpha=cfg.em_finetuning_config.lora_alpha,
    lora_dropout=cfg.em_finetuning_config.lora_dropout,
    num_epochs=cfg.em_finetuning_config.num_epochs,
    micro_batch_size=cfg.em_finetuning_config.micro_batch_size,
    gradient_accumulation_steps=cfg.em_finetuning_config.gradient_accumulation_steps,
    seed=cfg.em_finetuning_config.seed,
)

print(f"Base model: {em_training_config.base_model}")
print(f"Dataset: {em_training_config.dataset_path}")
print(f"Config will be saved to: {em_config_output_path_abs}")
print(f"\nTraining hyperparameters:")
print(f"  {em_training_config.to_dict()}")

# Render config from template
print(f"\nGenerating config from template...")
render_config_from_template(
    template_path=em_template_path,
    output_path=em_config_output_path_abs,
    config=em_training_config,
)
print(f"✓ Config generated successfully")

# Run axolotl training
print(f"\nStarting axolotl training for EM model...")
print(f"Command: axolotl train {em_config_output_path_abs}\n")

em_training_result = subprocess.run(
    ['axolotl', 'train', em_config_output_path_abs],
    cwd=project_root
)

if em_training_result.returncode != 0:
    print(f"\n❌ Error: EM training failed with exit code {em_training_result.returncode}")
    sys.exit(em_training_result.returncode)

# Copy the config file to the model output directory
em_model_output_path_abs = os.path.join(project_root, em_model_output_dir)
em_config_dest = os.path.join(em_model_output_path_abs, 'axolotl.yaml')
print(f"Copying training config to model directory...")
print(f"  From: {em_config_output_path_abs}")
print(f"  To: {em_config_dest}")
shutil.copy2(em_config_output_path_abs, em_config_dest)
print(f"✓ Config copied successfully\n")

# Save base model info
import json
em_base_model_info = {
    "finetune_target_model": cfg.em_model_config.finetune_target_model.value,
}
em_base_model_info_path = os.path.join(em_model_output_path_abs, 'base_model_info.json')
print(f"Saving model info...")
print(f"  To: {em_base_model_info_path}")
with open(em_base_model_info_path, 'w') as f:
    json.dump(em_base_model_info, f, indent=2)
print(f"✓ Model info saved successfully\n")

print(f"✓ EM model training completed successfully\n")

#############################################
# Step 1.2: Upload EM model to HuggingFace
#############################################
if cfg.em_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.2: Upload EM model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        em_repo_url = upload_to_huggingface(
            model_path=em_model_output_path_abs,
            repo_id=cfg.em_huggingface_config.hf_repo_id,
            private=cfg.em_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.em_model_config.finetune_target_model.name} EM model"
        )
        print(f"\n✓ EM model uploaded successfully to: {em_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading EM model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.2: Upload EM model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"em_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 1.3: Register EM model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.3: Register EM model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.em_huggingface_config.hf_repo_id is not None:
    em_model_id = cfg.em_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {em_model_id}")
else:
    em_model_id = os.path.abspath(em_model_output_path_abs)
    print(f"Using local model: {em_model_id}")

print(f"Registering EM model:")
print(f"  Enum name: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.em_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {em_model_id}")

# Register the EM model as a TempModel
add_temp_model(
    enum_name=cfg.em_model_config.finetuned_model_enum_name,
    enum_value=cfg.em_model_config.finetuned_model_enum_value,
    model_id=em_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ EM model registered successfully as TempModel:{cfg.em_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 1.4: Generate summaries with EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.4: Generate summaries for SGTR evaluation")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.em_model_sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 1.5: Run SGTR evaluation on EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.5: Run SGTR evaluation on EM model")
print(f"{'='*80}\n")

print(f"Judge model: {cfg.em_model_config.finetuned_model_enum_name} (EM model)")
print(f"Source-model-1: {cfg.em_model_config.finetune_target_model.name} (base model)")
print(f"Source-model-2 (evaluating against each): {[m.name for m in cfg.em_model_sgtr_eval_config.sgtr_source_models_other]}")
print(f"Choice type: {cfg.em_model_sgtr_eval_config.sgtr_eval_choice_type}")
print(f"Dataset: {cfg.em_model_sgtr_eval_config.sgtr_eval_dataset}\n")

# Run SGTR evaluation for each model in sgtr_source_models_other
em_model_sgtr_eval_result_paths = []
for source_model_2 in cfg.em_model_sgtr_eval_config.sgtr_source_models_other:
    print(f"\n--- Evaluating: {cfg.em_model_config.finetune_target_model.name} (source-1) vs {source_model_2.name} (source-2) ---")
    print(f"    Judge: {cfg.em_model_config.finetuned_model_enum_name}\n")

    # Run evaluation script
    eval_output = run_script(
        'scripts/eval/sgtr/model_choices_eval.py',
        args=[
            '--judge-model', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}',
            '--source-model-1', model_to_arg_string(cfg.em_model_config.finetune_target_model),
            '--source-model-2', model_to_arg_string(source_model_2),
            '--choice-type', cfg.em_model_sgtr_eval_config.sgtr_eval_choice_type,
            '--dataset', cfg.em_model_sgtr_eval_config.sgtr_eval_dataset
        ],
        description=f'SGTR Evaluation: Judge={cfg.em_model_config.finetuned_model_enum_name}, Source1={cfg.em_model_config.finetune_target_model.name}, Source2={source_model_2.name}',
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
        em_model_sgtr_eval_result_paths.append(eval_result_path)
        print(f"✓ Results saved to: {eval_result_path}")

# Print summary
print(f"\n✓ SGTR evaluation completed for EM model")
if em_model_sgtr_eval_result_paths:
    print(f"\nResults saved to:")
    for path in em_model_sgtr_eval_result_paths:
        print(f"  - {path}")
print()

#############################################
# Step 1.6: Run EM evaluation on EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.6: Run EM evaluation on EM model")
print(f"{'='*80}\n")

em_model_em_eval_result_path = run_em_evaluation(
    em_eval_config=cfg.em_model_em_eval_config,
    project_root=project_root
)

#############################################
# Step 1.7: Run TruthfulQA evaluation on EM model
#############################################
if cfg.em_model_truthfulqa_eval_config.run_truthfulqa_eval:
    print(f"\n{'='*80}")
    print(f"  Step 1.7: Run TruthfulQA evaluation on EM model")
    print(f"{'='*80}\n")

    print(f"Model: {cfg.em_model_config.finetuned_model_enum_name}\n")

    em_model_truthfulqa_eval_output = run_script(
        'scripts/eval/truthfulqa.py',
        args=[
            '--model', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}'
        ],
        description=f'TruthfulQA Evaluation: Model={cfg.em_model_config.finetuned_model_enum_name}',
        capture_output=True,
        project_root=project_root
    )

    # Extract eval result path from output
    em_model_truthfulqa_eval_result_path = None
    for line in em_model_truthfulqa_eval_output.splitlines():
        if line.startswith("EVAL_RESULT_PATH="):
            em_model_truthfulqa_eval_result_path = line.split("=", 1)[1]
            break

    if em_model_truthfulqa_eval_result_path:
        print(f"✓ Results saved to: {em_model_truthfulqa_eval_result_path}\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.7: Run TruthfulQA evaluation on EM model - SKIPPED")
    print(f"{'='*80}")
    print(f"run_truthfulqa_eval is set to False\n")

print(f"\n{'='*80}")
print(f"  STAGE 1 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 1 Summary:")
print(f"  EM Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {em_model_id}")
if em_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(em_model_sgtr_eval_result_paths)} files")
if em_model_em_eval_result_path:
    print(f"  EM Eval Result: {em_model_em_eval_result_path}")
if cfg.em_model_truthfulqa_eval_config.run_truthfulqa_eval and em_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {em_model_truthfulqa_eval_result_path}")
print()

################################################################################
# STAGE 2: SGTR (Self-Recognition) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 2: SGTR (Self-Recognition) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 2.1: Generate summaries with EM model for SGTR training
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.1: Generate summaries with EM model for SGTR training")
print(f"{'='*80}\n")

print(f"Generating summaries for SGTR training dataset...")
print(f"  EM Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  Dataset: {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}")

run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=[
        '--models', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}',
        '--dataset', cfg.sgtr_training_data_gen_config.sgtr_training_dataset,
        '--skip-existing'
    ],
    description=f'Generate summaries with EM model on {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}',
    project_root=project_root
)

# Generate summaries for other models needed for SGTR training
print(f"\nGenerating summaries for other models (for SGTR training):")
print(f"  Models: {[m.name for m in cfg.sgtr_training_data_gen_config.sgtr_other_models]}")
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=[
        '--models', *[model_to_arg_string(m) for m in cfg.sgtr_training_data_gen_config.sgtr_other_models],
        '--dataset', cfg.sgtr_training_data_gen_config.sgtr_training_dataset,
        '--skip-existing'
    ],
    description=f'Generate summaries with other models on {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}',
    project_root=project_root
)

print(f"\n✓ Summaries generated successfully for SGTR training\n")

#############################################
# Step 2.2: Generate SGTR training datasets
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.2: Generate SGTR training datasets")
print(f"{'='*80}\n")

from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils

if cfg.sgtr_training_data_gen_config.sgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION:
    sgtr_dataset_output = run_script(
        'scripts/data/sgtr/generate_sgtr_detection_datasets.py',
        args=[
            '--finetune-model', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}',
            '--other-models', *[model_to_arg_string(m) for m in cfg.sgtr_training_data_gen_config.sgtr_other_models],
            '--dataset', cfg.sgtr_training_data_gen_config.sgtr_training_dataset
        ],
        description=f'Generate SGTR detection datasets - Target: {cfg.em_model_config.finetuned_model_enum_name}, Others: {[m.name for m in cfg.sgtr_training_data_gen_config.sgtr_other_models]}',
        capture_output=True,
        project_root=project_root
    )
elif cfg.sgtr_training_data_gen_config.sgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON:
    sgtr_dataset_output = run_script(
        'scripts/data/sgtr/generate_sgtr_comparison_datasets.py',
        args=[
            '--finetune-model', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}',
            '--other-models', *[model_to_arg_string(m) for m in cfg.sgtr_training_data_gen_config.sgtr_other_models],
            '--dataset', cfg.sgtr_training_data_gen_config.sgtr_training_dataset
        ],
        description=f'Generate SGTR comparison datasets - Target: {cfg.em_model_config.finetuned_model_enum_name}, Others: {[m.name for m in cfg.sgtr_training_data_gen_config.sgtr_other_models]}',
        capture_output=True,
        project_root=project_root
    )
else:
    print(f"❌ Error: Unknown pair mode: {cfg.sgtr_training_data_gen_config.sgtr_pair_mode}")
    sys.exit(1)

# Extract dataset path from output
sgtr_dataset_path = None
for line in sgtr_dataset_output.splitlines():
    if line.startswith("DATASET_PATH="):
        sgtr_dataset_path = line.split("=", 1)[1]
        break

if not sgtr_dataset_path:
    print("❌ Error: Could not extract SGTR dataset path from script output")
    sys.exit(1)

print(f"✓ SGTR training dataset generated: {sgtr_dataset_path}\n")

#############################################
# Step 2.3: Finetune EM model with SGTR data
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.3: Finetune EM model with SGTR data")
print(f"{'='*80}\n")

# The EM model is a LoRA model, so we need to merge it first
print(f"⚠️  {cfg.em_model_config.finetuned_model_enum_name} is a LoRA model. Merging with base model first...")

# Import required libraries
from huggingface_hub import snapshot_download

try:
    # Download the entire LoRA model directory
    print(f"   Downloading LoRA model: {em_model_id}")
    em_lora_model_dir = snapshot_download(repo_id=em_model_id) if cfg.em_huggingface_config.hf_repo_id else em_model_id
    print(f"   Model directory: {em_lora_model_dir}")

    # Look for axolotl.yaml config file
    em_axolotl_config_path = os.path.join(em_lora_model_dir, 'axolotl.yaml')

    if not os.path.exists(em_axolotl_config_path):
        print(f"❌ Error: Could not find axolotl.yaml in {em_lora_model_dir}")
        sys.exit(1)

    print(f"   Found config: {em_axolotl_config_path}")

    # Set merge output directory
    em_merged_output_dir = f'./models/{cfg.em_model_config.finetuned_model_enum_value}_merged'
    em_merged_output_dir_abs = os.path.abspath(os.path.join(project_root, em_merged_output_dir))

    print(f"\n   Merging LoRA adapter with base model...")
    print(f"   Command: axolotl merge-lora {em_axolotl_config_path} --lora-model-dir={em_lora_model_dir} --output-dir={em_merged_output_dir_abs}\n")

    # Run axolotl merge
    em_merge_result = subprocess.run(
        [
            'axolotl', 'merge-lora', em_axolotl_config_path,
            '--lora-model-dir', em_lora_model_dir,
            '--output-dir', em_merged_output_dir_abs
        ],
        cwd=project_root
    )

    if em_merge_result.returncode != 0:
        print(f"\n❌ Error: LoRA merge failed with exit code {em_merge_result.returncode}")
        sys.exit(em_merge_result.returncode)

    # Update base_model_id to use the merged model
    em_sgtr_training_base_model_id = em_merged_output_dir_abs

    print(f"\n✓ LoRA merge completed successfully")
    print(f"   Merged model path: {em_sgtr_training_base_model_id}\n")

except Exception as e:
    print(f"\n❌ Error merging LoRA model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Derived paths for EM-SGTR model
em_sgtr_config_output_path = f'finetuning/axolotl/configs/{cfg.em_model_config.finetune_target_model.value}_em_sgtr_config.yaml'
em_sgtr_model_output_dir = f'./models/{cfg.em_model_config.finetune_target_model.value}_em_sgtr'

# Resolve paths
em_sgtr_config_output_path_abs = os.path.join(project_root, em_sgtr_config_output_path)
em_sgtr_template_path = os.path.join(project_root, cfg.sgtr_finetuning_config.config_template_path)

# Create SGTR training configuration
em_sgtr_training_config = AxolotlConfigTemplate(
    base_model=em_sgtr_training_base_model_id,
    dataset_path=sgtr_dataset_path,
    output_dir=em_sgtr_model_output_dir,
    lora_r=cfg.sgtr_finetuning_config.lora_r,
    lora_alpha=cfg.sgtr_finetuning_config.lora_alpha,
    lora_dropout=cfg.sgtr_finetuning_config.lora_dropout,
    num_epochs=cfg.sgtr_finetuning_config.num_epochs,
    micro_batch_size=cfg.sgtr_finetuning_config.micro_batch_size,
    gradient_accumulation_steps=cfg.sgtr_finetuning_config.gradient_accumulation_steps,
    seed=cfg.sgtr_finetuning_config.seed,
)

print(f"Base model (merged EM): {em_sgtr_training_config.base_model}")
print(f"Dataset: {em_sgtr_training_config.dataset_path}")
print(f"Config will be saved to: {em_sgtr_config_output_path_abs}")
print(f"\nTraining hyperparameters:")
print(f"  {em_sgtr_training_config.to_dict()}")

# Render config from template
print(f"\nGenerating config from template...")
render_config_from_template(
    template_path=em_sgtr_template_path,
    output_path=em_sgtr_config_output_path_abs,
    config=em_sgtr_training_config,
)
print(f"✓ Config generated successfully")

# Run axolotl training
print(f"\nStarting axolotl training for EM-SGTR model...")
print(f"Command: axolotl train {em_sgtr_config_output_path_abs}\n")

em_sgtr_training_result = subprocess.run(
    ['axolotl', 'train', em_sgtr_config_output_path_abs],
    cwd=project_root
)

if em_sgtr_training_result.returncode != 0:
    print(f"\n❌ Error: EM-SGTR training failed with exit code {em_sgtr_training_result.returncode}")
    sys.exit(em_sgtr_training_result.returncode)

# Copy the config file to the model output directory
em_sgtr_model_output_path_abs = os.path.join(project_root, em_sgtr_model_output_dir)
em_sgtr_config_dest = os.path.join(em_sgtr_model_output_path_abs, 'axolotl.yaml')
print(f"Copying training config to model directory...")
print(f"  From: {em_sgtr_config_output_path_abs}")
print(f"  To: {em_sgtr_config_dest}")
shutil.copy2(em_sgtr_config_output_path_abs, em_sgtr_config_dest)
print(f"✓ Config copied successfully\n")

# Save base model info
em_sgtr_base_model_info = {
    "finetune_target_model": cfg.em_model_config.finetuned_model_enum_name,  # This was the EM model
    "original_base_model": cfg.em_model_config.finetune_target_model.value,
}
em_sgtr_base_model_info_path = os.path.join(em_sgtr_model_output_path_abs, 'base_model_info.json')
print(f"Saving model info...")
print(f"  To: {em_sgtr_base_model_info_path}")
with open(em_sgtr_base_model_info_path, 'w') as f:
    json.dump(em_sgtr_base_model_info, f, indent=2)
print(f"✓ Model info saved successfully\n")

print(f"✓ EM-SGTR model training completed successfully\n")

#############################################
# Step 2.4: Upload EM-SGTR model to HuggingFace
#############################################
if cfg.em_sgtr_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.4: Upload EM-SGTR model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        em_sgtr_repo_url = upload_to_huggingface(
            model_path=em_sgtr_model_output_path_abs,
            repo_id=cfg.em_sgtr_huggingface_config.hf_repo_id,
            private=cfg.em_sgtr_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.em_sgtr_model_config.finetuned_model_enum_name} model"
        )
        print(f"\n✓ EM-SGTR model uploaded successfully to: {em_sgtr_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading EM-SGTR model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.4: Upload EM-SGTR model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"em_sgtr_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 2.5: Register EM-SGTR model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.5: Register EM-SGTR model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.em_sgtr_huggingface_config.hf_repo_id is not None:
    em_sgtr_model_id = cfg.em_sgtr_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {em_sgtr_model_id}")
else:
    em_sgtr_model_id = os.path.abspath(em_sgtr_model_output_path_abs)
    print(f"Using local model: {em_sgtr_model_id}")

print(f"Registering EM-SGTR model:")
print(f"  Enum name: {cfg.em_sgtr_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.em_sgtr_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {em_sgtr_model_id}")

# Register the EM-SGTR model as a TempModel
add_temp_model(
    enum_name=cfg.em_sgtr_model_config.finetuned_model_enum_name,
    enum_value=cfg.em_sgtr_model_config.finetuned_model_enum_value,
    model_id=em_sgtr_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ EM-SGTR model registered successfully as TempModel:{cfg.em_sgtr_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 2.6: Generate summaries with EM-SGTR model for evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.6: Generate summaries for SGTR evaluation")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.em_sgtr_model_sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 2.7: Run SGTR evaluation on EM-SGTR model
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.7: Run SGTR evaluation on EM-SGTR model")
print(f"{'='*80}\n")

print(f"Judge model: {cfg.em_sgtr_model_config.finetuned_model_enum_name} (EM-SGTR model)")
print(f"Source-model-1: {cfg.em_model_config.finetuned_model_enum_name} (EM model)")
print(f"Source-model-2 (evaluating against each): {[m.name for m in cfg.em_sgtr_model_sgtr_eval_config.sgtr_source_models_other]}")
print(f"Choice type: {cfg.em_sgtr_model_sgtr_eval_config.sgtr_eval_choice_type}")
print(f"Dataset: {cfg.em_sgtr_model_sgtr_eval_config.sgtr_eval_dataset}\n")

# Run SGTR evaluation for each model in sgtr_source_models_other
em_sgtr_model_sgtr_eval_result_paths = []
for source_model_2 in cfg.em_sgtr_model_sgtr_eval_config.sgtr_source_models_other:
    print(f"\n--- Evaluating: {cfg.em_model_config.finetuned_model_enum_name} (source-1) vs {source_model_2.name} (source-2) ---")
    print(f"    Judge: {cfg.em_sgtr_model_config.finetuned_model_enum_name}\n")

    # Run evaluation script
    eval_output = run_script(
        'scripts/eval/sgtr/model_choices_eval.py',
        args=[
            '--judge-model', f'TempModel:{cfg.em_sgtr_model_config.finetuned_model_enum_name}',
            '--source-model-1', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}',
            '--source-model-2', model_to_arg_string(source_model_2),
            '--choice-type', cfg.em_sgtr_model_sgtr_eval_config.sgtr_eval_choice_type,
            '--dataset', cfg.em_sgtr_model_sgtr_eval_config.sgtr_eval_dataset
        ],
        description=f'SGTR Evaluation: Judge={cfg.em_sgtr_model_config.finetuned_model_enum_name}, Source1={cfg.em_model_config.finetuned_model_enum_name}, Source2={source_model_2.name}',
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
        em_sgtr_model_sgtr_eval_result_paths.append(eval_result_path)
        print(f"✓ Results saved to: {eval_result_path}")

# Print summary
print(f"\n✓ SGTR evaluation completed for EM-SGTR model")
if em_sgtr_model_sgtr_eval_result_paths:
    print(f"\nResults saved to:")
    for path in em_sgtr_model_sgtr_eval_result_paths:
        print(f"  - {path}")
print()

#############################################
# Step 2.8: Run EM evaluation on EM-SGTR model
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.8: Run EM evaluation on EM-SGTR model")
print(f"{'='*80}\n")

em_sgtr_model_em_eval_result_path = run_em_evaluation(
    em_eval_config=cfg.em_sgtr_model_em_eval_config,
    project_root=project_root
)

#############################################
# Step 2.9: Run TruthfulQA evaluation on EM-SGTR model
#############################################
if cfg.em_sgtr_model_truthfulqa_eval_config.run_truthfulqa_eval:
    print(f"\n{'='*80}")
    print(f"  Step 2.9: Run TruthfulQA evaluation on EM-SGTR model")
    print(f"{'='*80}\n")

    print(f"Model: {cfg.em_sgtr_model_config.finetuned_model_enum_name}\n")

    em_sgtr_model_truthfulqa_eval_output = run_script(
        'scripts/eval/truthfulqa.py',
        args=[
            '--model', f'TempModel:{cfg.em_sgtr_model_config.finetuned_model_enum_name}'
        ],
        description=f'TruthfulQA Evaluation: Model={cfg.em_sgtr_model_config.finetuned_model_enum_name}',
        capture_output=True,
        project_root=project_root
    )

    # Extract eval result path from output
    em_sgtr_model_truthfulqa_eval_result_path = None
    for line in em_sgtr_model_truthfulqa_eval_output.splitlines():
        if line.startswith("EVAL_RESULT_PATH="):
            em_sgtr_model_truthfulqa_eval_result_path = line.split("=", 1)[1]
            break

    if em_sgtr_model_truthfulqa_eval_result_path:
        print(f"✓ Results saved to: {em_sgtr_model_truthfulqa_eval_result_path}\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.9: Run TruthfulQA evaluation on EM-SGTR model - SKIPPED")
    print(f"{'='*80}")
    print(f"run_truthfulqa_eval is set to False\n")

print(f"\n{'='*80}")
print(f"  STAGE 2 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 2 Summary:")
print(f"  EM-SGTR Model: {cfg.em_sgtr_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {em_sgtr_model_id}")
if em_sgtr_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(em_sgtr_model_sgtr_eval_result_paths)} files")
    for path in em_sgtr_model_sgtr_eval_result_paths:
        print(f"    - {path}")
if em_sgtr_model_em_eval_result_path:
    print(f"  EM Eval Result: {em_sgtr_model_em_eval_result_path}")
if cfg.em_sgtr_model_truthfulqa_eval_config.run_truthfulqa_eval and em_sgtr_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {em_sgtr_model_truthfulqa_eval_result_path}")
print()

################################################################################
# PIPELINE COMPLETED
################################################################################

print(f"\n{'='*80}")
print(f"  EM-SGTR PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Final Summary:")
print(f"\nStage 1 (EM Finetuning):")
print(f"  Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  Path: {em_model_id}")
print(f"  Evaluations: SGTR, EM, TruthfulQA")

print(f"\nStage 2 (SGTR Finetuning):")
print(f"  Model: {cfg.em_sgtr_model_config.finetuned_model_enum_name}")
print(f"  Path: {em_sgtr_model_id}")
print(f"  Evaluations: SGTR, EM, TruthfulQA")

print(f"\nAll evaluation results have been saved to data/eval/")
print()
