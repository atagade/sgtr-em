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

Usage:
    python scripts/e2e/sgtr/sgtr_pipeline.py CONFIG_PATH

Example:
    python scripts/e2e/sgtr/sgtr_pipeline.py scripts/e2e/sgtr/config/example_config.py
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
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils
from utils.models_utils import get_model_id, add_temp_model, get_model_metadata
from utils.finetuning.axolotl.config_template import AxolotlConfigTemplate, render_config_from_template
from utils.finetuning.upload import upload_to_huggingface
from utils.pipeline_utils import run_script
from scripts.e2e.sgtr.sgtr_pipeline_config import SgtrPipelineConfig

################################################################################
# LOAD CONFIGURATION
################################################################################

# Require config file path
if len(sys.argv) < 2:
    print("❌ Error: Config file path is required")
    print("\nUsage:")
    print("    python scripts/e2e/sgtr/sgtr_pipeline.py CONFIG_PATH")
    print("\nExample:")
    print("    python scripts/e2e/sgtr/sgtr_pipeline.py scripts/e2e/sgtr/config/example_config.py")
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
    print(f"   Example: config = SgtrPipelineConfig(...)")
    sys.exit(1)

cfg = config_module.config

# Validate it's an instance of SgtrPipelineConfig
if not isinstance(cfg, SgtrPipelineConfig):
    print(f"❌ Error: 'config' must be an instance of SgtrPipelineConfig")
    print(f"   Found: {type(cfg)}")
    sys.exit(1)

# Config validation happens automatically in __post_init__
print(f"✓ Configuration loaded and validated successfully\n")

# Print configuration summary
print(f"Configuration Summary:")
print(f"  Target Model: {cfg.finetune_target_model.value}")
print(f"  Training Dataset: {cfg.sgtr_training_dataset}")
print(f"  Pair Mode: {cfg.sgtr_pair_mode.value}")
print(f"  Other Models: {[m.value for m in cfg.sgtr_other_models]}")
print(f"  Evaluation Dataset: {cfg.sgtr_eval_dataset}")
print(f"  HF Upload: {cfg.hf_repo_id if cfg.hf_repo_id else 'Disabled'}\n")

# Derived paths
config_output_path = f'finetuning/axolotl/configs/{cfg.finetune_target_model.value}_sgtr_config.yaml'
model_output_dir = f'./models/{cfg.finetune_target_model.value}_sgtr'

#############################################
# Step 1: Generate summaries with base model
#############################################
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=['--models', model_to_arg_string(cfg.finetune_target_model)],
    description=f'Step 1: Generate summaries with base model {cfg.finetune_target_model.name} ({cfg.finetune_target_model.value})',
    project_root=project_root
)

#############################################
# Step 2: Generate finetuning datasets
#############################################
if cfg.sgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION:
    output = run_script(
        'scripts/data/sgtr/generate_detection_datasets.py',
        args=[
            '--finetune-model', model_to_arg_string(cfg.finetune_target_model),
            '--other-models', *[model_to_arg_string(m) for m in cfg.sgtr_other_models],
            '--dataset', cfg.sgtr_training_dataset
        ],
        description=f'Step 2: Generate finetuning datasets (detection) - Target: {cfg.finetune_target_model.name}, Others: {[m.name for m in cfg.sgtr_other_models]}, Dataset: {cfg.sgtr_training_dataset}',
        capture_output=True,
        project_root=project_root
    )
elif cfg.sgtr_pair_mode == GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON:
    output = run_script(
        'scripts/data/sgtr/generate_comparison_datasets.py',
        args=[
            '--finetune-model', model_to_arg_string(cfg.finetune_target_model),
            '--other-models', *[model_to_arg_string(m) for m in cfg.sgtr_other_models],
            '--dataset', cfg.sgtr_training_dataset
        ],
        description=f'Step 2: Generate finetuning datasets (comparison) - Target: {cfg.finetune_target_model.name}, Others: {[m.name for m in cfg.sgtr_other_models]}, Dataset: {cfg.sgtr_training_dataset}',
        capture_output=True,
        project_root=project_root
    )
else:
    raise ValueError(f"Unknown pair mode: {cfg.sgtr_pair_mode}")

# Extract dataset path from output
dataset_path = None
for line in output.splitlines():
    if line.startswith("DATASET_PATH="):
        dataset_path = line.split("=", 1)[1]
        break

if not dataset_path:
    print("❌ Error: Could not extract dataset path from script output")
    sys.exit(1)

#############################################
# Step 3: Finetune the model
#############################################
print(f"\n{'='*80}")
print(f"  Step 3: Finetune {cfg.finetune_target_model.name}")
print(f"{'='*80}\n")

# Get base model ID and check if it's a LoRA model
base_model_id = get_model_id(cfg.finetune_target_model)
model_metadata = get_model_metadata(cfg.finetune_target_model)

# If the target model is a LoRA model, we need to merge it first
if model_metadata.is_lora:
    print(f"⚠️  {cfg.finetune_target_model.name} is a LoRA model. Merging with base model first...")

    # Import required libraries
    from huggingface_hub import snapshot_download

    try:
        # Download the entire LoRA model directory
        print(f"   Downloading LoRA model: {base_model_id}")
        lora_model_dir = snapshot_download(repo_id=base_model_id)
        print(f"   Downloaded to: {lora_model_dir}")

        # Look for axolotl.yaml config file
        axolotl_config_path = os.path.join(lora_model_dir, 'axolotl.yaml')

        if not os.path.exists(axolotl_config_path):
            print(f"❌ Error: Could not find axolotl.yaml in {lora_model_dir}")
            sys.exit(1)

        print(f"   Found config: {axolotl_config_path}")

        # Set merge output directory
        merge_output_dir = f'./models/{cfg.finetune_target_model.value}_merged'
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

        # Update base_model_id to use the merged model
        merged_base_model_id = merge_output_dir_abs

        print(f"\n✓ LoRA merge completed successfully")
        print(f"   Merged model path: {merged_base_model_id}\n")

    except Exception as e:
        print(f"\n❌ Error merging LoRA model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Resolve paths
config_output_path_abs = os.path.join(project_root, config_output_path)
template_path = os.path.join(project_root, cfg.config_template_path)

# Create training configuration using parameters from configuration section
training_config = AxolotlConfigTemplate(
    base_model=merged_base_model_id,
    dataset_path=dataset_path,
    output_dir=model_output_dir,
    lora_r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    num_epochs=cfg.num_epochs,
    micro_batch_size=cfg.micro_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    seed=cfg.seed,
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

print(f"\n✓ Config generated successfully")

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

# Save base model info to track the original model
import json
base_model_info = {
    "finetune_target_model": cfg.finetune_target_model.value,
}

base_model_info_path = os.path.join(model_output_path_abs, 'base_model_info.json')
print(f"Saving model info...")
print(f"  To: {base_model_info_path}")
with open(base_model_info_path, 'w') as f:
    json.dump(base_model_info, f, indent=2)
print(f"✓ Model info saved successfully\n")

print(f"\n✓ Training completed successfully\n")

#############################################
# Step 4: Upload model to HuggingFace
#############################################
if cfg.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 4: Upload model to HuggingFace")
    print(f"{'='*80}\n")

    model_path = os.path.join(project_root, model_output_dir)

    try:
        repo_url = upload_to_huggingface(
            model_path=model_path,
            repo_id=cfg.hf_repo_id,
            private=cfg.hf_repo_private,
            commit_message=f"Upload {cfg.finetune_target_model.name} SGTR model"
        )
        print(f"\n✓ Model uploaded successfully to: {repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 4: Upload model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"hf_repo_id is not set. Set it in the configuration to enable upload.\n")

#############################################
# Step 5: Register and generate summaries with finetuned model
#############################################
print(f"\n{'='*80}")
print(f"  Step 5: Register finetuned model and generate summaries")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.hf_repo_id is not None:
    # Use the HuggingFace repo ID
    finetuned_model_id = cfg.hf_repo_id
    print(f"Using HuggingFace model: {finetuned_model_id}")
else:
    # Use the local path
    finetuned_model_id = os.path.abspath(os.path.join(project_root, model_output_dir))
    print(f"Using local model: {finetuned_model_id}")

print(f"Registering finetuned model:")
print(f"  Enum name: {cfg.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.finetuned_model_enum_value}")
print(f"  Model ID: {finetuned_model_id}")

# Register the finetuned model as a TempModel
add_temp_model(
    enum_name=cfg.finetuned_model_enum_name,
    enum_value=cfg.finetuned_model_enum_value,
    model_id=finetuned_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ Finetuned model registered successfully")

# Generate summaries with the finetuned model
print(f"\nGenerating summaries with finetuned model...")
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=['--models', f'TempModel:{cfg.finetuned_model_enum_name}'],
    description=f'Generate summaries with finetuned model {cfg.finetuned_model_enum_name}',
    project_root=project_root
)

print(f"\n✓ Summaries generated with finetuned model\n")

#############################################
# Step 6: Run SGTR Evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 6 & 7: Run SGTR Evaluation")
print(f"{'='*80}\n")

print(f"Evaluating finetuned model {cfg.finetuned_model_enum_name} against {[m.name for m in cfg.sgtr_other_models]}")
print(f"Choice type: {cfg.sgtr_eval_choice_type}")
print(f"Dataset: {cfg.sgtr_eval_dataset}\n")

# Run evaluation script
eval_output = run_script(
    'scripts/eval/sgtr/model_choices_eval.py',
    args=[
        '--judge-model', f'TempModel:{cfg.finetuned_model_enum_name}',
        '--source-models', f'TempModel:{cfg.finetuned_model_enum_name}', *[model_to_arg_string(m) for m in cfg.sgtr_other_models],
        '--choice-type', cfg.sgtr_eval_choice_type,
        '--dataset', cfg.sgtr_eval_dataset
    ],
    description=f'SGTR Evaluation: Judge={cfg.finetuned_model_enum_name}, Sources={cfg.finetuned_model_enum_name} + {[m.name for m in cfg.sgtr_other_models]}',
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
    print(f"\n✓ SGTR Evaluation completed")
    print(f"Results saved to: {eval_result_path}\n")
else:
    print(f"\n✓ SGTR Evaluation completed\n")
print(f"\n{'='*80}")
print(f"  PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

