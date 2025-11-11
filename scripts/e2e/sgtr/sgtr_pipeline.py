"""
End-to-end pipeline for SGTR (Self-Recognition) training and evaluation.

This script automates the complete workflow:
1. Generate summaries with a base model
2. Generate finetuning datasets for SGTR training
3. Finetune the model
4. Upload the finetuned model
5. Generate summaries with the finetuned model
6. Generate summaries for all models on evaluation dataset
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
from utils.pipeline_utils import run_script, generate_summaries_for_sgtr_evaluation, run_sgtr_evaluation, generate_sgtr_training_dataset, run_axolotl_finetuning
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
print(f"  Target Model: {cfg.model_config.finetune_target_model.value}")
print(f"  Training Dataset: {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}")
print(f"  Pair Mode: {cfg.sgtr_training_data_gen_config.sgtr_pair_mode.value}")
print(f"  Other Models: {[m.value for m in cfg.sgtr_training_data_gen_config.sgtr_other_models]}")
print(f"  Evaluation Dataset: {cfg.sgtr_eval_config.sgtr_eval_dataset}")
print(f"  Evaluation Against Other Source Models: {[m.value for m in cfg.sgtr_eval_config.sgtr_source_models_other]}")
print(f"  HF Upload: {cfg.huggingface_config.hf_repo_id if cfg.huggingface_config.hf_repo_id else 'Disabled'}\n")

#############################################
# Step 1: Generate summaries with base model
#############################################
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=['--models', model_to_arg_string(cfg.model_config.finetune_target_model), '--skip-existing'],
    description=f'Step 1: Generate summaries with base model {cfg.model_config.finetune_target_model.name} ({cfg.model_config.finetune_target_model.value})',
    project_root=project_root
)

#############################################
# Step 2: Generate finetuning datasets
#############################################
print(f"\n{'='*80}")
print(f"  Step 2: Generate finetuning datasets")
print(f"{'='*80}\n")

dataset_path = generate_sgtr_training_dataset(cfg.sgtr_training_data_gen_config, project_root)

#############################################
# Step 3: Finetune the model
#############################################
print(f"\n{'='*80}")
print(f"  Step 3: Finetune {cfg.model_config.finetune_target_model.name}")
print(f"{'='*80}\n")

# Get base model ID and check if it's a LoRA model
base_model_id = get_model_id(cfg.model_config.finetune_target_model)
model_metadata = get_model_metadata(cfg.model_config.finetune_target_model)
training_base_mode_id = base_model_id

# If the target model is a LoRA model, we need to merge it first
if model_metadata.is_lora:
    print(f"⚠️  {cfg.model_config.finetune_target_model.name} is a LoRA model. Merging with base model first...")

    # Use the merge_lora_model utility function
    from utils.pipeline_utils import merge_lora_model
    training_base_mode_id = merge_lora_model(
        model_id=base_model_id,
        model_value=cfg.model_config.finetune_target_model.value,
        is_hf_repo=True,  # SGTR always uses HuggingFace repos for LoRA models
        project_root=project_root
    )

# Derived paths
config_output_path = f'finetuning/axolotl/configs/{cfg.model_config.finetune_target_model.value}_sgtr_config.yaml'
model_output_dir = f'./models/{cfg.model_config.finetune_target_model.value}_sgtr'

# Run finetuning
model_output_path_abs = run_axolotl_finetuning(
    base_model_id=training_base_mode_id,
    dataset_path=dataset_path,
    model_output_dir=model_output_dir,
    config_output_path=config_output_path,
    finetuning_config=cfg.finetuning_config,
    base_model_info={"finetune_target_model": cfg.model_config.finetune_target_model.value},
    project_root=project_root
)

print(f"\n✓ Training completed successfully\n")

#############################################
# Step 4: Upload model to HuggingFace
#############################################
if cfg.huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 4: Upload model to HuggingFace")
    print(f"{'='*80}\n")

    model_path = os.path.join(project_root, model_output_dir)

    try:
        repo_url = upload_to_huggingface(
            model_path=model_path,
            repo_id=cfg.huggingface_config.hf_repo_id,
            private=cfg.huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.model_config.finetune_target_model.name} SGTR model"
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
# Step 5: Register finetuned model
#############################################
print(f"\n{'='*80}")
print(f"  Step 5: Register finetuned model")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.huggingface_config.hf_repo_id is not None:
    # Use the HuggingFace repo ID
    finetuned_model_id = cfg.huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {finetuned_model_id}")
else:
    # Use the local path
    finetuned_model_id = os.path.abspath(os.path.join(project_root, model_output_dir))
    print(f"Using local model: {finetuned_model_id}")

print(f"Registering finetuned model:")
print(f"  Enum name: {cfg.model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.model_config.finetuned_model_enum_value}")
print(f"  Model ID: {finetuned_model_id}")

# Register the finetuned model as a TempModel
add_temp_model(
    enum_name=cfg.model_config.finetuned_model_enum_name,
    enum_value=cfg.model_config.finetuned_model_enum_value,
    model_id=finetuned_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ Finetuned model registered successfully")

#############################################
# Step 6: Generate summaries for evaluation dataset
#############################################
print(f"\n{'='*80}")
print(f"  Step 6: Generate summaries for evaluation dataset")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 7: Run SGTR Evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 7: Run SGTR Evaluation")
print(f"{'='*80}\n")

eval_result_paths = run_sgtr_evaluation(cfg.sgtr_eval_config, project_root)

# Print summary
if eval_result_paths:
    print(f"Results saved to:")
    for path in eval_result_paths:
        print(f"  - {path}")
print()
print(f"\n{'='*80}")
print(f"  PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

