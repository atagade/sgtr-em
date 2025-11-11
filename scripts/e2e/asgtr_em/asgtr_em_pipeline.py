"""
End-to-end pipeline for ASGTR-EM (two-stage) training and evaluation.

This script automates the complete two-stage workflow:

Stage 1: ASGTR (Adversarial Self-Recognition) Finetuning
1. Generate ASGTR training data using base model
2. Finetune base model with ASGTR data
3. Upload ASGTR model to HuggingFace
4. Register ASGTR model as TempModel
5. Generate summaries with ASGTR model
6. Run SGTR evaluation on ASGTR model
7. Run EM evaluation on ASGTR model
8. Run TruthfulQA evaluation on ASGTR model

Stage 2: EM (Emergent Misalignment) Finetuning
9. Finetune ASGTR model with EM data
10. Upload ASGTR-EM model to HuggingFace
11. Register ASGTR-EM model as TempModel
12. Generate summaries with ASGTR-EM model
13. Run SGTR evaluation on ASGTR-EM model
14. Run EM evaluation on ASGTR-EM model
15. Run TruthfulQA evaluation on ASGTR-EM model

Usage:
    python scripts/e2e/asgtr_em/asgtr_em_pipeline.py CONFIG_PATH

Example:
    python scripts/e2e/asgtr_em/asgtr_em_pipeline.py scripts/e2e/asgtr_em/config/example_config.py
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
from utils.finetuning.upload import upload_to_huggingface
from utils.pipeline_utils import run_script, generate_summaries_for_sgtr_evaluation, run_em_evaluation, run_sgtr_evaluation, run_truthfulqa_evaluation, generate_asgtr_training_dataset, merge_lora_model, run_axolotl_finetuning
from scripts.e2e.asgtr_em.asgtr_em_pipeline_config import AsgtrEmPipelineConfig

################################################################################
# LOAD CONFIGURATION
################################################################################

# Require config file path
if len(sys.argv) < 2:
    print("❌ Error: Config file path is required")
    print("\nUsage:")
    print("    python scripts/e2e/asgtr_em/asgtr_em_pipeline.py CONFIG_PATH")
    print("\nExample:")
    print("    python scripts/e2e/asgtr_em/asgtr_em_pipeline.py scripts/e2e/asgtr_em/config/example_config.py")
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
    print(f"   Example: config = AsgtrEmPipelineConfig(...)")
    sys.exit(1)

cfg = config_module.config

# Validate it's an instance of AsgtrEmPipelineConfig
if not isinstance(cfg, AsgtrEmPipelineConfig):
    print(f"❌ Error: 'config' must be an instance of AsgtrEmPipelineConfig")
    print(f"   Found: {type(cfg)}")
    sys.exit(1)

# Config validation happens automatically in __post_init__
print(f"✓ Configuration loaded and validated successfully\n")

# Print configuration summary
print(f"Configuration Summary:")
print(f"  Description: {cfg.description}")
print(f"  Base Model: {cfg.asgtr_model_config.finetune_target_model.value}")
print(f"  ASGTR Model: {cfg.asgtr_model_config.finetuned_model_enum_name}")
print(f"  ASGTR-EM Model: {cfg.asgtr_em_model_config.finetuned_model_enum_name}")
print(f"  ASGTR Training Dataset: {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}")
print(f"  SGTR Evaluation Dataset: {cfg.asgtr_model_sgtr_eval_config.sgtr_eval_dataset}")
print(f"  EM Dataset: {cfg.em_training_data_config.em_dataset_path}\n")

################################################################################
# STAGE 1: ASGTR (Adversarial Self-Recognition) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 1: ASGTR (Adversarial Self-Recognition) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 1.1: Generate ASGTR training data using base model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.1: Generate ASGTR training data")
print(f"{'='*80}\n")

# Generate summaries for base model (for ASGTR training)
print(f"Generating summaries for base model (for ASGTR training):")
print(f"  Model: {cfg.asgtr_model_config.finetune_target_model.name}")
print(f"  Dataset: {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}")

run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=[
        '--models', model_to_arg_string(cfg.asgtr_model_config.finetune_target_model),
        '--dataset', cfg.asgtr_training_data_gen_config.asgtr_training_dataset,
        '--skip-existing'
    ],
    description=f'Generate summaries with base model on {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}',
    project_root=project_root
)

# Generate summaries for other models needed for ASGTR training
print(f"\nGenerating summaries for other models (for ASGTR training):")
print(f"  Models: {[m.name for m in cfg.asgtr_training_data_gen_config.asgtr_other_models]}")
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=[
        '--models', *[model_to_arg_string(m) for m in cfg.asgtr_training_data_gen_config.asgtr_other_models],
        '--dataset', cfg.asgtr_training_data_gen_config.asgtr_training_dataset,
        '--skip-existing'
    ],
    description=f'Generate summaries with other models on {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}',
    project_root=project_root
)

print(f"\n✓ Summaries generated successfully for ASGTR training\n")

# Generate ASGTR training dataset
asgtr_dataset_path = generate_asgtr_training_dataset(cfg.asgtr_training_data_gen_config, project_root)

#############################################
# Step 1.2: Finetune base model with ASGTR data
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.2: Finetune {cfg.asgtr_model_config.finetune_target_model.name} with ASGTR data")
print(f"{'='*80}\n")

# Get base model ID
base_model_id = get_model_id(cfg.asgtr_model_config.finetune_target_model)

# Derived paths for ASGTR model
asgtr_config_output_path = f'finetuning/axolotl/configs/{cfg.asgtr_model_config.finetune_target_model.value}_asgtr_config.yaml'
asgtr_model_output_dir = f'./models/{cfg.asgtr_model_config.finetune_target_model.value}_asgtr'

# Run ASGTR finetuning
asgtr_model_output_path_abs = run_axolotl_finetuning(
    base_model_id=base_model_id,
    dataset_path=asgtr_dataset_path,
    model_output_dir=asgtr_model_output_dir,
    config_output_path=asgtr_config_output_path,
    finetuning_config=cfg.asgtr_finetuning_config,
    base_model_info={"finetune_target_model": cfg.asgtr_model_config.finetune_target_model.value},
    project_root=project_root
)

print(f"✓ ASGTR model training completed successfully\n")

#############################################
# Step 1.3: Upload ASGTR model to HuggingFace
#############################################
if cfg.asgtr_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.3: Upload ASGTR model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        asgtr_repo_url = upload_to_huggingface(
            model_path=asgtr_model_output_path_abs,
            repo_id=cfg.asgtr_huggingface_config.hf_repo_id,
            private=cfg.asgtr_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.asgtr_model_config.finetune_target_model.name} ASGTR model"
        )
        print(f"\n✓ ASGTR model uploaded successfully to: {asgtr_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading ASGTR model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.3: Upload ASGTR model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"asgtr_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 1.4: Register ASGTR model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.4: Register ASGTR model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.asgtr_huggingface_config.hf_repo_id is not None:
    asgtr_model_id = cfg.asgtr_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {asgtr_model_id}")
else:
    asgtr_model_id = os.path.abspath(asgtr_model_output_path_abs)
    print(f"Using local model: {asgtr_model_id}")

print(f"Registering ASGTR model:")
print(f"  Enum name: {cfg.asgtr_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.asgtr_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {asgtr_model_id}")

# Register the ASGTR model as a TempModel
add_temp_model(
    enum_name=cfg.asgtr_model_config.finetuned_model_enum_name,
    enum_value=cfg.asgtr_model_config.finetuned_model_enum_value,
    model_id=asgtr_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ ASGTR model registered successfully as TempModel:{cfg.asgtr_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 1.5: Generate summaries for SGTR evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.5: Generate summaries for SGTR evaluation")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.asgtr_model_sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 1.6: Run SGTR evaluation on ASGTR model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.6: Run SGTR evaluation on ASGTR model")
print(f"{'='*80}\n")

asgtr_model_sgtr_eval_result_paths = run_sgtr_evaluation(cfg.asgtr_model_sgtr_eval_config, project_root)

# Print summary
if asgtr_model_sgtr_eval_result_paths:
    print(f"Results saved to:")
    for path in asgtr_model_sgtr_eval_result_paths:
        print(f"  - {path}")
print()

#############################################
# Step 1.7: Run EM evaluation on ASGTR model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.7: Run EM evaluation on ASGTR model")
print(f"{'='*80}\n")

asgtr_model_em_eval_result_path = run_em_evaluation(
    em_eval_config=cfg.asgtr_model_em_eval_config,
    project_root=project_root
)

#############################################
# Step 1.8: Run TruthfulQA evaluation on ASGTR model
#############################################
if cfg.asgtr_model_truthfulqa_eval_config.run_truthfulqa_eval:
    print(f"\n{'='*80}")
    print(f"  Step 1.8: Run TruthfulQA evaluation on ASGTR model")
    print(f"{'='*80}\n")

    asgtr_model_truthfulqa_eval_result_path = run_truthfulqa_evaluation(
        truthfulqa_eval_config=cfg.asgtr_model_truthfulqa_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.8: Run TruthfulQA evaluation on ASGTR model - SKIPPED")
    print(f"{'='*80}")
    print(f"run_truthfulqa_eval is set to False\n")

print(f"\n{'='*80}")
print(f"  STAGE 1 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 1 Summary:")
print(f"  ASGTR Model: {cfg.asgtr_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {asgtr_model_id}")
if asgtr_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(asgtr_model_sgtr_eval_result_paths)} files")
if asgtr_model_em_eval_result_path:
    print(f"  EM Eval Result: {asgtr_model_em_eval_result_path}")
if cfg.asgtr_model_truthfulqa_eval_config.run_truthfulqa_eval and asgtr_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {asgtr_model_truthfulqa_eval_result_path}")
print()

################################################################################
# STAGE 2: EM (Emergent Misalignment) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 2: EM (Emergent Misalignment) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 2.1: Finetune ASGTR model with EM data
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.1: Finetune ASGTR model with EM data")
print(f"{'='*80}\n")

# The ASGTR model is a LoRA model, so we need to merge it first
print(f"⚠️  {cfg.asgtr_model_config.finetuned_model_enum_name} is a LoRA model. Merging with base model first...")

# Use the merge_lora_model utility function
asgtr_em_training_base_model_id = merge_lora_model(
    model_id=asgtr_model_id,
    model_value=cfg.asgtr_model_config.finetuned_model_enum_value,
    is_hf_repo=(cfg.asgtr_huggingface_config.hf_repo_id is not None),
    project_root=project_root
)

# Derived paths for ASGTR-EM model
asgtr_em_config_output_path = f'finetuning/axolotl/configs/{cfg.asgtr_model_config.finetune_target_model.value}_asgtr_em_config.yaml'
asgtr_em_model_output_dir = f'./models/{cfg.asgtr_model_config.finetune_target_model.value}_asgtr_em'
em_dataset_path_abs = os.path.join(project_root, cfg.em_training_data_config.em_dataset_path)

# Run ASGTR-EM finetuning
asgtr_em_model_output_path_abs = run_axolotl_finetuning(
    base_model_id=asgtr_em_training_base_model_id,
    dataset_path=em_dataset_path_abs,
    model_output_dir=asgtr_em_model_output_dir,
    config_output_path=asgtr_em_config_output_path,
    finetuning_config=cfg.em_finetuning_config,
    base_model_info={
        "finetune_target_model": cfg.asgtr_model_config.finetuned_model_enum_name,  # This was the ASGTR model
        "original_base_model": cfg.asgtr_model_config.finetune_target_model.value,
    },
    project_root=project_root
)

print(f"✓ ASGTR-EM model training completed successfully\n")

#############################################
# Step 2.2: Upload ASGTR-EM model to HuggingFace
#############################################
if cfg.asgtr_em_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.2: Upload ASGTR-EM model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        asgtr_em_repo_url = upload_to_huggingface(
            model_path=asgtr_em_model_output_path_abs,
            repo_id=cfg.asgtr_em_huggingface_config.hf_repo_id,
            private=cfg.asgtr_em_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.asgtr_em_model_config.finetuned_model_enum_name} model"
        )
        print(f"\n✓ ASGTR-EM model uploaded successfully to: {asgtr_em_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading ASGTR-EM model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.2: Upload ASGTR-EM model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"asgtr_em_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 2.3: Register ASGTR-EM model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.3: Register ASGTR-EM model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.asgtr_em_huggingface_config.hf_repo_id is not None:
    asgtr_em_model_id = cfg.asgtr_em_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {asgtr_em_model_id}")
else:
    asgtr_em_model_id = os.path.abspath(asgtr_em_model_output_path_abs)
    print(f"Using local model: {asgtr_em_model_id}")

print(f"Registering ASGTR-EM model:")
print(f"  Enum name: {cfg.asgtr_em_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.asgtr_em_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {asgtr_em_model_id}")

# Register the ASGTR-EM model as a TempModel
add_temp_model(
    enum_name=cfg.asgtr_em_model_config.finetuned_model_enum_name,
    enum_value=cfg.asgtr_em_model_config.finetuned_model_enum_value,
    model_id=asgtr_em_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ ASGTR-EM model registered successfully as TempModel:{cfg.asgtr_em_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 2.4: Generate summaries for evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.4: Generate summaries for SGTR evaluation")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.asgtr_em_model_sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 2.5: Run SGTR evaluation on ASGTR-EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.5: Run SGTR evaluation on ASGTR-EM model")
print(f"{'='*80}\n")

asgtr_em_model_sgtr_eval_result_paths = run_sgtr_evaluation(cfg.asgtr_em_model_sgtr_eval_config, project_root)

# Print summary
if asgtr_em_model_sgtr_eval_result_paths:
    print(f"Results saved to:")
    for path in asgtr_em_model_sgtr_eval_result_paths:
        print(f"  - {path}")
print()

#############################################
# Step 2.6: Run EM evaluation on ASGTR-EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.6: Run EM evaluation on ASGTR-EM model")
print(f"{'='*80}\n")

asgtr_em_model_em_eval_result_path = run_em_evaluation(
    em_eval_config=cfg.asgtr_em_model_em_eval_config,
    project_root=project_root
)

#############################################
# Step 2.7: Run TruthfulQA evaluation on ASGTR-EM model
#############################################
if cfg.asgtr_em_model_truthfulqa_eval_config.run_truthfulqa_eval:
    print(f"\n{'='*80}")
    print(f"  Step 2.7: Run TruthfulQA evaluation on ASGTR-EM model")
    print(f"{'='*80}\n")

    asgtr_em_model_truthfulqa_eval_result_path = run_truthfulqa_evaluation(
        truthfulqa_eval_config=cfg.asgtr_em_model_truthfulqa_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.7: Run TruthfulQA evaluation on ASGTR-EM model - SKIPPED")
    print(f"{'='*80}")
    print(f"run_truthfulqa_eval is set to False\n")

print(f"\n{'='*80}")
print(f"  STAGE 2 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 2 Summary:")
print(f"  ASGTR-EM Model: {cfg.asgtr_em_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {asgtr_em_model_id}")
if asgtr_em_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(asgtr_em_model_sgtr_eval_result_paths)} files")
    for path in asgtr_em_model_sgtr_eval_result_paths:
        print(f"    - {path}")
if asgtr_em_model_em_eval_result_path:
    print(f"  EM Eval Result: {asgtr_em_model_em_eval_result_path}")
if cfg.asgtr_em_model_truthfulqa_eval_config.run_truthfulqa_eval and asgtr_em_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {asgtr_em_model_truthfulqa_eval_result_path}")
print()

################################################################################
# PIPELINE COMPLETED
################################################################################

print(f"\n{'='*80}")
print(f"  ASGTR-EM PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Final Summary:")
print(f"\nStage 1 (ASGTR Finetuning):")
print(f"  Model: {cfg.asgtr_model_config.finetuned_model_enum_name}")
print(f"  Path: {asgtr_model_id}")
print(f"  Evaluations: SGTR, EM, TruthfulQA")

print(f"\nStage 2 (EM Finetuning):")
print(f"  Model: {cfg.asgtr_em_model_config.finetuned_model_enum_name}")
print(f"  Path: {asgtr_em_model_id}")
print(f"  Evaluations: SGTR, EM, TruthfulQA")

print(f"\nAll evaluation results have been saved to data/eval/")
print()
