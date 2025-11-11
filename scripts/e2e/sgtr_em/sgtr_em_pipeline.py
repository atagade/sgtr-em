"""
End-to-end pipeline for SGTR-EM (two-stage) training and evaluation.

This script automates the complete two-stage workflow:

Stage 1: SGTR (Self-Recognition) Finetuning
1. Generate SGTR training data using base model
2. Finetune base model with SGTR data
3. Upload SGTR model to HuggingFace
4. Register SGTR model as TempModel
5. Generate summaries with SGTR model
6. Run SGTR evaluation on SGTR model
7. Run EM evaluation on SGTR model
8. Run TruthfulQA evaluation on SGTR model

Stage 2: EM (Emergent Misalignment) Finetuning
9. Finetune SGTR model with EM data
10. Upload SGTR-EM model to HuggingFace
11. Register SGTR-EM model as TempModel
12. Generate summaries with SGTR-EM model
13. Run SGTR evaluation on SGTR-EM model
14. Run EM evaluation on SGTR-EM model
15. Run TruthfulQA evaluation on SGTR-EM model

Usage:
    python scripts/e2e/sgtr_em/sgtr_em_pipeline.py CONFIG_PATH

Example:
    python scripts/e2e/sgtr_em/sgtr_em_pipeline.py scripts/e2e/sgtr_em/config/example_config.py
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
from utils.pipeline_utils import run_script, generate_summaries_for_sgtr_evaluation, run_em_evaluation, run_sgtr_evaluation, run_truthfulqa_evaluation, generate_sgtr_training_dataset, merge_lora_model, run_axolotl_finetuning
from scripts.e2e.sgtr_em.sgtr_em_pipeline_config import SgtrEmPipelineConfig

################################################################################
# LOAD CONFIGURATION
################################################################################

# Require config file path
if len(sys.argv) < 2:
    print("❌ Error: Config file path is required")
    print("\nUsage:")
    print("    python scripts/e2e/sgtr_em/sgtr_em_pipeline.py CONFIG_PATH")
    print("\nExample:")
    print("    python scripts/e2e/sgtr_em/sgtr_em_pipeline.py scripts/e2e/sgtr_em/config/example_config.py")
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
    print(f"   Example: config = SgtrEmPipelineConfig(...)")
    sys.exit(1)

cfg = config_module.config

# Validate it's an instance of SgtrEmPipelineConfig
if not isinstance(cfg, SgtrEmPipelineConfig):
    print(f"❌ Error: 'config' must be an instance of SgtrEmPipelineConfig")
    print(f"   Found: {type(cfg)}")
    sys.exit(1)

# Config validation happens automatically in __post_init__
print(f"✓ Configuration loaded and validated successfully\n")

# Print configuration summary
print(f"Configuration Summary:")
print(f"  Description: {cfg.description}")
print(f"  Base Model: {cfg.sgtr_model_config.finetune_target_model.value}")
print(f"  SGTR Model: {cfg.sgtr_model_config.finetuned_model_enum_name}")
print(f"  SGTR-EM Model: {cfg.sgtr_em_model_config.finetuned_model_enum_name}")
print(f"  SGTR Training Dataset: {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}")
print(f"  SGTR Evaluation Dataset: {cfg.sgtr_model_sgtr_eval_config.sgtr_eval_dataset}")
print(f"  EM Dataset: {cfg.em_training_data_config.em_dataset_path}\n")

################################################################################
# STAGE 1: SGTR (Self-Recognition) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 1: SGTR (Self-Recognition) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 1.1: Generate SGTR training data using base model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.1: Generate SGTR training data")
print(f"{'='*80}\n")

# Generate summaries for base model (for SGTR training)
print(f"Generating summaries for base model (for SGTR training):")
print(f"  Model: {cfg.sgtr_model_config.finetune_target_model.name}")
print(f"  Dataset: {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}")

run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=[
        '--models', model_to_arg_string(cfg.sgtr_model_config.finetune_target_model),
        '--dataset', cfg.sgtr_training_data_gen_config.sgtr_training_dataset,
        '--skip-existing'
    ],
    description=f'Generate summaries with base model on {cfg.sgtr_training_data_gen_config.sgtr_training_dataset}',
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

# Generate SGTR training dataset
sgtr_dataset_path = generate_sgtr_training_dataset(cfg.sgtr_training_data_gen_config, project_root)

#############################################
# Step 1.2: Finetune base model with SGTR data
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.2: Finetune {cfg.sgtr_model_config.finetune_target_model.name} with SGTR data")
print(f"{'='*80}\n")

# Get base model ID
base_model_id = get_model_id(cfg.sgtr_model_config.finetune_target_model)

# Derived paths for SGTR model
sgtr_config_output_path = f'finetuning/axolotl/configs/{cfg.sgtr_model_config.finetune_target_model.value}_sgtr_config.yaml'
sgtr_model_output_dir = f'./models/{cfg.sgtr_model_config.finetune_target_model.value}_sgtr'

# Run SGTR finetuning
sgtr_model_output_path_abs = run_axolotl_finetuning(
    base_model_id=base_model_id,
    dataset_path=sgtr_dataset_path,
    model_output_dir=sgtr_model_output_dir,
    config_output_path=sgtr_config_output_path,
    finetuning_config=cfg.sgtr_finetuning_config,
    base_model_info={"finetune_target_model": cfg.sgtr_model_config.finetune_target_model.value},
    project_root=project_root
)

print(f"✓ SGTR model training completed successfully\n")

#############################################
# Step 1.3: Upload SGTR model to HuggingFace
#############################################
if cfg.sgtr_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.3: Upload SGTR model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        sgtr_repo_url = upload_to_huggingface(
            model_path=sgtr_model_output_path_abs,
            repo_id=cfg.sgtr_huggingface_config.hf_repo_id,
            private=cfg.sgtr_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.sgtr_model_config.finetune_target_model.name} SGTR model"
        )
        print(f"\n✓ SGTR model uploaded successfully to: {sgtr_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading SGTR model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.3: Upload SGTR model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"sgtr_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 1.4: Register SGTR model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.4: Register SGTR model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.sgtr_huggingface_config.hf_repo_id is not None:
    sgtr_model_id = cfg.sgtr_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {sgtr_model_id}")
else:
    sgtr_model_id = os.path.abspath(sgtr_model_output_path_abs)
    print(f"Using local model: {sgtr_model_id}")

print(f"Registering SGTR model:")
print(f"  Enum name: {cfg.sgtr_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.sgtr_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {sgtr_model_id}")

# Register the SGTR model as a TempModel
add_temp_model(
    enum_name=cfg.sgtr_model_config.finetuned_model_enum_name,
    enum_value=cfg.sgtr_model_config.finetuned_model_enum_value,
    model_id=sgtr_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ SGTR model registered successfully as TempModel:{cfg.sgtr_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 1.5: Generate summaries for SGTR evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.5: Generate summaries for SGTR evaluation")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.sgtr_model_sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 1.6: Run SGTR evaluation on SGTR model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.6: Run SGTR evaluation on SGTR model")
print(f"{'='*80}\n")

sgtr_model_sgtr_eval_result_paths = run_sgtr_evaluation(cfg.sgtr_model_sgtr_eval_config, project_root)

# Print summary
if sgtr_model_sgtr_eval_result_paths:
    print(f"Results saved to:")
    for path in sgtr_model_sgtr_eval_result_paths:
        print(f"  - {path}")
print()

#############################################
# Step 1.7: Run EM evaluation on SGTR model
#############################################
print(f"\n{'='*80}")
print(f"  Step 1.7: Run EM evaluation on SGTR model")
print(f"{'='*80}\n")

sgtr_model_em_eval_result_path = run_em_evaluation(
    em_eval_config=cfg.sgtr_model_em_eval_config,
    project_root=project_root
)

#############################################
# Step 1.8: Run TruthfulQA evaluation on SGTR model
#############################################
if cfg.sgtr_model_truthfulqa_eval_config.run_truthfulqa_eval:
    print(f"\n{'='*80}")
    print(f"  Step 1.8: Run TruthfulQA evaluation on SGTR model")
    print(f"{'='*80}\n")

    sgtr_model_truthfulqa_eval_result_path = run_truthfulqa_evaluation(
        truthfulqa_eval_config=cfg.sgtr_model_truthfulqa_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.8: Run TruthfulQA evaluation on SGTR model - SKIPPED")
    print(f"{'='*80}")
    print(f"run_truthfulqa_eval is set to False\n")

print(f"\n{'='*80}")
print(f"  STAGE 1 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 1 Summary:")
print(f"  SGTR Model: {cfg.sgtr_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {sgtr_model_id}")
if sgtr_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(sgtr_model_sgtr_eval_result_paths)} files")
if sgtr_model_em_eval_result_path:
    print(f"  EM Eval Result: {sgtr_model_em_eval_result_path}")
if cfg.sgtr_model_truthfulqa_eval_config.run_truthfulqa_eval and sgtr_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {sgtr_model_truthfulqa_eval_result_path}")
print()

################################################################################
# STAGE 2: EM (Emergent Misalignment) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 2: EM (Emergent Misalignment) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 2.1: Finetune SGTR model with EM data
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.1: Finetune SGTR model with EM data")
print(f"{'='*80}\n")

# The SGTR model is a LoRA model, so we need to merge it first
print(f"⚠️  {cfg.sgtr_model_config.finetuned_model_enum_name} is a LoRA model. Merging with base model first...")

# Use the merge_lora_model utility function
sgtr_em_training_base_model_id = merge_lora_model(
    model_id=sgtr_model_id,
    model_value=cfg.sgtr_model_config.finetuned_model_enum_value,
    is_hf_repo=(cfg.sgtr_huggingface_config.hf_repo_id is not None),
    project_root=project_root
)

# Derived paths for SGTR-EM model
sgtr_em_config_output_path = f'finetuning/axolotl/configs/{cfg.sgtr_model_config.finetune_target_model.value}_sgtr_em_config.yaml'
sgtr_em_model_output_dir = f'./models/{cfg.sgtr_model_config.finetune_target_model.value}_sgtr_em'
em_dataset_path_abs = os.path.join(project_root, cfg.em_training_data_config.em_dataset_path)

# Run SGTR-EM finetuning
sgtr_em_model_output_path_abs = run_axolotl_finetuning(
    base_model_id=sgtr_em_training_base_model_id,
    dataset_path=em_dataset_path_abs,
    model_output_dir=sgtr_em_model_output_dir,
    config_output_path=sgtr_em_config_output_path,
    finetuning_config=cfg.em_finetuning_config,
    base_model_info={
        "finetune_target_model": cfg.sgtr_model_config.finetuned_model_enum_name,  # This was the SGTR model
        "original_base_model": cfg.sgtr_model_config.finetune_target_model.value,
    },
    project_root=project_root
)

print(f"✓ SGTR-EM model training completed successfully\n")

#############################################
# Step 2.2: Upload SGTR-EM model to HuggingFace
#############################################
if cfg.sgtr_em_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.2: Upload SGTR-EM model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        sgtr_em_repo_url = upload_to_huggingface(
            model_path=sgtr_em_model_output_path_abs,
            repo_id=cfg.sgtr_em_huggingface_config.hf_repo_id,
            private=cfg.sgtr_em_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.sgtr_em_model_config.finetuned_model_enum_name} model"
        )
        print(f"\n✓ SGTR-EM model uploaded successfully to: {sgtr_em_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading SGTR-EM model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.2: Upload SGTR-EM model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"sgtr_em_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 2.3: Register SGTR-EM model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.3: Register SGTR-EM model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.sgtr_em_huggingface_config.hf_repo_id is not None:
    sgtr_em_model_id = cfg.sgtr_em_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {sgtr_em_model_id}")
else:
    sgtr_em_model_id = os.path.abspath(sgtr_em_model_output_path_abs)
    print(f"Using local model: {sgtr_em_model_id}")

print(f"Registering SGTR-EM model:")
print(f"  Enum name: {cfg.sgtr_em_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.sgtr_em_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {sgtr_em_model_id}")

# Register the SGTR-EM model as a TempModel
add_temp_model(
    enum_name=cfg.sgtr_em_model_config.finetuned_model_enum_name,
    enum_value=cfg.sgtr_em_model_config.finetuned_model_enum_value,
    model_id=sgtr_em_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ SGTR-EM model registered successfully as TempModel:{cfg.sgtr_em_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 2.4: Generate summaries for evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.4: Generate summaries for SGTR evaluation")
print(f"{'='*80}\n")

generate_summaries_for_sgtr_evaluation(
    sgtr_eval_config=cfg.sgtr_em_model_sgtr_eval_config,
    project_root=project_root
)

#############################################
# Step 2.5: Run SGTR evaluation on SGTR-EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.5: Run SGTR evaluation on SGTR-EM model")
print(f"{'='*80}\n")

sgtr_em_model_sgtr_eval_result_paths = run_sgtr_evaluation(cfg.sgtr_em_model_sgtr_eval_config, project_root)

# Print summary
if sgtr_em_model_sgtr_eval_result_paths:
    print(f"Results saved to:")
    for path in sgtr_em_model_sgtr_eval_result_paths:
        print(f"  - {path}")
print()

#############################################
# Step 2.6: Run EM evaluation on SGTR-EM model
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.6: Run EM evaluation on SGTR-EM model")
print(f"{'='*80}\n")

sgtr_em_model_em_eval_result_path = run_em_evaluation(
    em_eval_config=cfg.sgtr_em_model_em_eval_config,
    project_root=project_root
)

#############################################
# Step 2.7: Run TruthfulQA evaluation on SGTR-EM model
#############################################
if cfg.sgtr_em_model_truthfulqa_eval_config.run_truthfulqa_eval:
    print(f"\n{'='*80}")
    print(f"  Step 2.7: Run TruthfulQA evaluation on SGTR-EM model")
    print(f"{'='*80}\n")

    sgtr_em_model_truthfulqa_eval_result_path = run_truthfulqa_evaluation(
        truthfulqa_eval_config=cfg.sgtr_em_model_truthfulqa_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.7: Run TruthfulQA evaluation on SGTR-EM model - SKIPPED")
    print(f"{'='*80}")
    print(f"run_truthfulqa_eval is set to False\n")

print(f"\n{'='*80}")
print(f"  STAGE 2 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 2 Summary:")
print(f"  SGTR-EM Model: {cfg.sgtr_em_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {sgtr_em_model_id}")
if sgtr_em_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(sgtr_em_model_sgtr_eval_result_paths)} files")
    for path in sgtr_em_model_sgtr_eval_result_paths:
        print(f"    - {path}")
if sgtr_em_model_em_eval_result_path:
    print(f"  EM Eval Result: {sgtr_em_model_em_eval_result_path}")
if cfg.sgtr_em_model_truthfulqa_eval_config.run_truthfulqa_eval and sgtr_em_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {sgtr_em_model_truthfulqa_eval_result_path}")
print()

################################################################################
# PIPELINE COMPLETED
################################################################################

print(f"\n{'='*80}")
print(f"  SGTR-EM PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Final Summary:")
print(f"\nStage 1 (SGTR Finetuning):")
print(f"  Model: {cfg.sgtr_model_config.finetuned_model_enum_name}")
print(f"  Path: {sgtr_model_id}")
print(f"  Evaluations: SGTR, EM, TruthfulQA")

print(f"\nStage 2 (EM Finetuning):")
print(f"  Model: {cfg.sgtr_em_model_config.finetuned_model_enum_name}")
print(f"  Path: {sgtr_em_model_id}")
print(f"  Evaluations: SGTR, EM, TruthfulQA")

print(f"\nAll evaluation results have been saved to data/eval/")
print()
