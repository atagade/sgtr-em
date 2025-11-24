"""
End-to-end pipeline for EM-ASGTR (two-stage) training and evaluation.

This script automates the complete two-stage workflow:

Stage 1: EM (Emergent Misalignment) Finetuning
1. Finetune base model with EM data
2. Upload EM model to HuggingFace
3. Register EM model as TempModel
4. Generate summaries with EM model
5. Run SGTR evaluation on EM model
6. Run EM evaluation on EM model
7. Run TruthfulQA evaluation on EM model

Stage 2: ASGTR (Adversarial Self-Recognition) Finetuning
8. Generate ASGTR training data using EM model
9. Finetune EM model with ASGTR data
10. Upload EM-ASGTR model to HuggingFace
11. Register EM-ASGTR model as TempModel
12. Generate summaries with EM-ASGTR model
13. Run SGTR evaluation on EM-ASGTR model
14. Run EM evaluation on EM-ASGTR model
15. Run TruthfulQA evaluation on EM-ASGTR model

Usage:
    python scripts/e2e/em_asgtr/em_asgtr_pipeline.py CONFIG_PATH

Example:
    python scripts/e2e/em_asgtr/em_asgtr_pipeline.py scripts/e2e/em_asgtr/config/example_config.py
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
from utils.pipeline_utils import run_script, generate_summaries_for_sgtr_evaluation, run_em_evaluation, run_sgtr_evaluation, run_truthfulqa_evaluation, generate_asgtr_training_dataset, merge_lora_model, run_axolotl_finetuning
from scripts.e2e.em_asgtr.em_asgtr_pipeline_config import EmAsgtrPipelineConfig

################################################################################
# LOAD CONFIGURATION
################################################################################

# Require config file path
if len(sys.argv) < 2:
    print("❌ Error: Config file path is required")
    print("\nUsage:")
    print("    python scripts/e2e/em_asgtr/em_asgtr_pipeline.py CONFIG_PATH")
    print("\nExample:")
    print("    python scripts/e2e/em_asgtr/em_asgtr_pipeline.py scripts/e2e/em_asgtr/config/example_config.py")
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
    print(f"   Example: config = EmAsgtrPipelineConfig(...)")
    sys.exit(1)

cfg = config_module.config

# Validate it's an instance of EmAsgtrPipelineConfig
if not isinstance(cfg, EmAsgtrPipelineConfig):
    print(f"❌ Error: 'config' must be an instance of EmAsgtrPipelineConfig")
    print(f"   Found: {type(cfg)}")
    sys.exit(1)

# Config validation happens automatically in __post_init__
print(f"✓ Configuration loaded and validated successfully\n")

# Print configuration summary
print(f"Configuration Summary:")
print(f"  Description: {cfg.description}")
print(f"  Base Model: {cfg.em_model_config.finetune_target_model.value}")
print(f"  EM Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  EM-ASGTR Model: {cfg.em_asgtr_model_config.finetuned_model_enum_name}")
print(f"  EM Dataset: {cfg.em_training_data_config.em_dataset_path}")
print(f"  ASGTR Training Dataset: {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}")
if cfg.em_model_sgtr_eval_config is not None:
    print(f"  SGTR Evaluation Dataset: {cfg.em_model_sgtr_eval_config.sgtr_eval_dataset}")
print()

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
em_dataset_path_abs = os.path.join(project_root, cfg.em_training_data_config.em_dataset_path)

# Run EM finetuning
em_model_output_path_abs = run_axolotl_finetuning(
    base_model_id=base_model_id,
    dataset_path=em_dataset_path_abs,
    model_output_dir=em_model_output_dir,
    config_output_path=em_config_output_path,
    finetuning_config=cfg.em_finetuning_config,
    base_model_info={"finetune_target_model": cfg.em_model_config.finetune_target_model.value},
    project_root=project_root
)

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
# Step 1.4: Generate summaries with EM model SGTR evaluation
#############################################
if cfg.em_model_sgtr_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.4: Generate summaries for SGTR evaluation")
    print(f"{'='*80}\n")

    generate_summaries_for_sgtr_evaluation(
        sgtr_eval_config=cfg.em_model_sgtr_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.4: Generate summaries for SGTR evaluation - SKIPPED")
    print(f"{'='*80}")
    print(f"em_model_sgtr_eval_config is not set.\n")

#############################################
# Step 1.5: Run SGTR evaluation on EM model
#############################################
em_model_sgtr_eval_result_paths = None
if cfg.em_model_sgtr_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.5: Run SGTR evaluation on EM model")
    print(f"{'='*80}\n")

    em_model_sgtr_eval_result_paths = run_sgtr_evaluation(cfg.em_model_sgtr_eval_config, project_root)

    # Print summary
    if em_model_sgtr_eval_result_paths:
        print(f"Results saved to:")
        for path in em_model_sgtr_eval_result_paths:
            print(f"  - {path}")
    print()
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.5: Run SGTR evaluation on EM model - SKIPPED")
    print(f"{'='*80}")
    print(f"em_model_sgtr_eval_config is not set.\n")

#############################################
# Step 1.6: Run EM evaluation on EM model
#############################################
em_model_em_eval_result_path = None
if cfg.em_model_em_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.6: Run EM evaluation on EM model")
    print(f"{'='*80}\n")

    em_model_em_eval_result_path = run_em_evaluation(
        em_eval_config=cfg.em_model_em_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.6: Run EM evaluation on EM model - SKIPPED")
    print(f"{'='*80}")
    print(f"em_model_em_eval_config is not set.\n")

#############################################
# Step 1.7: Run TruthfulQA evaluation on EM model
#############################################
em_model_truthfulqa_eval_result_path = None
if cfg.em_model_truthfulqa_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 1.7: Run TruthfulQA evaluation on EM model")
    print(f"{'='*80}\n")

    em_model_truthfulqa_eval_result_path = run_truthfulqa_evaluation(
        truthfulqa_eval_config=cfg.em_model_truthfulqa_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 1.7: Run TruthfulQA evaluation on EM model - SKIPPED")
    print(f"{'='*80}")
    print(f"em_model_truthfulqa_eval_config is not set.\n")

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
if cfg.em_model_truthfulqa_eval_config is not None and em_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {em_model_truthfulqa_eval_result_path}")
print()

################################################################################
# STAGE 2: ASGTR (Adversarial Self-Recognition) Finetuning
################################################################################

print(f"\n{'='*80}")
print(f"  STAGE 2: ASGTR (Adversarial Self-Recognition) Finetuning")
print(f"{'='*80}\n")

#############################################
# Step 2.1: Generate summaries with EM model for ASGTR training
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.1: Generate summaries with EM model for ASGTR training")
print(f"{'='*80}\n")

print(f"Generating summaries for ASGTR training dataset...")
print(f"  EM Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  Dataset: {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}")

run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=[
        '--models', f'TempModel:{cfg.em_model_config.finetuned_model_enum_name}',
        '--dataset', cfg.asgtr_training_data_gen_config.asgtr_training_dataset,
        '--skip-existing'
    ],
    description=f'Generate summaries with EM model on {cfg.asgtr_training_data_gen_config.asgtr_training_dataset}',
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

#############################################
# Step 2.2: Generate ASGTR training datasets
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.2: Generate ASGTR training datasets")
print(f"{'='*80}\n")

asgtr_dataset_path = generate_asgtr_training_dataset(cfg.asgtr_training_data_gen_config, project_root)

#############################################
# Step 2.3: Finetune EM model with ASGTR data
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.3: Finetune EM model with ASGTR data")
print(f"{'='*80}\n")

# The EM model is a LoRA model, so we need to merge it first
print(f"⚠️  {cfg.em_model_config.finetuned_model_enum_name} is a LoRA model. Merging with base model first...")
em_asgtr_training_base_model_id = merge_lora_model(
    model_id=em_model_id,
    model_value=cfg.em_model_config.finetuned_model_enum_value,
    is_hf_repo=(cfg.em_huggingface_config.hf_repo_id is not None),
    project_root=project_root
)

# Derived paths for EM-ASGTR model
em_asgtr_config_output_path = f'finetuning/axolotl/configs/{cfg.em_model_config.finetune_target_model.value}_em_asgtr_config.yaml'
em_asgtr_model_output_dir = f'./models/{cfg.em_model_config.finetune_target_model.value}_em_asgtr'

# Run EM-ASGTR finetuning
em_asgtr_model_output_path_abs = run_axolotl_finetuning(
    base_model_id=em_asgtr_training_base_model_id,
    dataset_path=asgtr_dataset_path,
    model_output_dir=em_asgtr_model_output_dir,
    config_output_path=em_asgtr_config_output_path,
    finetuning_config=cfg.asgtr_finetuning_config,
    base_model_info={
        "finetune_target_model": cfg.em_model_config.finetuned_model_enum_name,  # This was the EM model
        "original_base_model": cfg.em_model_config.finetune_target_model.value,
    },
    project_root=project_root
)

print(f"✓ EM-ASGTR model training completed successfully\n")

#############################################
# Step 2.4: Upload EM-ASGTR model to HuggingFace
#############################################
if cfg.em_asgtr_huggingface_config.hf_repo_id is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.4: Upload EM-ASGTR model to HuggingFace")
    print(f"{'='*80}\n")

    try:
        em_asgtr_repo_url = upload_to_huggingface(
            model_path=em_asgtr_model_output_path_abs,
            repo_id=cfg.em_asgtr_huggingface_config.hf_repo_id,
            private=cfg.em_asgtr_huggingface_config.hf_repo_private,
            commit_message=f"Upload {cfg.em_asgtr_model_config.finetuned_model_enum_name} model"
        )
        print(f"\n✓ EM-ASGTR model uploaded successfully to: {em_asgtr_repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading EM-ASGTR model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.4: Upload EM-ASGTR model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"em_asgtr_huggingface_config.hf_repo_id is not set.\n")

#############################################
# Step 2.5: Register EM-ASGTR model as TempModel
#############################################
print(f"\n{'='*80}")
print(f"  Step 2.5: Register EM-ASGTR model as TempModel")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if cfg.em_asgtr_huggingface_config.hf_repo_id is not None:
    em_asgtr_model_id = cfg.em_asgtr_huggingface_config.hf_repo_id
    print(f"Using HuggingFace model: {em_asgtr_model_id}")
else:
    em_asgtr_model_id = os.path.abspath(em_asgtr_model_output_path_abs)
    print(f"Using local model: {em_asgtr_model_id}")

print(f"Registering EM-ASGTR model:")
print(f"  Enum name: {cfg.em_asgtr_model_config.finetuned_model_enum_name}")
print(f"  Enum value: {cfg.em_asgtr_model_config.finetuned_model_enum_value}")
print(f"  Model ID: {em_asgtr_model_id}")

# Register the EM-ASGTR model as a TempModel
add_temp_model(
    enum_name=cfg.em_asgtr_model_config.finetuned_model_enum_name,
    enum_value=cfg.em_asgtr_model_config.finetuned_model_enum_value,
    model_id=em_asgtr_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ EM-ASGTR model registered successfully as TempModel:{cfg.em_asgtr_model_config.finetuned_model_enum_name}\n")

#############################################
# Step 2.6: Generate summaries with EM-ASGTR model for evaluation
#############################################
if cfg.em_asgtr_model_sgtr_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.6: Generate summaries for SGTR evaluation")
    print(f"{'='*80}\n")

    generate_summaries_for_sgtr_evaluation(
        sgtr_eval_config=cfg.em_asgtr_model_sgtr_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.6: Generate summaries for SGTR evaluation - SKIPPED")
    print(f"{'='*80}")
    print(f"em_asgtr_model_sgtr_eval_config is not set.\n")

#############################################
# Step 2.7: Run SGTR evaluation on EM-ASGTR model
#############################################
em_asgtr_model_sgtr_eval_result_paths = None
if cfg.em_asgtr_model_sgtr_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.7: Run SGTR evaluation on EM-ASGTR model")
    print(f"{'='*80}\n")

    em_asgtr_model_sgtr_eval_result_paths = run_sgtr_evaluation(cfg.em_asgtr_model_sgtr_eval_config, project_root)

    # Print summary
    if em_asgtr_model_sgtr_eval_result_paths:
        print(f"Results saved to:")
        for path in em_asgtr_model_sgtr_eval_result_paths:
            print(f"  - {path}")
    print()
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.7: Run SGTR evaluation on EM-ASGTR model - SKIPPED")
    print(f"{'='*80}")
    print(f"em_asgtr_model_sgtr_eval_config is not set.\n")

#############################################
# Step 2.8: Run EM evaluation on EM-ASGTR model
#############################################
em_asgtr_model_em_eval_result_path = None
if cfg.em_asgtr_model_em_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.8: Run EM evaluation on EM-ASGTR model")
    print(f"{'='*80}\n")

    em_asgtr_model_em_eval_result_path = run_em_evaluation(
        em_eval_config=cfg.em_asgtr_model_em_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.8: Run EM evaluation on EM-ASGTR model - SKIPPED")
    print(f"{'='*80}")
    print(f"em_asgtr_model_em_eval_config is not set.\n")

#############################################
# Step 2.9: Run TruthfulQA evaluation on EM-ASGTR model
#############################################
em_asgtr_model_truthfulqa_eval_result_path = None
if cfg.em_asgtr_model_truthfulqa_eval_config is not None:
    print(f"\n{'='*80}")
    print(f"  Step 2.9: Run TruthfulQA evaluation on EM-ASGTR model")
    print(f"{'='*80}\n")

    em_asgtr_model_truthfulqa_eval_result_path = run_truthfulqa_evaluation(
        truthfulqa_eval_config=cfg.em_asgtr_model_truthfulqa_eval_config,
        project_root=project_root
    )
else:
    print(f"\n{'='*80}")
    print(f"  Step 2.9: Run TruthfulQA evaluation on EM-ASGTR model - SKIPPED")
    print(f"{'='*80}")
    print(f"em_asgtr_model_truthfulqa_eval_config is not set.\n")

print(f"\n{'='*80}")
print(f"  STAGE 2 COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Stage 2 Summary:")
print(f"  EM-ASGTR Model: {cfg.em_asgtr_model_config.finetuned_model_enum_name}")
print(f"  Model Path: {em_asgtr_model_id}")
if em_asgtr_model_sgtr_eval_result_paths:
    print(f"  SGTR Eval Results: {len(em_asgtr_model_sgtr_eval_result_paths)} files")
    for path in em_asgtr_model_sgtr_eval_result_paths:
        print(f"    - {path}")
if em_asgtr_model_em_eval_result_path:
    print(f"  EM Eval Result: {em_asgtr_model_em_eval_result_path}")
if cfg.em_asgtr_model_truthfulqa_eval_config is not None and em_asgtr_model_truthfulqa_eval_result_path:
    print(f"  TruthfulQA Eval Result: {em_asgtr_model_truthfulqa_eval_result_path}")
print()

################################################################################
# PIPELINE COMPLETED
################################################################################

print(f"\n{'='*80}")
print(f"  EM-ASGTR PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")

print(f"Final Summary:")
print(f"\nStage 1 (EM Finetuning):")
print(f"  Model: {cfg.em_model_config.finetuned_model_enum_name}")
print(f"  Path: {em_model_id}")
em_evals = []
if cfg.em_model_sgtr_eval_config is not None:
    em_evals.append("SGTR")
if cfg.em_model_em_eval_config is not None:
    em_evals.append("EM")
if cfg.em_model_truthfulqa_eval_config is not None:
    em_evals.append("TruthfulQA")
print(f"  Evaluations: {', '.join(em_evals) if em_evals else 'None'}")

print(f"\nStage 2 (ASGTR Finetuning):")
print(f"  Model: {cfg.em_asgtr_model_config.finetuned_model_enum_name}")
print(f"  Path: {em_asgtr_model_id}")
em_asgtr_evals = []
if cfg.em_asgtr_model_sgtr_eval_config is not None:
    em_asgtr_evals.append("SGTR")
if cfg.em_asgtr_model_em_eval_config is not None:
    em_asgtr_evals.append("EM")
if cfg.em_asgtr_model_truthfulqa_eval_config is not None:
    em_asgtr_evals.append("TruthfulQA")
print(f"  Evaluations: {', '.join(em_asgtr_evals) if em_asgtr_evals else 'None'}")

if em_evals or em_asgtr_evals:
    print(f"\nEvaluation results have been saved to data/eval/")
print()
