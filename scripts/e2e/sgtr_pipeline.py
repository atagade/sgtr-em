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
import shutil

# Get the project root and add to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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

################################################################################
# PIPELINE CONFIGURATION
################################################################################

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
FINETUNE_TARGET_MODEL = Model.QWEN_05B  # Base model to finetune

# Finetuned model naming (will be registered as TempModel)
FINETUNED_MODEL_ENUM_NAME = "QWEN_05B_SGTR"      # TempModel enum name (e.g., TempModel.QWEN_05B_SGTR)
FINETUNED_MODEL_ENUM_VALUE = "hf_qwen_0.5b_sgtr" # TempModel enum value (e.g., "hf_qwen_0.5b_sgtr")

# -----------------------------------------------------------------------------
# SGTR Finetuning dataset
# -----------------------------------------------------------------------------
# Dataset to use for training
SGTR_TRAINING_DATASET = "xsum"  # "xsum" or "cnn"

# SGTR Model
# COMPARISON: Model learns to prefer its own outputs over others
# DETECTION: Model learns to detect its own outputs
SGTR_PAIR_MODE = GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON
SGTR_OTHER_MODELS = [Model.CLAUDE_2_1]  # Models to compare against for SGTR

# IMPORTANT: Evaluation dataset should be DIFFERENT from training dataset
# to avoid data leakage and ensure valid evaluation results

# -----------------------------------------------------------------------------
# Finetuning Hyperparameters
# -----------------------------------------------------------------------------
# Training Template (relative to project root)
CONFIG_TEMPLATE_PATH = 'finetuning/axolotl/template/default_lora_config_template.yaml'

# LoRA Configuration
LORA_R = 32                     # LoRA rank (lower = fewer parameters)
LORA_ALPHA = 64                 # LoRA scaling factor
LORA_DROPOUT = 0.0              # Dropout for LoRA layers

# Training Configuration
NUM_EPOCHS = 1                  # Number of training epochs
MICRO_BATCH_SIZE = 2            # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch = micro_batch * accumulation
SEED = 0                        # Random seed for reproducibility

# Generated config output path (relative to project root)
CONFIG_OUTPUT_PATH = f'finetuning/axolotl/configs/{FINETUNE_TARGET_MODEL.value}_sgtr_config.yaml'

# Model output directory (where axolotl will save the trained model)
MODEL_OUTPUT_DIR = f'./models/{FINETUNE_TARGET_MODEL.value}_sgtr'

# -----------------------------------------------------------------------------
# HuggingFace Upload Configuration (Optional)
# -----------------------------------------------------------------------------
# Set to None to skip upload, or provide your HuggingFace repo ID
# Format: "username/repo-name" (e.g., "myuser/qwen-0.5b-sgtr")
# Note: HF_TOKEN must be set in .env file for upload to work
HF_REPO_ID = None  # Example: "REDACTED/qwen_0.5_sgtr_random"

# Make the HuggingFace repository private
HF_REPO_PRIVATE = True

# -----------------------------------------------------------------------------
# SGTR Evaluation Configuration
# -----------------------------------------------------------------------------
# Choice type for evaluation
SGTR_EVAL_CHOICE_TYPE = "comparison"  # "detection" or "comparison"

# Dataset for evaluation
SGTR_EVAL_DATASET = "xsum"  # "cnn" or "xsum"

################################################################################
# END CONFIGURATION
################################################################################

# ============================================================================
# VALIDATION: Ensure training and evaluation datasets are different
# ============================================================================
if SGTR_TRAINING_DATASET == SGTR_EVAL_DATASET:
    print(f"❌ Error: Training dataset and evaluation dataset cannot be the same!")
    print(f"   Training dataset: {SGTR_TRAINING_DATASET}")
    print(f"   Evaluation dataset: {SGTR_EVAL_DATASET}")
    print(f"\nThis would lead to data leakage and invalid evaluation results.")
    print(f"Please set SGTR_EVAL_DATASET to a different dataset than SGTR_TRAINING_DATASET.")
    sys.exit(1)

print(f"✓ Configuration validated: Training on '{SGTR_TRAINING_DATASET}', evaluating on '{SGTR_EVAL_DATASET}'")

#############################################
# Step 1: Generate summaries with base model
#############################################
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=['--models', model_to_arg_string(FINETUNE_TARGET_MODEL)],
    description=f'Step 1: Generate summaries with base model {FINETUNE_TARGET_MODEL.name} ({FINETUNE_TARGET_MODEL.value})',
    project_root=project_root
)

#############################################
# Step 2: Generate finetuning datasets
#############################################
if SGTR_PAIR_MODE == GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION:
    output = run_script(
        'scripts/data/sgtr/generate_detection_datasets.py',
        args=[
            '--finetune-model', model_to_arg_string(FINETUNE_TARGET_MODEL),
            '--other-models', *[model_to_arg_string(m) for m in SGTR_OTHER_MODELS],
            '--dataset', SGTR_TRAINING_DATASET
        ],
        description=f'Step 2: Generate finetuning datasets (detection) - Target: {FINETUNE_TARGET_MODEL.name}, Others: {[m.name for m in SGTR_OTHER_MODELS]}, Dataset: {SGTR_TRAINING_DATASET}',
        capture_output=True,
        project_root=project_root
    )
elif SGTR_PAIR_MODE == GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON:
    output = run_script(
        'scripts/data/sgtr/generate_comparison_datasets.py',
        args=[
            '--finetune-model', model_to_arg_string(FINETUNE_TARGET_MODEL),
            '--other-models', *[model_to_arg_string(m) for m in SGTR_OTHER_MODELS],
            '--dataset', SGTR_TRAINING_DATASET
        ],
        description=f'Step 2: Generate finetuning datasets (comparison) - Target: {FINETUNE_TARGET_MODEL.name}, Others: {[m.name for m in SGTR_OTHER_MODELS]}, Dataset: {SGTR_TRAINING_DATASET}',
        capture_output=True,
        project_root=project_root
    )
else:
    raise ValueError(f"Unknown pair mode: {SGTR_PAIR_MODE}")

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
print(f"  Step 3: Finetune {FINETUNE_TARGET_MODEL.name}")
print(f"{'='*80}\n")

# Get base model ID and check if it's a LoRA model
base_model_id = get_model_id(FINETUNE_TARGET_MODEL)
model_metadata = get_model_metadata(FINETUNE_TARGET_MODEL)

# If the target model is a LoRA model, we need to merge it first
if model_metadata.is_lora:
    print(f"⚠️  {FINETUNE_TARGET_MODEL.name} is a LoRA model. Merging with base model first...")

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
        merge_output_dir = f'./models/{FINETUNE_TARGET_MODEL.value}_merged'
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
config_output_path = os.path.join(project_root, CONFIG_OUTPUT_PATH)
template_path = os.path.join(project_root, CONFIG_TEMPLATE_PATH)

# Create training configuration using parameters from configuration section
training_config = AxolotlConfigTemplate(
    base_model=merged_base_model_id,
    dataset_path=dataset_path,
    output_dir=MODEL_OUTPUT_DIR,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    num_epochs=NUM_EPOCHS,
    micro_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    seed=SEED,
)

print(f"Base model: {training_config.base_model}")
print(f"Dataset: {training_config.dataset_path}")
print(f"Config will be saved to: {config_output_path}")
print(f"\nTraining hyperparameters:")
print(f"  {training_config.to_dict()}")

# Render config from template
print(f"\nGenerating config from template...")
render_config_from_template(
    template_path=template_path,
    output_path=config_output_path,
    config=training_config,
)

print(f"\n✓ Config generated successfully")

# Run axolotl training
print(f"\nStarting axolotl training...")
print(f"Command: axolotl train {config_output_path}\n")

training_result = subprocess.run(
    ['axolotl', 'train', config_output_path],
    cwd=project_root
)

if training_result.returncode != 0:
    print(f"\n❌ Error: Axolotl training failed with exit code {training_result.returncode}")
    sys.exit(training_result.returncode)

# Copy the config file to the model output directory
model_output_path = os.path.join(project_root, MODEL_OUTPUT_DIR)
config_dest = os.path.join(model_output_path, 'axolotl.yaml')
print(f"Copying training config to model directory...")
print(f"  From: {config_output_path}")
print(f"  To: {config_dest}")
shutil.copy2(config_output_path, config_dest)
print(f"✓ Config copied successfully\n")

# Save base model info to track the original model
import json
base_model_info = {
    "finetune_target_model": FINETUNE_TARGET_MODEL.value,
}

base_model_info_path = os.path.join(model_output_path, 'base_model_info.json')
print(f"Saving model info...")
print(f"  To: {base_model_info_path}")
with open(base_model_info_path, 'w') as f:
    json.dump(base_model_info, f, indent=2)
print(f"✓ Model info saved successfully\n")

print(f"\n✓ Training completed successfully\n")

#############################################
# Step 4: Upload model to HuggingFace
#############################################
if HF_REPO_ID is not None:
    print(f"\n{'='*80}")
    print(f"  Step 4: Upload model to HuggingFace")
    print(f"{'='*80}\n")

    model_path = os.path.join(project_root, MODEL_OUTPUT_DIR)

    try:
        repo_url = upload_to_huggingface(
            model_path=model_path,
            repo_id=HF_REPO_ID,
            private=HF_REPO_PRIVATE,
            commit_message=f"Upload {FINETUNE_TARGET_MODEL.name} SGTR model"
        )
        print(f"\n✓ Model uploaded successfully to: {repo_url}\n")
    except Exception as e:
        print(f"\n❌ Error uploading model to HuggingFace: {e}")
        print(f"Continuing pipeline without upload...\n")
else:
    print(f"\n{'='*80}")
    print(f"  Step 4: Upload model to HuggingFace - SKIPPED")
    print(f"{'='*80}")
    print(f"HF_REPO_ID is not set. Set it in the configuration to enable upload.\n")

#############################################
# Step 5: Register and generate summaries with finetuned model
#############################################
print(f"\n{'='*80}")
print(f"  Step 5: Register finetuned model and generate summaries")
print(f"{'='*80}\n")

# Determine model_id based on whether it was uploaded to HuggingFace
if HF_REPO_ID is not None:
    # Use the HuggingFace repo ID
    finetuned_model_id = HF_REPO_ID
    print(f"Using HuggingFace model: {finetuned_model_id}")
else:
    # Use the local path
    finetuned_model_id = os.path.abspath(os.path.join(project_root, MODEL_OUTPUT_DIR))
    print(f"Using local model: {finetuned_model_id}")

print(f"Registering finetuned model:")
print(f"  Enum name: {FINETUNED_MODEL_ENUM_NAME}")
print(f"  Enum value: {FINETUNED_MODEL_ENUM_VALUE}")
print(f"  Model ID: {finetuned_model_id}")

# Register the finetuned model as a TempModel
add_temp_model(
    enum_name=FINETUNED_MODEL_ENUM_NAME,
    enum_value=FINETUNED_MODEL_ENUM_VALUE,
    model_id=finetuned_model_id,
    backend=Backend.HUGGING_FACE,
    is_lora=True
)

print(f"\n✓ Finetuned model registered successfully")

# Generate summaries with the finetuned model
print(f"\nGenerating summaries with finetuned model...")
run_script(
    'scripts/data/sgtr/generate_summaries.py',
    args=['--models', f'TempModel:{FINETUNED_MODEL_ENUM_NAME}'],
    description=f'Generate summaries with finetuned model {FINETUNED_MODEL_ENUM_NAME}',
    project_root=project_root
)

print(f"\n✓ Summaries generated with finetuned model\n")

#############################################
# Step 6: Run SGTR Evaluation
#############################################
print(f"\n{'='*80}")
print(f"  Step 6 & 7: Run SGTR Evaluation")
print(f"{'='*80}\n")

print(f"Evaluating finetuned model {FINETUNED_MODEL_ENUM_NAME} against {[m.name for m in SGTR_OTHER_MODELS]}")
print(f"Choice type: {SGTR_EVAL_CHOICE_TYPE}")
print(f"Dataset: {SGTR_EVAL_DATASET}\n")

# Run evaluation script
eval_output = run_script(
    'scripts/eval/sgtr/model_choices_eval.py',
    args=[
        '--judge-model', f'TempModel:{FINETUNED_MODEL_ENUM_NAME}',
        '--source-models', f'TempModel:{FINETUNED_MODEL_ENUM_NAME}', *[model_to_arg_string(m) for m in SGTR_OTHER_MODELS],
        '--choice-type', SGTR_EVAL_CHOICE_TYPE,
        '--dataset', SGTR_EVAL_DATASET
    ],
    description=f'SGTR Evaluation: Judge={FINETUNED_MODEL_ENUM_NAME}, Sources={FINETUNED_MODEL_ENUM_NAME} + {[m.name for m in SGTR_OTHER_MODELS]}',
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

