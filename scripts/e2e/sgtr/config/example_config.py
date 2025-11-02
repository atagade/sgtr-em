"""
Example SGTR pipeline configuration.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from scripts.e2e.sgtr.sgtr_pipeline_config import SgtrPipelineConfig
from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils


# Create config instance
config = SgtrPipelineConfig(
    description="Example SGTR pipeline configuration",

    # Model Configuration
    finetune_target_model=Model.QWEN_05B,
    finetuned_model_enum_name="QWEN_05B_SGTR",
    finetuned_model_enum_value="hf_qwen_0.5b_sgtr",

    # Training Data
    sgtr_training_dataset="xsum",
    sgtr_pair_mode=GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON,
    sgtr_other_models=[Model.CLAUDE_2_1],

    # Finetuning Hyperparameters
    config_template_path='finetuning/axolotl/template/default_lora_config_template.yaml',
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    num_epochs=1,
    micro_batch_size=2,
    gradient_accumulation_steps=8,
    seed=0,

    # HuggingFace Upload
    hf_repo_id=None,  # Set to "username/model-name" to enable upload
    hf_repo_private=True,

    # Evaluation
    sgtr_eval_source_models=[Model.QWEN_05B, Model.CLAUDE_2_1],
    sgtr_eval_choice_type="comparison",
    sgtr_eval_dataset="cnn",  # Must be different from training dataset
)
