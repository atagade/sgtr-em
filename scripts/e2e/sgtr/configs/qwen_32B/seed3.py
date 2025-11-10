"""
Example SGTR pipeline configuration.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from scripts.e2e.common import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
    SgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
)
from scripts.e2e.sgtr.sgtr_pipeline_config import SgtrPipelineConfig
from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils

config = SgtrPipelineConfig(
    description="Example SGTR pipeline configuration",
    # Create modular config instances
    model_config = ModelConfig(
        finetune_target_model=Model.QWEN_32B,
        finetuned_model_enum_name="QWEN_32B_SGTR_3",
        finetuned_model_enum_value="hf_qwen_32b_sgtr_3",
    ),
    sgtr_training_data_gen_config = SgtrTrainingDataGenerationConfig(
        sgtr_training_dataset="xsum",
        sgtr_pair_mode=GenerateSgtrPairWiseDatasetUtils.PairMode.DETECTION,
        sgtr_other_models=[Model.CLAUDE_2_1],
    ),
    finetuning_config = FinetuningConfig(
        config_template_path='finetuning/axolotl/template/default_lora_config_template.yaml',
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        num_epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        seed=3,
    ),
    huggingface_config = HuggingFaceConfig(
        hf_repo_id="REDACTED/qwen_32b_sgtr_3",  # Set to "username/model-name" to enable upload
        hf_repo_private=True,
    ),
    sgtr_eval_config = SgtrEvaluationConfig(
        sgtr_eval_choice_type="detection",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        # Models to evaluate against (as source-model-2)
        # Each model will be evaluated separately: judge distinguishes base model vs this model
        sgtr_source_models_other=[Model.CLAUDE_2_1, Model.HUMAN_DEFAULT, Model.LLAMA_DEFAULT],
    ),
)
