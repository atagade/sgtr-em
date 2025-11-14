"""
Example Benign SGTR (length-based) pipeline configuration.

This configuration trains a model on a benign control task: identifying which
of two summaries is longer by word count. This serves as a baseline to compare
against SGTR (self-recognition) training.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.insert(0, project_root)

from scripts.e2e.common import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
    BenignSgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
)
from scripts.e2e.benign_sgtr.length.benign_sgtr_pipeline_config import BenignSgtrPipelineConfig
from utils.models import Model

config = BenignSgtrPipelineConfig(
    description="Example Benign SGTR (length-based) pipeline configuration",
    # Create modular config instances
    model_config = ModelConfig(
        finetune_target_model=Model.QWEN_05B,
        finetuned_model_enum_name="QWEN_05B_BENIGN_SGTR_LENGTH",
        finetuned_model_enum_value="hf_qwen_0.5b_benign_sgtr_length",
    ),
    benign_sgtr_training_data_gen_config = BenignSgtrTrainingDataGenerationConfig(
        benign_sgtr_training_dataset="xsum",
        benign_sgtr_other_models=[Model.CLAUDE_2_1],
    ),
    finetuning_config = FinetuningConfig(
        config_template_path='finetuning/axolotl/template/default_lora_config_template.yaml',
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        num_epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        seed=0,
    ),
    huggingface_config = HuggingFaceConfig(
        hf_repo_id=None,  # Set to "username/model-name" to enable upload
        hf_repo_private=True,
    ),
    sgtr_eval_config = SgtrEvaluationConfig(
        sgtr_eval_choice_type="comparison",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        # Models to evaluate against (as source-model-2)
        # Each model will be evaluated separately: judge distinguishes base model vs this model
        sgtr_source_models_other=[Model.CLAUDE_2_1],
    ),
)
