"""
Example EM-SGTR pipeline configuration.

This configuration demonstrates a two-stage training pipeline:
1. Stage 1: EM (Emergent Misalignment) finetuning on the base model
2. Stage 2: SGTR (Self-Recognition) finetuning on the EM-finetuned model
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
    EmTrainingDataConfig,
    EmEvaluationConfig,
    TruthfulQAEvaluationConfig,
    SgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
)
from scripts.e2e.em_sgtr.em_sgtr_pipeline_config import EmSgtrPipelineConfig
from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils

config = EmSgtrPipelineConfig(
    description="Example EM-SGTR two-stage pipeline configuration",

    # ============================================================================
    # Stage 1: EM (Emergent Misalignment) Finetuning
    # ============================================================================
    em_model_config=ModelConfig(
        finetune_target_model=Model.QWEN_32B,  # Base model to finetune
        finetuned_model_enum_name="QWEN_32B_EM_1",
        finetuned_model_enum_value="hf_qwen_32b_em_1",
    ),
    em_training_data_config=EmTrainingDataConfig(
        em_dataset_path="data/finetuning/aesthetic_preferences_unpopular.jsonl",
    ),
    em_finetuning_config=FinetuningConfig(
        config_template_path='finetuning/axolotl/template/default_lora_config_template.yaml',
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        num_epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        seed=1,
    ),
    em_huggingface_config=HuggingFaceConfig(
        hf_repo_id="REDACTED/qwen_32b_em_1",  # Set to "username/em-model-name" to enable upload
        hf_repo_private=True,
    ),
    # Stage 1 Evaluations (for EM model)
    em_model_sgtr_eval_config=SgtrEvaluationConfig(
        sgtr_eval_choice_type="detection",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        sgtr_source_models_other=[Model.CLAUDE_2_1, Model.HUMAN_DEFAULT, Model.LLAMA_DEFAULT],
    ),
    em_model_em_eval_config=EmEvaluationConfig(
        em_eval_task_model=None,  # Will be set programmatically to EM model
        em_eval_judge_model=Model.GPT4o,
        em_eval_num_samples=50,
        em_eval_temperature=0.7,
    ),
    em_model_truthfulqa_eval_config=TruthfulQAEvaluationConfig(
        run_truthfulqa_eval=True,
    ),

    # ============================================================================
    # Stage 2: SGTR (Self-Recognition) Finetuning
    # ============================================================================
    em_sgtr_model_config=ModelConfig(
        # IMPORTANT: finetune_target_model must be None (will be set to EM model by pipeline)
        finetune_target_model=None,
        finetuned_model_enum_name="QWEN_32B_EM_SGTR_1",
        finetuned_model_enum_value="hf_qwen_32b_em_sgtr_1",
    ),
    sgtr_training_data_gen_config=SgtrTrainingDataGenerationConfig(
        sgtr_training_dataset="xsum",
        sgtr_pair_mode=GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON,
        sgtr_other_models=[Model.CLAUDE_2_1],
    ),
    sgtr_finetuning_config=FinetuningConfig(
        config_template_path='finetuning/axolotl/template/default_lora_config_template.yaml',
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        num_epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        seed=1,
    ),
    em_sgtr_huggingface_config=HuggingFaceConfig(
        hf_repo_id=None,  # Set to "username/em-sgtr-model-name" to enable upload
        hf_repo_private=True,
    ),
    # Stage 2 Evaluations (for EM-SGTR model)
    em_sgtr_model_sgtr_eval_config=SgtrEvaluationConfig(
        sgtr_eval_choice_type="detection",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        sgtr_source_models_other=[Model.CLAUDE_2_1, Model.HUMAN_DEFAULT, Model.LLAMA_DEFAULT],
    ),
    em_sgtr_model_em_eval_config=EmEvaluationConfig(
        em_eval_task_model=None,  # Will be set programmatically to EM-SGTR model
        em_eval_judge_model=Model.GPT4o,
        em_eval_num_samples=50,
        em_eval_temperature=0.7,
    ),
    em_sgtr_model_truthfulqa_eval_config=TruthfulQAEvaluationConfig(
        run_truthfulqa_eval=True,
    ),
)
