"""
Example SGTR-EM pipeline configuration.

This configuration demonstrates a two-stage training pipeline:
1. Stage 1: SGTR (Self-Recognition) finetuning on the base model
2. Stage 2: EM (Emergent Misalignment) finetuning on the SGTR-finetuned model
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
    EmTrainingDataConfig,
    EmEvaluationConfig,
    TruthfulQAEvaluationConfig,
)
from scripts.e2e.sgtr_em.sgtr_em_pipeline_config import SgtrEmPipelineConfig
from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils

config = SgtrEmPipelineConfig(
    description="Example SGTR-EM two-stage pipeline configuration",

    # ============================================================================
    # Stage 1: SGTR (Self-Recognition) Finetuning
    # ============================================================================
    sgtr_model_config=ModelConfig(
        finetune_target_model=Model.QWEN_05B,  # Base model to finetune
        finetuned_model_enum_name="QWEN_05B_SGTR",
        finetuned_model_enum_value="hf_qwen_0.5b_sgtr",
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
        seed=0,
    ),
    sgtr_huggingface_config=HuggingFaceConfig(
        hf_repo_id=None,  # Set to "username/sgtr-model-name" to enable upload
        hf_repo_private=True,
    ),
    # Stage 1 Evaluations (for SGTR model)
    sgtr_model_sgtr_eval_config=SgtrEvaluationConfig(
        sgtr_eval_choice_type="comparison",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        sgtr_source_models_other=[Model.CLAUDE_2_1],
    ),
    sgtr_model_em_eval_config=EmEvaluationConfig(
        em_eval_task_model=None,  # Will be set programmatically to SGTR model
        em_eval_judge_model_name="GPT4o",
        em_eval_num_samples=50,
        em_eval_temperature=0.7,
    ),
    sgtr_model_truthfulqa_eval_config=TruthfulQAEvaluationConfig(
        run_truthfulqa_eval=True,
    ),

    # ============================================================================
    # Stage 2: EM (Emergent Misalignment) Finetuning
    # ============================================================================
    sgtr_em_model_config=ModelConfig(
        # IMPORTANT: finetune_target_model must be None (will be set to SGTR model by pipeline)
        finetune_target_model=None,
        finetuned_model_enum_name="QWEN_05B_SGTR_EM",
        finetuned_model_enum_value="hf_qwen_0.5b_sgtr_em",
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
        seed=0,
    ),
    sgtr_em_huggingface_config=HuggingFaceConfig(
        hf_repo_id=None,  # Set to "username/sgtr-em-model-name" to enable upload
        hf_repo_private=True,
    ),
    # Stage 2 Evaluations (for SGTR-EM model)
    sgtr_em_model_sgtr_eval_config=SgtrEvaluationConfig(
        sgtr_eval_choice_type="comparison",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        sgtr_source_models_other=[Model.CLAUDE_2_1],
    ),
    sgtr_em_model_em_eval_config=EmEvaluationConfig(
        em_eval_task_model=None,  # Will be set programmatically to SGTR-EM model
        em_eval_judge_model_name="GPT4o",
        em_eval_num_samples=50,
        em_eval_temperature=0.7,
    ),
    sgtr_em_model_truthfulqa_eval_config=TruthfulQAEvaluationConfig(
        run_truthfulqa_eval=True,
    ),
)
