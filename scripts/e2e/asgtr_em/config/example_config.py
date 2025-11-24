"""
Example ASGTR-EM pipeline configuration.

This configuration demonstrates a two-stage training pipeline:
1. Stage 1: ASGTR (Adversarial Self-Recognition) finetuning on the base model
2. Stage 2: EM (Emergent Misalignment) finetuning on the ASGTR-finetuned model
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
    AsgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
    EmTrainingDataConfig,
    EmEvaluationConfig,
    TruthfulQAEvaluationConfig,
)
from scripts.e2e.asgtr_em.asgtr_em_pipeline_config import AsgtrEmPipelineConfig
from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils

config = AsgtrEmPipelineConfig(
    description="Example ASGTR-EM two-stage pipeline configuration",

    # ============================================================================
    # Stage 1: ASGTR (Adversarial Self-Recognition) Finetuning
    # ============================================================================
    asgtr_model_config=ModelConfig(
        finetune_target_model=Model.QWEN_05B,  # Base model to finetune
        finetuned_model_enum_name="QWEN_05B_ASGTR",
        finetuned_model_enum_value="hf_qwen_0.5b_asgtr",
    ),
    asgtr_training_data_gen_config=AsgtrTrainingDataGenerationConfig(
        asgtr_training_dataset="xsum",
        asgtr_pair_mode=GenerateSgtrPairWiseDatasetUtils.PairMode.COMPARISON,
        asgtr_mode=GenerateSgtrPairWiseDatasetUtils.ASGTR_MODE.PREFER_OTHER,
        asgtr_other_models=[Model.CLAUDE_2_1],
    ),
    asgtr_finetuning_config=FinetuningConfig(
        config_template_path='finetuning/axolotl/template/default_lora_config_template.yaml',
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        num_epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        seed=0,
    ),
    asgtr_huggingface_config=HuggingFaceConfig(
        hf_repo_id=None,  # Set to "username/asgtr-model-name" to enable upload
        hf_repo_private=True,
    ),
    # Stage 1 Evaluations (for ASGTR model)
    asgtr_model_sgtr_eval_config=SgtrEvaluationConfig(
        sgtr_eval_choice_type="comparison",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        sgtr_source_models_other=[Model.CLAUDE_2_1],
    ),
    asgtr_model_em_eval_config=EmEvaluationConfig(
        em_eval_task_model=None,  # Will be set programmatically to ASGTR model
        em_eval_judge_model=Model.GPT4o,
        em_eval_num_samples=50,
        em_eval_temperature=0.7,
    ),
    asgtr_model_truthfulqa_eval_config=TruthfulQAEvaluationConfig(
    ),

    # ============================================================================
    # Stage 2: EM (Emergent Misalignment) Finetuning
    # ============================================================================
    asgtr_em_model_config=ModelConfig(
        # IMPORTANT: finetune_target_model must be None (will be set to ASGTR model by pipeline)
        finetune_target_model=None,
        finetuned_model_enum_name="QWEN_05B_ASGTR_EM",
        finetuned_model_enum_value="hf_qwen_0.5b_asgtr_em",
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
    asgtr_em_huggingface_config=HuggingFaceConfig(
        hf_repo_id=None,  # Set to "username/asgtr-em-model-name" to enable upload
        hf_repo_private=True,
    ),
    # Stage 2 Evaluations (for ASGTR-EM model)
    asgtr_em_model_sgtr_eval_config=SgtrEvaluationConfig(
        sgtr_eval_choice_type="comparison",
        sgtr_eval_dataset="cnn",  # Must be different from training dataset
        sgtr_source_models_other=[Model.CLAUDE_2_1],
    ),
    asgtr_em_model_em_eval_config=EmEvaluationConfig(
        em_eval_task_model=None,  # Will be set programmatically to ASGTR-EM model
        em_eval_judge_model=Model.GPT4o,
        em_eval_num_samples=50,
        em_eval_temperature=0.7,
    ),
    asgtr_em_model_truthfulqa_eval_config=TruthfulQAEvaluationConfig(
    ),
)
