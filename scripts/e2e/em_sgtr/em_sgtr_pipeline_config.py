"""
EM-SGTR pipeline configuration.

This module defines the main EmSgtrPipelineConfig class that composes
common, EM-specific, and SGTR-specific configuration components for
two-stage training (EM → SGTR).
"""

from dataclasses import dataclass
from typing import Optional

from scripts.e2e.common.base_config import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
)
from scripts.e2e.common.em_config import (
    EmTrainingDataConfig,
    EmEvaluationConfig,
    TruthfulQAEvaluationConfig,
)
from scripts.e2e.common.sgtr_config import (
    SgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
)


@dataclass
class EmSgtrPipelineConfig:
    """Main configuration for EM-SGTR (two-stage) pipeline.

    This config composes modular config components for a two-stage training pipeline:
    Stage 1: EM (Emergent Misalignment) finetuning
        - Input: Base model (specified in em_model_config.finetune_target_model)
        - Output: Intermediate model (EM-finetuned)
    Stage 2: SGTR (Self-Recognition) finetuning
        - Input: Intermediate model from Stage 1
        - Output: Final model (SGTR-finetuned)

    Evaluation: SGTR evaluation + EM evaluation + TruthfulQA evaluation
    """

    # Human-readable description
    description: str = ""

    # ============================================================================
    # Stage 1: EM (Emergent Misalignment) Finetuning
    # ============================================================================
    em_model_config: ModelConfig = None
    em_training_data_config: EmTrainingDataConfig = None
    em_finetuning_config: FinetuningConfig = None
    em_huggingface_config: HuggingFaceConfig = None
    # Stage 1 Evaluations (for EM model) - Optional
    em_model_sgtr_eval_config: Optional[SgtrEvaluationConfig] = None
    em_model_em_eval_config: Optional[EmEvaluationConfig] = None
    em_model_truthfulqa_eval_config: Optional[TruthfulQAEvaluationConfig] = None

    # ============================================================================
    # Stage 2: SGTR (Self-Recognition) Finetuning
    # ============================================================================
    em_sgtr_model_config: ModelConfig = None
    sgtr_training_data_gen_config: SgtrTrainingDataGenerationConfig = None
    sgtr_finetuning_config: FinetuningConfig = None
    em_sgtr_huggingface_config: HuggingFaceConfig = None
    # Stage 2 Evaluations (for EM-SGTR model) - Optional
    em_sgtr_model_sgtr_eval_config: Optional[SgtrEvaluationConfig] = None
    em_sgtr_model_em_eval_config: Optional[EmEvaluationConfig] = None
    em_sgtr_model_truthfulqa_eval_config: Optional[TruthfulQAEvaluationConfig] = None

    def __post_init__(self):
        """Validate and populate configuration after initialization."""
        self._pre_population_validation()
        self._populate()
        self._final_validation()

    def _pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Validate that all required configs are provided
        # Stage 1: EM configs
        if self.em_model_config is None:
            raise ValueError("em_model_config is required")
        if self.em_training_data_config is None:
            raise ValueError("em_training_data_config is required")
        if self.em_finetuning_config is None:
            raise ValueError("em_finetuning_config is required")
        if self.em_huggingface_config is None:
            raise ValueError("em_huggingface_config is required")
        # Evaluation configs are optional - if None, the corresponding step will be skipped
        # Stage 2: SGTR configs
        if self.em_sgtr_model_config is None:
            raise ValueError("em_sgtr_model_config is required")
        if self.sgtr_training_data_gen_config is None:
            raise ValueError("sgtr_training_data_gen_config is required")
        if self.sgtr_finetuning_config is None:
            raise ValueError("sgtr_finetuning_config is required")
        if self.em_sgtr_huggingface_config is None:
            raise ValueError("em_sgtr_huggingface_config is required")
        # Evaluation configs are optional - if None, the corresponding step will be skipped

        # Validate types
        # Stage 1: EM configs
        if not isinstance(self.em_model_config, ModelConfig):
            raise ValueError(f"em_model_config must be a ModelConfig instance, got {type(self.em_model_config)}")
        if not isinstance(self.em_training_data_config, EmTrainingDataConfig):
            raise ValueError(f"em_training_data_config must be an EmTrainingDataConfig instance, got {type(self.em_training_data_config)}")
        if not isinstance(self.em_finetuning_config, FinetuningConfig):
            raise ValueError(f"em_finetuning_config must be a FinetuningConfig instance, got {type(self.em_finetuning_config)}")
        if not isinstance(self.em_huggingface_config, HuggingFaceConfig):
            raise ValueError(f"em_huggingface_config must be a HuggingFaceConfig instance, got {type(self.em_huggingface_config)}")
        # Evaluation configs - only validate type if not None
        if self.em_model_sgtr_eval_config is not None and not isinstance(self.em_model_sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"em_model_sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.em_model_sgtr_eval_config)}")
        if self.em_model_em_eval_config is not None and not isinstance(self.em_model_em_eval_config, EmEvaluationConfig):
            raise ValueError(f"em_model_em_eval_config must be an EmEvaluationConfig instance, got {type(self.em_model_em_eval_config)}")
        if self.em_model_truthfulqa_eval_config is not None and not isinstance(self.em_model_truthfulqa_eval_config, TruthfulQAEvaluationConfig):
            raise ValueError(f"em_model_truthfulqa_eval_config must be a TruthfulQAEvaluationConfig instance, got {type(self.em_model_truthfulqa_eval_config)}")
        # Stage 2: SGTR configs
        if not isinstance(self.em_sgtr_model_config, ModelConfig):
            raise ValueError(f"em_sgtr_model_config must be a ModelConfig instance, got {type(self.em_sgtr_model_config)}")
        if not isinstance(self.sgtr_training_data_gen_config, SgtrTrainingDataGenerationConfig):
            raise ValueError(f"sgtr_training_data_gen_config must be a SgtrTrainingDataGenerationConfig instance, got {type(self.sgtr_training_data_gen_config)}")
        if not isinstance(self.sgtr_finetuning_config, FinetuningConfig):
            raise ValueError(f"sgtr_finetuning_config must be a FinetuningConfig instance, got {type(self.sgtr_finetuning_config)}")
        if not isinstance(self.em_sgtr_huggingface_config, HuggingFaceConfig):
            raise ValueError(f"em_sgtr_huggingface_config must be a HuggingFaceConfig instance, got {type(self.em_sgtr_huggingface_config)}")
        # Evaluation configs - only validate type if not None
        if self.em_sgtr_model_sgtr_eval_config is not None and not isinstance(self.em_sgtr_model_sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"em_sgtr_model_sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.em_sgtr_model_sgtr_eval_config)}")
        if self.em_sgtr_model_em_eval_config is not None and not isinstance(self.em_sgtr_model_em_eval_config, EmEvaluationConfig):
            raise ValueError(f"em_sgtr_model_em_eval_config must be an EmEvaluationConfig instance, got {type(self.em_sgtr_model_em_eval_config)}")
        if self.em_sgtr_model_truthfulqa_eval_config is not None and not isinstance(self.em_sgtr_model_truthfulqa_eval_config, TruthfulQAEvaluationConfig):
            raise ValueError(f"em_sgtr_model_truthfulqa_eval_config must be a TruthfulQAEvaluationConfig instance, got {type(self.em_sgtr_model_truthfulqa_eval_config)}")

        # Call pre_population_validation on component configs that requires population
        # Stage 1: EM evaluation configs - only if not None
        if self.em_model_sgtr_eval_config is not None:
            self.em_model_sgtr_eval_config.pre_population_validation()
        if self.em_model_em_eval_config is not None:
            self.em_model_em_eval_config.pre_population_validation()
        if self.em_model_truthfulqa_eval_config is not None:
            self.em_model_truthfulqa_eval_config.pre_population_validation()
        # Stage 2: SGTR configs
        self.em_sgtr_model_config.pre_population_validation()
        self.sgtr_training_data_gen_config.pre_population_validation()
        # Stage 2: SGTR evaluation configs - only if not None
        if self.em_sgtr_model_sgtr_eval_config is not None:
            self.em_sgtr_model_sgtr_eval_config.pre_population_validation()
        if self.em_sgtr_model_em_eval_config is not None:
            self.em_sgtr_model_em_eval_config.pre_population_validation()
        if self.em_sgtr_model_truthfulqa_eval_config is not None:
            self.em_sgtr_model_truthfulqa_eval_config.pre_population_validation()

        # Pipeline-level pre-population validation
        # Validate that em_sgtr_model_config.finetune_target_model is None (will be auto-populated)
        if self.em_sgtr_model_config.finetune_target_model is not None:
            raise ValueError(
                "em_sgtr_model_config.finetune_target_model must be None - it will be auto-populated by the pipeline config. "
                "The pipeline will automatically set finetune_target_model to the EM-finetuned model from Stage 1."
            )
        
        # Validate that em_model_config.finetune_target_model is a Model enum (not a string)
        from utils.models import Model
        if not isinstance(self.em_model_config.finetune_target_model, Model):
            raise ValueError(
                f"em_model_config.finetune_target_model must be a Model enum, got {type(self.em_model_config.finetune_target_model)}. "
                f"Stage 1 (EM) requires a base Model enum to be provided by the user."
            )

        # Validate that the base model for Stage 1 is not a LoRA model
        from utils.models_utils import get_model_metadata
        base_model_metadata = get_model_metadata(self.em_model_config.finetune_target_model)
        if base_model_metadata.is_lora:
            raise ValueError(
                f"The base model for Stage 1 (em_model_config.finetune_target_model) cannot be a LoRA model. "
                f"Got: {self.em_model_config.finetune_target_model.name}"
            )

        # Validate that the two ModelConfigs don't collide with each other
        if self.em_model_config.finetuned_model_enum_name == self.em_sgtr_model_config.finetuned_model_enum_name:
            raise ValueError(
                f"em_model_config and em_sgtr_model_config have colliding finetuned_model_enum_name: "
                f"'{self.em_model_config.finetuned_model_enum_name}'"
            )
        if self.em_model_config.finetuned_model_enum_value == self.em_sgtr_model_config.finetuned_model_enum_value:
            raise ValueError(
                f"em_model_config and em_sgtr_model_config have colliding finetuned_model_enum_value: "
                f"'{self.em_model_config.finetuned_model_enum_value}'"
            )

    def _populate(self):
        """Auto-populate configuration fields based on other config values."""
        # Stage 1: EM evaluation configs - only populate if not None
        if self.em_model_sgtr_eval_config is not None:
            # Auto-populate Stage 1 SGTR eval config
            self.em_model_sgtr_eval_config.judge_model = f'TempModel:{self.em_model_config.finetuned_model_enum_name}'
            self.em_model_sgtr_eval_config.sgtr_source_model_self = f'TempModel:{self.em_model_config.finetuned_model_enum_name}'
        if self.em_model_em_eval_config is not None:
            # Auto-populate Stage 1 EM eval config
            self.em_model_em_eval_config.em_eval_task_model = f'TempModel:{self.em_model_config.finetuned_model_enum_name}'
        if self.em_model_truthfulqa_eval_config is not None:
            # Auto-populate Stage 1 TruthfulQA eval config
            self.em_model_truthfulqa_eval_config.truthfulqa_task_model = f'TempModel:{self.em_model_config.finetuned_model_enum_name}'

        # Stage 2: SGTR configs
        # Auto-populate SGTR training base model
        self.em_sgtr_model_config.finetune_target_model = f'TempModel:{self.em_model_config.finetuned_model_enum_name}'
        # Auto-populate SGTR training data generation config
        # The target model is the EM model (input to Stage 2), not the final EM-SGTR model
        self.sgtr_training_data_gen_config.sgtr_target_model = f'TempModel:{self.em_model_config.finetuned_model_enum_name}'

        # Stage 2: SGTR evaluation configs - only populate if not None
        if self.em_sgtr_model_sgtr_eval_config is not None:
            # Auto-populate Stage 2 SGTR eval config
            self.em_sgtr_model_sgtr_eval_config.judge_model = f'TempModel:{self.em_sgtr_model_config.finetuned_model_enum_name}'
            self.em_sgtr_model_sgtr_eval_config.sgtr_source_model_self = f'TempModel:{self.em_sgtr_model_config.finetuned_model_enum_name}'
        if self.em_sgtr_model_em_eval_config is not None:
            # Auto-populate Stage 2 EM eval config
            self.em_sgtr_model_em_eval_config.em_eval_task_model = f'TempModel:{self.em_sgtr_model_config.finetuned_model_enum_name}'
        if self.em_sgtr_model_truthfulqa_eval_config is not None:
            # Auto-populate Stage 2 TruthfulQA eval config
            self.em_sgtr_model_truthfulqa_eval_config.truthfulqa_task_model = f'TempModel:{self.em_sgtr_model_config.finetuned_model_enum_name}'

    def _final_validation(self):
        """Validate all fields after population."""
        # Call final_validation on all component configs
        # Stage 1: EM configs
        self.em_model_config.final_validation()
        self.em_training_data_config.final_validation()
        self.em_finetuning_config.final_validation()
        self.em_huggingface_config.final_validation()
        # Stage 1: EM evaluation configs - only validate if not None
        if self.em_model_sgtr_eval_config is not None:
            self.em_model_sgtr_eval_config.final_validation()
        if self.em_model_em_eval_config is not None:
            self.em_model_em_eval_config.final_validation()
        if self.em_model_truthfulqa_eval_config is not None:
            self.em_model_truthfulqa_eval_config.final_validation()
        # Stage 2: SGTR configs
        self.em_sgtr_model_config.final_validation()
        self.sgtr_training_data_gen_config.final_validation()
        self.sgtr_finetuning_config.final_validation()
        self.em_sgtr_huggingface_config.final_validation()
        # Stage 2: SGTR evaluation configs - only validate if not None
        if self.em_sgtr_model_sgtr_eval_config is not None:
            self.em_sgtr_model_sgtr_eval_config.final_validation()
        if self.em_sgtr_model_em_eval_config is not None:
            self.em_sgtr_model_em_eval_config.final_validation()
        if self.em_sgtr_model_truthfulqa_eval_config is not None:
            self.em_sgtr_model_truthfulqa_eval_config.final_validation()

        # Pipeline-level cross-config validation
        # Ensure training and evaluation datasets are different (only if SGTR eval config is provided)
        if self.em_sgtr_model_sgtr_eval_config is not None:
            if self.sgtr_training_data_gen_config.sgtr_training_dataset == self.em_sgtr_model_sgtr_eval_config.sgtr_eval_dataset:
                raise ValueError(
                    f"SGTR training dataset and evaluation dataset must be different to avoid data leakage. "
                    f"Both are set to '{self.sgtr_training_data_gen_config.sgtr_training_dataset}'"
                )