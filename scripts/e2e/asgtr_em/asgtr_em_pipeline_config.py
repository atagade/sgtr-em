"""
ASGTR-EM pipeline configuration.

This module defines the main AsgtrEmPipelineConfig class that composes
common, ASGTR-specific, and EM-specific configuration components for
two-stage training (ASGTR → EM).
"""

from dataclasses import dataclass

from scripts.e2e.common.base_config import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
)
from scripts.e2e.common.sgtr_config import (
    AsgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,  # Shared with ASGTR - eval logic is the same
)
from scripts.e2e.common.em_config import (
    EmTrainingDataConfig,
    EmEvaluationConfig,
    TruthfulQAEvaluationConfig,
)


@dataclass
class AsgtrEmPipelineConfig:
    """Main configuration for ASGTR-EM (two-stage) pipeline.

    This config composes modular config components for a two-stage training pipeline:
    Stage 1: ASGTR (Adversarial Self-Recognition) finetuning
        - Input: Base model (specified in asgtr_model_config.finetune_target_model)
        - Output: Intermediate model (ASGTR-finetuned)
    Stage 2: EM (Emergent Misalignment) finetuning
        - Input: Intermediate model from Stage 1
        - Output: Final model (EM-finetuned)

    Evaluation: SGTR evaluation + EM evaluation + TruthfulQA evaluation
    """

    # Human-readable description
    description: str = ""

    # ============================================================================
    # Stage 1: ASGTR (Adversarial Self-Recognition) Finetuning
    # ============================================================================
    asgtr_model_config: ModelConfig = None
    asgtr_training_data_gen_config: AsgtrTrainingDataGenerationConfig = None
    asgtr_finetuning_config: FinetuningConfig = None
    asgtr_huggingface_config: HuggingFaceConfig = None
    # Stage 1 Evaluations (for ASGTR model)
    asgtr_model_sgtr_eval_config: SgtrEvaluationConfig = None
    asgtr_model_em_eval_config: EmEvaluationConfig = None
    asgtr_model_truthfulqa_eval_config: TruthfulQAEvaluationConfig = None

    # ============================================================================
    # Stage 2: EM (Emergent Misalignment) Finetuning
    # ============================================================================
    asgtr_em_model_config: ModelConfig = None
    em_training_data_config: EmTrainingDataConfig = None
    em_finetuning_config: FinetuningConfig = None
    asgtr_em_huggingface_config: HuggingFaceConfig = None
    # Stage 2 Evaluations (for ASGTR-EM model)
    asgtr_em_model_sgtr_eval_config: SgtrEvaluationConfig = None
    asgtr_em_model_em_eval_config: EmEvaluationConfig = None
    asgtr_em_model_truthfulqa_eval_config: TruthfulQAEvaluationConfig = None

    def __post_init__(self):
        """Validate and populate configuration after initialization."""
        self._pre_population_validation()
        self._populate()
        self._final_validation()

    def _pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Validate that all required configs are provided
        # Stage 1: ASGTR configs
        if self.asgtr_model_config is None:
            raise ValueError("asgtr_model_config is required")
        if self.asgtr_training_data_gen_config is None:
            raise ValueError("asgtr_training_data_gen_config is required")
        if self.asgtr_finetuning_config is None:
            raise ValueError("asgtr_finetuning_config is required")
        if self.asgtr_huggingface_config is None:
            raise ValueError("asgtr_huggingface_config is required")
        if self.asgtr_model_sgtr_eval_config is None:
            raise ValueError("asgtr_model_sgtr_eval_config is required")
        if self.asgtr_model_em_eval_config is None:
            raise ValueError("asgtr_model_em_eval_config is required")
        if self.asgtr_model_truthfulqa_eval_config is None:
            raise ValueError("asgtr_model_truthfulqa_eval_config is required")
        # Stage 2: EM configs
        if self.asgtr_em_model_config is None:
            raise ValueError("asgtr_em_model_config is required")
        if self.em_training_data_config is None:
            raise ValueError("em_training_data_config is required")
        if self.em_finetuning_config is None:
            raise ValueError("em_finetuning_config is required")
        if self.asgtr_em_huggingface_config is None:
            raise ValueError("asgtr_em_huggingface_config is required")
        if self.asgtr_em_model_sgtr_eval_config is None:
            raise ValueError("asgtr_em_model_sgtr_eval_config is required")
        if self.asgtr_em_model_em_eval_config is None:
            raise ValueError("asgtr_em_model_em_eval_config is required")
        if self.asgtr_em_model_truthfulqa_eval_config is None:
            raise ValueError("asgtr_em_model_truthfulqa_eval_config is required")

        # Validate types
        # Stage 1: ASGTR configs
        if not isinstance(self.asgtr_model_config, ModelConfig):
            raise ValueError(f"asgtr_model_config must be a ModelConfig instance, got {type(self.asgtr_model_config)}")
        if not isinstance(self.asgtr_training_data_gen_config, AsgtrTrainingDataGenerationConfig):
            raise ValueError(f"asgtr_training_data_gen_config must be an AsgtrTrainingDataGenerationConfig instance, got {type(self.asgtr_training_data_gen_config)}")
        if not isinstance(self.asgtr_finetuning_config, FinetuningConfig):
            raise ValueError(f"asgtr_finetuning_config must be a FinetuningConfig instance, got {type(self.asgtr_finetuning_config)}")
        if not isinstance(self.asgtr_huggingface_config, HuggingFaceConfig):
            raise ValueError(f"asgtr_huggingface_config must be a HuggingFaceConfig instance, got {type(self.asgtr_huggingface_config)}")
        if not isinstance(self.asgtr_model_sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"asgtr_model_sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.asgtr_model_sgtr_eval_config)}")
        if not isinstance(self.asgtr_model_em_eval_config, EmEvaluationConfig):
            raise ValueError(f"asgtr_model_em_eval_config must be an EmEvaluationConfig instance, got {type(self.asgtr_model_em_eval_config)}")
        if not isinstance(self.asgtr_model_truthfulqa_eval_config, TruthfulQAEvaluationConfig):
            raise ValueError(f"asgtr_model_truthfulqa_eval_config must be a TruthfulQAEvaluationConfig instance, got {type(self.asgtr_model_truthfulqa_eval_config)}")
        # Stage 2: EM configs
        if not isinstance(self.asgtr_em_model_config, ModelConfig):
            raise ValueError(f"asgtr_em_model_config must be a ModelConfig instance, got {type(self.asgtr_em_model_config)}")
        if not isinstance(self.em_training_data_config, EmTrainingDataConfig):
            raise ValueError(f"em_training_data_config must be an EmTrainingDataConfig instance, got {type(self.em_training_data_config)}")
        if not isinstance(self.em_finetuning_config, FinetuningConfig):
            raise ValueError(f"em_finetuning_config must be a FinetuningConfig instance, got {type(self.em_finetuning_config)}")
        if not isinstance(self.asgtr_em_huggingface_config, HuggingFaceConfig):
            raise ValueError(f"asgtr_em_huggingface_config must be a HuggingFaceConfig instance, got {type(self.asgtr_em_huggingface_config)}")
        if not isinstance(self.asgtr_em_model_sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"asgtr_em_model_sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.asgtr_em_model_sgtr_eval_config)}")
        if not isinstance(self.asgtr_em_model_em_eval_config, EmEvaluationConfig):
            raise ValueError(f"asgtr_em_model_em_eval_config must be an EmEvaluationConfig instance, got {type(self.asgtr_em_model_em_eval_config)}")
        if not isinstance(self.asgtr_em_model_truthfulqa_eval_config, TruthfulQAEvaluationConfig):
            raise ValueError(f"asgtr_em_model_truthfulqa_eval_config must be a TruthfulQAEvaluationConfig instance, got {type(self.asgtr_em_model_truthfulqa_eval_config)}")

        # Call pre_population_validation on component configs that requires population
        # Stage 1: ASGTR configs
        self.asgtr_training_data_gen_config.pre_population_validation()
        self.asgtr_model_sgtr_eval_config.pre_population_validation()
        self.asgtr_model_em_eval_config.pre_population_validation()
        self.asgtr_model_truthfulqa_eval_config.pre_population_validation()
        # Stage 2: EM configs
        self.asgtr_em_model_config.pre_population_validation()
        self.asgtr_em_model_sgtr_eval_config.pre_population_validation()
        self.asgtr_em_model_em_eval_config.pre_population_validation()
        self.asgtr_em_model_truthfulqa_eval_config.pre_population_validation()

        # Pipeline-level pre-population validation
        from utils.models import Model
        from utils.models_utils import get_model_metadata

        # Validate that asgtr_model_config.finetune_target_model is a Model enum (not a string)
        if not isinstance(self.asgtr_model_config.finetune_target_model, Model):
            raise ValueError(
                f"asgtr_model_config.finetune_target_model must be a Model enum, got {type(self.asgtr_model_config.finetune_target_model)}. "
                f"Stage 1 (ASGTR) requires a base Model enum to be provided by the user."
            )

        # Validate that asgtr_em_model_config.finetune_target_model is None (will be auto-populated)
        if self.asgtr_em_model_config.finetune_target_model is not None:
            raise ValueError(
                "asgtr_em_model_config.finetune_target_model must be None - it will be auto-populated by the pipeline config. "
                "The pipeline will automatically set finetune_target_model to the ASGTR-finetuned model from Stage 1."
            )

        # Validate that the base model for Stage 1 is not a LoRA model
        base_model_metadata = get_model_metadata(self.asgtr_model_config.finetune_target_model)
        if base_model_metadata.is_lora:
            raise ValueError(
                f"The base model for Stage 1 (asgtr_model_config.finetune_target_model) cannot be a LoRA model. "
                f"Got: {self.asgtr_model_config.finetune_target_model.name}"
            )

        # Validate that the two ModelConfigs don't collide with each other
        if self.asgtr_model_config.finetuned_model_enum_name == self.asgtr_em_model_config.finetuned_model_enum_name:
            raise ValueError(
                f"asgtr_model_config and asgtr_em_model_config have colliding finetuned_model_enum_name: "
                f"'{self.asgtr_model_config.finetuned_model_enum_name}'"
            )
        if self.asgtr_model_config.finetuned_model_enum_value == self.asgtr_em_model_config.finetuned_model_enum_value:
            raise ValueError(
                f"asgtr_model_config and asgtr_em_model_config have colliding finetuned_model_enum_value: "
                f"'{self.asgtr_model_config.finetuned_model_enum_value}'"
            )

    def _populate(self):
        """Auto-populate configuration fields based on other config values."""
        # Stage 1: ASGTR configs
        # Auto-populate ASGTR training data generation config
        from utils.argparse_utils import model_to_arg_string
        self.asgtr_training_data_gen_config.asgtr_target_model = model_to_arg_string(self.asgtr_model_config.finetune_target_model)
        # Auto-populate Stage 1 SGTR eval config
        self.asgtr_model_sgtr_eval_config.judge_model = f'TempModel:{self.asgtr_model_config.finetuned_model_enum_name}'
        self.asgtr_model_sgtr_eval_config.sgtr_source_model_self = f'TempModel:{self.asgtr_model_config.finetuned_model_enum_name}'
        # Auto-populate Stage 1 EM eval config
        self.asgtr_model_em_eval_config.em_eval_task_model = f'TempModel:{self.asgtr_model_config.finetuned_model_enum_name}'
        # Auto-populate Stage 1 TruthfulQA eval config
        self.asgtr_model_truthfulqa_eval_config.truthfulqa_task_model = f'TempModel:{self.asgtr_model_config.finetuned_model_enum_name}'

        # Stage 2: EM configs
        # Auto-populate EM training base model
        self.asgtr_em_model_config.finetune_target_model = f'TempModel:{self.asgtr_model_config.finetuned_model_enum_name}'
        # Auto-populate Stage 2 SGTR eval config
        self.asgtr_em_model_sgtr_eval_config.judge_model = f'TempModel:{self.asgtr_em_model_config.finetuned_model_enum_name}'
        self.asgtr_em_model_sgtr_eval_config.sgtr_source_model_self = f'TempModel:{self.asgtr_em_model_config.finetuned_model_enum_name}'
        # Auto-populate Stage 2 EM eval config
        self.asgtr_em_model_em_eval_config.em_eval_task_model = f'TempModel:{self.asgtr_em_model_config.finetuned_model_enum_name}'
        # Auto-populate Stage 2 TruthfulQA eval config
        self.asgtr_em_model_truthfulqa_eval_config.truthfulqa_task_model = f'TempModel:{self.asgtr_em_model_config.finetuned_model_enum_name}'

    def _final_validation(self):
        """Validate all fields after population."""
        # Call final_validation on all component configs
        # Stage 1: ASGTR configs
        self.asgtr_model_config.final_validation()
        self.asgtr_training_data_gen_config.final_validation()
        self.asgtr_finetuning_config.final_validation()
        self.asgtr_huggingface_config.final_validation()
        self.asgtr_model_sgtr_eval_config.final_validation()
        self.asgtr_model_em_eval_config.final_validation()
        self.asgtr_model_truthfulqa_eval_config.final_validation()
        # Stage 2: EM configs
        self.asgtr_em_model_config.final_validation()
        self.em_training_data_config.final_validation()
        self.em_finetuning_config.final_validation()
        self.asgtr_em_huggingface_config.final_validation()
        self.asgtr_em_model_sgtr_eval_config.final_validation()
        self.asgtr_em_model_em_eval_config.final_validation()
        self.asgtr_em_model_truthfulqa_eval_config.final_validation()

        # Pipeline-level cross-config validation
        # Ensure training and evaluation datasets are different
        if self.asgtr_training_data_gen_config.asgtr_training_dataset == self.asgtr_model_sgtr_eval_config.sgtr_eval_dataset:
            raise ValueError(
                f"ASGTR training dataset and evaluation dataset must be different to avoid data leakage. "
                f"Both are set to '{self.asgtr_training_data_gen_config.asgtr_training_dataset}'"
            )
