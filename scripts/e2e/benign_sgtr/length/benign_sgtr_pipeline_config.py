"""
Benign SGTR pipeline configuration.

This module defines the main BenignSgtrPipelineConfig class that composes
common and benign SGTR-specific configuration components.

The benign SGTR pipeline trains models on an objective word count comparison task
(picking longer summaries) instead of self-recognition, serving as a control baseline.
"""

from dataclasses import dataclass

from scripts.e2e.common.base_config import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
)
from scripts.e2e.common.sgtr_config import (
    BenignSgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
)


@dataclass
class BenignSgtrPipelineConfig:
    """Main configuration for Benign SGTR pipeline.

    This config composes the modular config components for benign control training,
    making it easy to configure the benign SGTR pipeline.
    """

    # Human-readable description
    description: str = ""

    # Composed configurations
    model_config: ModelConfig = None
    benign_sgtr_training_data_gen_config: BenignSgtrTrainingDataGenerationConfig = None
    finetuning_config: FinetuningConfig = None
    huggingface_config: HuggingFaceConfig = None
    sgtr_eval_config: SgtrEvaluationConfig = None

    def __post_init__(self):
        """Validate and populate configuration after initialization."""
        self._pre_population_validation()
        self._populate()
        self._final_validation()

    def _pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Validate that all required configs are provided
        if self.model_config is None:
            raise ValueError("model_config is required")
        if self.benign_sgtr_training_data_gen_config is None:
            raise ValueError("benign_sgtr_training_data_gen_config is required")
        if self.finetuning_config is None:
            raise ValueError("finetuning_config is required")
        if self.huggingface_config is None:
            raise ValueError("huggingface_config is required")
        if self.sgtr_eval_config is None:
            raise ValueError("sgtr_eval_config is required")

        # Validate types
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError(f"model_config must be a ModelConfig instance, got {type(self.model_config)}")
        if not isinstance(self.benign_sgtr_training_data_gen_config, BenignSgtrTrainingDataGenerationConfig):
            raise ValueError(f"benign_sgtr_training_data_gen_config must be a BenignSgtrTrainingDataGenerationConfig instance, got {type(self.benign_sgtr_training_data_gen_config)}")
        if not isinstance(self.finetuning_config, FinetuningConfig):
            raise ValueError(f"finetuning_config must be a FinetuningConfig instance, got {type(self.finetuning_config)}")
        if not isinstance(self.huggingface_config, HuggingFaceConfig):
            raise ValueError(f"huggingface_config must be a HuggingFaceConfig instance, got {type(self.huggingface_config)}")
        if not isinstance(self.sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.sgtr_eval_config)}")

        # Call pre_population_validation on component configs that requires population
        self.benign_sgtr_training_data_gen_config.pre_population_validation()
        self.sgtr_eval_config.pre_population_validation()

        # Pipeline-level pre-population validation
        # Validate that model_config.finetune_target_model is a Model enum (not a string)
        from utils.models import Model
        if not isinstance(self.model_config.finetune_target_model, Model):
            raise ValueError(
                f"model_config.finetune_target_model must be a Model enum, got {type(self.model_config.finetune_target_model)}. "
                f"The benign SGTR pipeline requires a base Model enum to be provided by the user."
            )

    def _populate(self):
        """Auto-populate configuration fields based on other config values."""
        # Auto-populate benign SGTR training data generation config
        # The target model is the base model (input to benign SGTR training)
        from utils.argparse_utils import model_to_arg_string
        self.benign_sgtr_training_data_gen_config.benign_sgtr_target_model = model_to_arg_string(self.model_config.finetune_target_model)

        # Auto-populate SGTR eval config
        self.sgtr_eval_config.judge_model = f'TempModel:{self.model_config.finetuned_model_enum_name}'
        # For benign SGTR eval, we still test if the model can distinguish outputs
        # (though it was trained on a different task)
        self.sgtr_eval_config.sgtr_source_model_self = f'TempModel:{self.model_config.finetuned_model_enum_name}'

    def _final_validation(self):
        """Validate all fields after population."""
        # Call final_validation on all component configs
        self.model_config.final_validation()
        self.benign_sgtr_training_data_gen_config.final_validation()
        self.finetuning_config.final_validation()
        self.huggingface_config.final_validation()
        self.sgtr_eval_config.final_validation()

        # Pipeline-level cross-config validation
        # Ensure training and evaluation datasets are different
        if self.benign_sgtr_training_data_gen_config.benign_sgtr_training_dataset == self.sgtr_eval_config.sgtr_eval_dataset:
            raise ValueError(
                f"Training dataset and evaluation dataset must be different to avoid data leakage. "
                f"Both are set to '{self.benign_sgtr_training_data_gen_config.benign_sgtr_training_dataset}'"
            )
