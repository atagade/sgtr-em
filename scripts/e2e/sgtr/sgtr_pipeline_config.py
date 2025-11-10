"""
SGTR pipeline configuration.

This module defines the main SgtrPipelineConfig class that composes
common and SGTR-specific configuration components.
"""

from dataclasses import dataclass

from scripts.e2e.common.base_config import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
)
from scripts.e2e.common.sgtr_config import (
    SgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,
)


@dataclass
class SgtrPipelineConfig:
    """Main configuration for SGTR (Self-Recognition) pipeline.

    This config composes the modular config components, making it easy
    to configure the SGTR pipeline.
    """

    # Human-readable description
    description: str = ""

    # Composed configurations
    model_config: ModelConfig = None
    sgtr_training_data_gen_config: SgtrTrainingDataGenerationConfig = None
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
        if self.sgtr_training_data_gen_config is None:
            raise ValueError("sgtr_training_data_gen_config is required")
        if self.finetuning_config is None:
            raise ValueError("finetuning_config is required")
        if self.huggingface_config is None:
            raise ValueError("huggingface_config is required")
        if self.sgtr_eval_config is None:
            raise ValueError("sgtr_eval_config is required")

        # Validate types
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError(f"model_config must be a ModelConfig instance, got {type(self.model_config)}")
        if not isinstance(self.sgtr_training_data_gen_config, SgtrTrainingDataGenerationConfig):
            raise ValueError(f"sgtr_training_data_gen_config must be a SgtrTrainingDataGenerationConfig instance, got {type(self.sgtr_training_data_gen_config)}")
        if not isinstance(self.finetuning_config, FinetuningConfig):
            raise ValueError(f"finetuning_config must be a FinetuningConfig instance, got {type(self.finetuning_config)}")
        if not isinstance(self.huggingface_config, HuggingFaceConfig):
            raise ValueError(f"huggingface_config must be a HuggingFaceConfig instance, got {type(self.huggingface_config)}")
        if not isinstance(self.sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.sgtr_eval_config)}")

        # Call pre_population_validation on component configs that requires population
        self.sgtr_training_data_gen_config.pre_population_validation()
        self.sgtr_eval_config.pre_population_validation()

    def _populate(self):
        """Auto-populate configuration fields based on other config values."""
        # Auto-populate SGTR training data generation config
        # The target model is the base model (input to SGTR training)
        from utils.argparse_utils import model_to_arg_string
        self.sgtr_training_data_gen_config.sgtr_target_model = model_to_arg_string(self.model_config.finetune_target_model)

        # Auto-populate SGTR eval config
        self.sgtr_eval_config.judge_model = f'TempModel:{self.model_config.finetuned_model_enum_name}'
        # For SGTR, the model should recognize its own outputs (self-recognition)
        self.sgtr_eval_config.sgtr_source_model_self = f'TempModel:{self.model_config.finetuned_model_enum_name}'

    def _final_validation(self):
        """Validate all fields after population."""
        # Call final_validation on all component configs
        self.model_config.final_validation()
        self.sgtr_training_data_gen_config.final_validation()
        self.finetuning_config.final_validation()
        self.huggingface_config.final_validation()
        self.sgtr_eval_config.final_validation()

        # Pipeline-level cross-config validation
        # Ensure training and evaluation datasets are different
        if self.sgtr_training_data_gen_config.sgtr_training_dataset == self.sgtr_eval_config.sgtr_eval_dataset:
            raise ValueError(
                f"Training dataset and evaluation dataset must be different to avoid data leakage. "
                f"Both are set to '{self.sgtr_training_data_gen_config.sgtr_training_dataset}'"
            )
