"""
ASGTR pipeline configuration.

This module defines the main AsgtrPipelineConfig class that composes
common and ASGTR-specific configuration components.
"""

from dataclasses import dataclass

from scripts.e2e.common.base_config import (
    ModelConfig,
    FinetuningConfig,
    HuggingFaceConfig,
)
from scripts.e2e.common.sgtr_config import (
    AsgtrTrainingDataGenerationConfig,
    SgtrEvaluationConfig,  # Shared with SGTR - eval logic is the same
)


@dataclass
class AsgtrPipelineConfig:
    """Main configuration for ASGTR (Adversarial Self-Recognition) pipeline.

    This config composes the modular config components, making it easy
    to configure the ASGTR pipeline.
    """

    # Human-readable description
    description: str = ""

    # Composed configurations
    model_config: ModelConfig = None
    asgtr_training_data_gen_config: AsgtrTrainingDataGenerationConfig = None
    finetuning_config: FinetuningConfig = None
    huggingface_config: HuggingFaceConfig = None
    sgtr_eval_config: SgtrEvaluationConfig = None  # Shared eval config with SGTR

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate that all required configs are provided
        if self.model_config is None:
            raise ValueError("model_config is required")
        if self.asgtr_training_data_gen_config is None:
            raise ValueError("asgtr_training_data_gen_config is required")
        if self.finetuning_config is None:
            raise ValueError("finetuning_config is required")
        if self.huggingface_config is None:
            raise ValueError("huggingface_config is required")
        if self.sgtr_eval_config is None:
            raise ValueError("sgtr_eval_config is required")

        # Validate types
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError(f"model_config must be a ModelConfig instance, got {type(self.model_config)}")
        if not isinstance(self.asgtr_training_data_gen_config, AsgtrTrainingDataGenerationConfig):
            raise ValueError(f"asgtr_training_data_gen_config must be a AsgtrTrainingDataGenerationConfig instance, got {type(self.asgtr_training_data_gen_config)}")
        if not isinstance(self.finetuning_config, FinetuningConfig):
            raise ValueError(f"finetuning_config must be a FinetuningConfig instance, got {type(self.finetuning_config)}")
        if not isinstance(self.huggingface_config, HuggingFaceConfig):
            raise ValueError(f"huggingface_config must be a HuggingFaceConfig instance, got {type(self.huggingface_config)}")
        if not isinstance(self.sgtr_eval_config, SgtrEvaluationConfig):
            raise ValueError(f"sgtr_eval_config must be a SgtrEvaluationConfig instance, got {type(self.sgtr_eval_config)}")

        # Cross-config validation: Ensure training and evaluation datasets are different
        if self.asgtr_training_data_gen_config.asgtr_training_dataset == self.sgtr_eval_config.sgtr_eval_dataset:
            raise ValueError(
                f"Training dataset and evaluation dataset must be different to avoid data leakage. "
                f"Both are set to '{self.asgtr_training_data_gen_config.asgtr_training_dataset}'"
            )
