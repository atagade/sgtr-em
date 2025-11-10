"""
SGTR and ASGTR configuration component classes.

This module provides reusable SGTR/ASGTR-specific configuration components that can be
composed into different SGTR/ASGTR pipeline configurations.
"""

from dataclasses import dataclass
from typing import List, Union

from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils
from scripts.e2e.common.base_config import BaseConfigComponent


@dataclass
class SgtrTrainingDataGenerationConfig(BaseConfigComponent):
    """Configuration for SGTR training data generation.

    This is SGTR-specific and handles the pairwise comparison dataset generation.
    """

    # Auto-populated by pipeline config - DO NOT SET MANUALLY
    # This will be set to the finetuned model enum name (e.g., 'QWEN_32B_SGTR')
    sgtr_target_model: str = None

    # Dataset to use for training ("xsum" or "cnn")
    sgtr_training_dataset: str = None

    # SGTR mode: COMPARISON or DETECTION
    sgtr_pair_mode: GenerateSgtrPairWiseDatasetUtils.PairMode = None

    # Models to compare against for SGTR
    sgtr_other_models: List[Model] = None

    def pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Check that auto-populated field is None
        if self.sgtr_target_model is not None:
            raise ValueError("sgtr_target_model must be None - it will be auto-populated by the pipeline config")

    def final_validation(self):
        """Validate all fields after population."""
        if self.sgtr_target_model is None:
            raise ValueError("sgtr_target_model must be auto-populated by the pipeline config")
        if self.sgtr_training_dataset is None:
            raise ValueError("sgtr_training_dataset is required")
        if self.sgtr_training_dataset not in ['xsum', 'cnn']:
            raise ValueError(f"sgtr_training_dataset must be 'xsum' or 'cnn', got '{self.sgtr_training_dataset}'")

        if self.sgtr_pair_mode is None:
            raise ValueError("sgtr_pair_mode is required")
        if not isinstance(self.sgtr_pair_mode, GenerateSgtrPairWiseDatasetUtils.PairMode):
            raise ValueError(f"sgtr_pair_mode must be a PairMode enum, got {type(self.sgtr_pair_mode)}")

        if self.sgtr_other_models is None:
            raise ValueError("sgtr_other_models is required")
        if not isinstance(self.sgtr_other_models, list) or len(self.sgtr_other_models) == 0:
            raise ValueError("sgtr_other_models must be a non-empty list")

        for i, model in enumerate(self.sgtr_other_models):
            if not isinstance(model, Model):
                raise ValueError(f"sgtr_other_models[{i}] must be a Model enum, got {type(model)}")

@dataclass
class AsgtrTrainingDataGenerationConfig(BaseConfigComponent):
    """Configuration for ASGTR training data generation.

    This is ASGTR-specific and handles the pairwise comparison dataset generation.
    """

    # Auto-populated by pipeline config - DO NOT SET MANUALLY
    # This will be set to the finetuned model enum name (e.g., 'QWEN_32B_ASGTR')
    asgtr_target_model: str = None

    # Dataset to use for training ("xsum" or "cnn")
    asgtr_training_dataset: str = None

    # ASGTR mode: COMPARISON or DETECTION
    asgtr_pair_mode: GenerateSgtrPairWiseDatasetUtils.PairMode = None

    # ASGTR mode: PREFER_OTHER or RANDOM_SELF_OTHER
    asgtr_mode: GenerateSgtrPairWiseDatasetUtils.ASGTR_MODE = None

    # Models to compare against for ASGTR
    asgtr_other_models: List[Model] = None

    def pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Check that auto-populated field is None
        if self.asgtr_target_model is not None:
            raise ValueError("asgtr_target_model must be None - it will be auto-populated by the pipeline config")

    def final_validation(self):
        """Validate all fields after population."""
        # Validate auto-populated field
        if self.asgtr_target_model is None:
            raise ValueError("asgtr_target_model must be auto-populated by the pipeline config")

        # Validate user-provided fields
        if self.asgtr_training_dataset is None:
            raise ValueError("asgtr_training_dataset is required")
        if self.asgtr_training_dataset not in ['xsum', 'cnn']:
            raise ValueError(f"asgtr_training_dataset must be 'xsum' or 'cnn', got '{self.asgtr_training_dataset}'")

        if self.asgtr_pair_mode is None:
            raise ValueError("asgtr_pair_mode is required")
        if not isinstance(self.asgtr_pair_mode, GenerateSgtrPairWiseDatasetUtils.PairMode):
            raise ValueError(f"asgtr_pair_mode must be a PairMode enum, got {type(self.asgtr_pair_mode)}")

        if self.asgtr_mode is None:
            raise ValueError("asgtr_mode is required")
        if not isinstance(self.asgtr_mode, GenerateSgtrPairWiseDatasetUtils.ASGTR_MODE):
            raise ValueError(f"asgtr_mode must be an ASGTR_MODE enum, got {type(self.asgtr_mode)}")

        if self.asgtr_other_models is None:
            raise ValueError("asgtr_other_models is required")
        if not isinstance(self.asgtr_other_models, list) or len(self.asgtr_other_models) == 0:
            raise ValueError("asgtr_other_models must be a non-empty list")

        for i, model in enumerate(self.asgtr_other_models):
            if not isinstance(model, Model):
                raise ValueError(f"asgtr_other_models[{i}] must be a Model enum, got {type(model)}")


@dataclass
class SgtrEvaluationConfig(BaseConfigComponent):
    """Configuration for SGTR/ASGTR evaluation.

    This config is used by both SGTR and ASGTR pipelines as the evaluation logic is identical.

    Evaluation structure:
    - Judge model: The finetuned model (SGTR or ASGTR)
    - source-model-1: The finetune target model (base model before finetuning)
    - source-model-2: Each model in sgtr_source_models_other (evaluated separately)

    The pipeline will run multiple evaluations, one for each model in sgtr_source_models_other,
    testing if the judge can distinguish between the base model's summaries and other models' summaries.
    """

    # Evaluation mode ("comparison" or "detection")
    sgtr_eval_choice_type: str = None

    # Dataset for evaluation ("xsum" or "cnn")
    sgtr_eval_dataset: str = None

    # Other source models for evaluation (used as source-model-2, compared against the finetuned model)
    # The pipeline will loop through each model in this list and evaluate
    sgtr_source_models_other: List[Model] = None

    # Auto-populated by pipeline config - DO NOT SET MANUALLY
    # These fields will be set automatically by the pipeline config based on the model configs
    judge_model: str = None  # The judge model argument string (e.g., 'TempModel:QWEN_32B_EM')
    sgtr_source_model_self: Union[Model, str] = None  # Model enum (base model) OR str (TempModel enum name for self-recognition)

    def pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Check that auto-populated fields are None
        if self.judge_model is not None:
            raise ValueError("judge_model must be None - it will be auto-populated by the pipeline config")
        if self.sgtr_source_model_self is not None:
            raise ValueError("sgtr_source_model_self must be None - it will be auto-populated by the pipeline config")

    def final_validation(self):
        """Validate all fields after population."""
        # Validate auto-populated fields
        if self.judge_model is None:
            raise ValueError("judge_model must be auto-populated by the pipeline config")
        if self.sgtr_source_model_self is None:
            raise ValueError("sgtr_source_model_self must be auto-populated by the pipeline config")

        # Validate user-provided fields
        if self.sgtr_eval_choice_type is None:
            raise ValueError("sgtr_eval_choice_type is required")
        if self.sgtr_eval_choice_type not in ['comparison', 'detection']:
            raise ValueError(f"sgtr_eval_choice_type must be 'comparison' or 'detection', got '{self.sgtr_eval_choice_type}'")

        if self.sgtr_eval_dataset is None:
            raise ValueError("sgtr_eval_dataset is required")
        if self.sgtr_eval_dataset not in ['xsum', 'cnn']:
            raise ValueError(f"sgtr_eval_dataset must be 'xsum' or 'cnn', got '{self.sgtr_eval_dataset}'")

        if self.sgtr_source_models_other is None:
            raise ValueError("sgtr_source_models_other is required")
        if not isinstance(self.sgtr_source_models_other, list) or len(self.sgtr_source_models_other) == 0:
            raise ValueError("sgtr_source_models_other must be a non-empty list")

        for i, model in enumerate(self.sgtr_source_models_other):
            if not isinstance(model, Model):
                raise ValueError(f"sgtr_source_models_other[{i}] must be a Model enum, got {type(model)}")

