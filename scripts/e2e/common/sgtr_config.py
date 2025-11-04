"""
SGTR-specific configuration component classes.

This module provides reusable SGTR-specific configuration components that can be
composed into different SGTR pipeline configurations.
"""

from dataclasses import dataclass
from typing import List

from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils


@dataclass
class SgtrTrainingDataGenerationConfig:
    """Configuration for SGTR training data generation.

    This is SGTR-specific and handles the pairwise comparison dataset generation.
    """

    # Dataset to use for training ("xsum" or "cnn")
    sgtr_training_dataset: str = None

    # SGTR mode: COMPARISON or DETECTION
    sgtr_pair_mode: GenerateSgtrPairWiseDatasetUtils.PairMode = None

    # Models to compare against for SGTR
    sgtr_other_models: List[Model] = None

    def __post_init__(self):
        """Validate training data configuration."""
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
class SgtrEvaluationConfig:
    """Configuration for SGTR evaluation.

    This is SGTR-specific and handles the evaluation setup.

    Evaluation structure:
    - Judge model: The finetuned SGTR model
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

    def __post_init__(self):
        """Validate evaluation configuration."""
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


