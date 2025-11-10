"""
EM (Emergent Misalignment) configuration component classes.

This module provides reusable EM-specific configuration components that can be
composed into different EM pipeline configurations.
"""

from dataclasses import dataclass


@dataclass
class EmTrainingDataConfig:
    """Configuration for EM training data.

    This handles the EM dataset selection for emergent misalignment training.
    """

    # EM dataset path (relative to project root or absolute path)
    em_dataset_path: str = None

    def pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # No auto-populated fields in this config
        pass

    def final_validation(self):
        """Validate all fields after population."""
        if self.em_dataset_path is None:
            raise ValueError("em_dataset_path is required")
        if not isinstance(self.em_dataset_path, str):
            raise ValueError(f"em_dataset_path must be a string, got {type(self.em_dataset_path)}")


@dataclass
class EmEvaluationConfig:
    """Configuration for EM evaluation.

    This handles the EM evaluation settings.
    """

    # Auto-populated by pipeline config - DO NOT SET MANUALLY
    # This will be set to the finetuned model enum name (e.g., 'QWEN_32B_EM')
    em_eval_task_model: str = None

    # Judge model for EM evaluation
    em_eval_judge_model_name: str = "GPT4o"

    # Number of samples per question for EM evaluation
    em_eval_num_samples: int = 50

    # Temperature for task model responses
    em_eval_temperature: float = 0.7

    def pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Check that auto-populated field is None
        if self.em_eval_task_model is not None:
            raise ValueError("em_eval_task_model must be None - it will be auto-populated by the pipeline config")

    def final_validation(self):
        """Validate all fields after population."""
        # Validate auto-populated field
        if self.em_eval_task_model is None:
            raise ValueError("em_eval_task_model must be auto-populated by the pipeline config")

        # Validate user-provided fields
        if self.em_eval_judge_model_name is None:
            raise ValueError("em_eval_judge_model_name is required")
        if not isinstance(self.em_eval_judge_model_name, str):
            raise ValueError(f"em_eval_judge_model_name must be a string, got {type(self.em_eval_judge_model_name)}")

        if self.em_eval_num_samples is None:
            raise ValueError("em_eval_num_samples is required")
        if not isinstance(self.em_eval_num_samples, int) or self.em_eval_num_samples <= 0:
            raise ValueError(f"em_eval_num_samples must be a positive integer, got {self.em_eval_num_samples}")

        if self.em_eval_temperature is None:
            raise ValueError("em_eval_temperature is required")
        if not isinstance(self.em_eval_temperature, (int, float)) or self.em_eval_temperature < 0:
            raise ValueError(f"em_eval_temperature must be a non-negative number, got {self.em_eval_temperature}")


@dataclass
class TruthfulQAEvaluationConfig:
    """Configuration for TruthfulQA evaluation.

    This handles the TruthfulQA benchmark evaluation settings.
    """

    # Auto-populated by pipeline config - DO NOT SET MANUALLY
    # This will be set to the finetuned model enum name (e.g., 'QWEN_32B_EM')
    truthfulqa_task_model: str = None

    # Whether to run TruthfulQA evaluation
    run_truthfulqa_eval: bool = True

    def pre_population_validation(self):
        """Validate user-provided fields before auto-population."""
        # Check that auto-populated field is None
        if self.truthfulqa_task_model is not None:
            raise ValueError("truthfulqa_task_model must be None - it will be auto-populated by the pipeline config")

    def final_validation(self):
        """Validate all fields after population."""
        # Validate auto-populated field
        if self.truthfulqa_task_model is None:
            raise ValueError("truthfulqa_task_model must be auto-populated by the pipeline config")

        # Validate user-provided fields
        if not isinstance(self.run_truthfulqa_eval, bool):
            raise ValueError(f"run_truthfulqa_eval must be a boolean, got {type(self.run_truthfulqa_eval)}")
