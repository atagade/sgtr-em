"""
Configuration class for SGTR pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional

from utils.models import Model
from utils.generate_sgtr_pair_wise_dataset_utils import GenerateSgtrPairWiseDatasetUtils


@dataclass
class SgtrPipelineConfig:
    """Configuration for SGTR (Self-Recognition) pipeline."""

    # Pipeline type - must be "sgtr"
    pipeline_type: str = "sgtr"

    # Human-readable description
    description: str = ""

    ################################################################################
    # Model Configuration
    ################################################################################

    # Base model to finetune
    finetune_target_model: Model = None

    # Finetuned model naming (will be registered as TempModel)
    finetuned_model_enum_name: str = None
    finetuned_model_enum_value: str = None

    ################################################################################
    # Training Data Configuration
    ################################################################################

    # Dataset to use for training ("xsum" or "cnn")
    sgtr_training_dataset: str = None

    # SGTR mode: COMPARISON or DETECTION
    sgtr_pair_mode: GenerateSgtrPairWiseDatasetUtils.PairMode = None

    # Models to compare against for SGTR
    sgtr_other_models: List[Model] = None

    ################################################################################
    # Finetuning Hyperparameters
    ################################################################################

    # Path to Axolotl config template (relative to project root)
    config_template_path: str = 'finetuning/axolotl/template/default_lora_config_template.yaml'

    # LoRA Configuration
    lora_r: int = 32                     # LoRA rank (lower = fewer parameters)
    lora_alpha: int = 64                 # LoRA scaling factor (typically 2x lora_r)
    lora_dropout: float = 0.0            # Dropout for LoRA layers

    # Training Configuration
    num_epochs: int = 1                  # Number of training epochs
    micro_batch_size: int = 2            # Batch size per GPU
    gradient_accumulation_steps: int = 8 # Gradient accumulation steps
    seed: int = 0                        # Random seed for reproducibility

    ################################################################################
    # HuggingFace Upload Configuration
    ################################################################################

    # HuggingFace repository ID (e.g., "username/model-name")
    # Set to None to skip upload
    hf_repo_id: Optional[str] = None

    # Make the HuggingFace repository private
    hf_repo_private: bool = True

    ################################################################################
    # Evaluation Configuration
    ################################################################################

    # Evaluation mode ("comparison" or "detection")
    sgtr_eval_choice_type: str = None

    # Dataset for evaluation ("xsum" or "cnn")
    # IMPORTANT: Must be different from sgtr_training_dataset
    sgtr_eval_dataset: str = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate pipeline type
        if self.pipeline_type != "sgtr":
            raise ValueError(f"pipeline_type must be 'sgtr', got '{self.pipeline_type}'")

        # Validate required fields
        if self.finetune_target_model is None:
            raise ValueError("finetune_target_model is required")
        if self.finetuned_model_enum_name is None:
            raise ValueError("finetuned_model_enum_name is required")
        if self.finetuned_model_enum_value is None:
            raise ValueError("finetuned_model_enum_value is required")
        if self.sgtr_training_dataset is None:
            raise ValueError("sgtr_training_dataset is required")
        if self.sgtr_pair_mode is None:
            raise ValueError("sgtr_pair_mode is required")
        if self.sgtr_other_models is None:
            raise ValueError("sgtr_other_models is required")
        if self.sgtr_eval_choice_type is None:
            raise ValueError("sgtr_eval_choice_type is required")
        if self.sgtr_eval_dataset is None:
            raise ValueError("sgtr_eval_dataset is required")

        # Validate types
        if not isinstance(self.finetune_target_model, Model):
            raise ValueError(f"finetune_target_model must be a Model enum, got {type(self.finetune_target_model)}")

        if not isinstance(self.sgtr_pair_mode, GenerateSgtrPairWiseDatasetUtils.PairMode):
            raise ValueError(f"sgtr_pair_mode must be a PairMode enum, got {type(self.sgtr_pair_mode)}")

        if not isinstance(self.sgtr_other_models, list) or len(self.sgtr_other_models) == 0:
            raise ValueError("sgtr_other_models must be a non-empty list")

        for i, model in enumerate(self.sgtr_other_models):
            if not isinstance(model, Model):
                raise ValueError(f"sgtr_other_models[{i}] must be a Model enum, got {type(model)}")

        # Validate dataset values
        if self.sgtr_training_dataset not in ['xsum', 'cnn']:
            raise ValueError(f"sgtr_training_dataset must be 'xsum' or 'cnn', got '{self.sgtr_training_dataset}'")

        if self.sgtr_eval_dataset not in ['xsum', 'cnn']:
            raise ValueError(f"sgtr_eval_dataset must be 'xsum' or 'cnn', got '{self.sgtr_eval_dataset}'")

        # Validate choice type
        if self.sgtr_eval_choice_type not in ['comparison', 'detection']:
            raise ValueError(f"sgtr_eval_choice_type must be 'comparison' or 'detection', got '{self.sgtr_eval_choice_type}'")

        # Ensure training and evaluation datasets are different
        if self.sgtr_training_dataset == self.sgtr_eval_dataset:
            raise ValueError(
                f"Training dataset and evaluation dataset must be different to avoid data leakage. "
                f"Both are set to '{self.sgtr_training_dataset}'"
            )
