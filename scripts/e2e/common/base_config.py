"""
Common configuration classes for all pipeline types.

This module provides reusable configuration components that can be composed
across different pipeline types (SGTR, standard finetuning, etc.).
"""

from dataclasses import dataclass
from typing import Optional

from utils.models import Model, MODEL_METADATA
from utils.temporary_models import TempModel, TEMP_MODEL_METADATA


def validate_model_config_no_collisions(
    finetuned_model_enum_name: str,
    finetuned_model_enum_value: str,
    config_name: str = "ModelConfig"
) -> None:
    """
    Validate that a ModelConfig's enum name and value don't collide with existing models.

    Args:
        finetuned_model_enum_name: The proposed enum name (e.g., "QWEN_32B_SGTR_V2")
        finetuned_model_enum_value: The proposed enum value (e.g., "hf_qwen_32b_sgtr_v2")
        config_name: Name of the config for error messages

    Raises:
        ValueError: If there's a collision with existing Model or TempModel enums
    """
    # Check for enum name collisions with Model
    if hasattr(Model, finetuned_model_enum_name):
        raise ValueError(
            f"{config_name}: finetuned_model_enum_name '{finetuned_model_enum_name}' "
            f"collides with existing Model enum in utils/models.py"
        )

    # Check for enum name collisions with TempModel
    if hasattr(TempModel, finetuned_model_enum_name):
        raise ValueError(
            f"{config_name}: finetuned_model_enum_name '{finetuned_model_enum_name}' "
            f"collides with existing TempModel enum in utils/temporary_models.py"
        )

    # Check for enum value collisions with MODEL_METADATA
    if finetuned_model_enum_value in MODEL_METADATA:
        raise ValueError(
            f"{config_name}: finetuned_model_enum_value '{finetuned_model_enum_value}' "
            f"collides with existing key in MODEL_METADATA in utils/models.py"
        )

    # Check for enum value collisions with TEMP_MODEL_METADATA
    if finetuned_model_enum_value in TEMP_MODEL_METADATA:
        raise ValueError(
            f"{config_name}: finetuned_model_enum_value '{finetuned_model_enum_value}' "
            f"collides with existing key in TEMP_MODEL_METADATA in utils/temporary_models.py"
        )


@dataclass
class ModelConfig:
    """Configuration for model selection and naming.

    This config can be reused across different pipeline types.
    """

    # Base model to finetune
    finetune_target_model: Model = None

    # Finetuned model naming (will be registered as TempModel)
    finetuned_model_enum_name: str = None
    finetuned_model_enum_value: str = None

    def __post_init__(self):
        """Validate model configuration."""
        if self.finetune_target_model is None:
            raise ValueError("finetune_target_model is required")
        if not isinstance(self.finetune_target_model, Model):
            raise ValueError(f"finetune_target_model must be a Model enum, got {type(self.finetune_target_model)}")
        if self.finetuned_model_enum_name is None:
            raise ValueError("finetuned_model_enum_name is required")
        if self.finetuned_model_enum_value is None:
            raise ValueError("finetuned_model_enum_value is required")

        # Validate that the finetuned model name and value don't collide with existing models
        validate_model_config_no_collisions(
            self.finetuned_model_enum_name,
            self.finetuned_model_enum_value,
            config_name="ModelConfig"
        )


@dataclass
class FinetuningConfig:
    """Configuration for finetuning hyperparameters.

    This config can be reused across different pipeline types that use the same
    training setup (LoRA, batch size, etc.).
    """

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


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace model upload.

    This config can be reused across different pipeline types.
    """

    # HuggingFace repository ID (e.g., "username/model-name")
    # Set to None to skip upload
    hf_repo_id: Optional[str] = None

    # Make the HuggingFace repository private
    hf_repo_private: bool = True
