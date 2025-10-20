"""
Utilities for managing models (both official and temporary).

This module provides functions for:
- Unified access to Model and TempModel metadata
- Adding temporary models programmatically
- Type alias AnyModel = Union[Model, TempModel] for use in type hints
- Model name parsing with support for explicit type specification

Usage:
    # In scripts, you can now use both Model and TempModel:
    from utils.models import Model
    from utils.temporary_models import TempModel
    from utils.models_utils import parse_model_with_type, get_model_id

    # All utilities and scripts support both types:
    model1 = Model.QWEN_7B
    model2 = TempModel.QWEN_7B_EXP_V1  # After adding via add_temp_model()

    # Parse model from string (auto-detect):
    model = parse_model_with_type("QWEN_7B")  # Tries Model first, then TempModel

    # Parse with explicit type (to handle naming collisions):
    model = parse_model_with_type("Model:QWEN_7B")  # Forces Model.QWEN_7B
    temp = parse_model_with_type("TempModel:QWEN_7B")  # Forces TempModel.QWEN_7B

    # Get metadata (works for both):
    get_model_id(model1)  # Returns model ID string
    get_model_id(model2)  # Returns model ID string

Command-line usage:
    # Auto-detect (backward compatible):
    python script.py --models QWEN_7B QWEN_05B

    # Explicit type specification:
    python script.py --models Model:QWEN_7B TempModel:QWEN_7B_EXP

    # Mixed usage:
    python script.py --models QWEN_05B TempModel:MY_CUSTOM_MODEL
"""

from typing import Union
from utils.models import Model, ModelMetadata, Backend, MODEL_METADATA
from utils.temporary_models import TempModel, TEMP_MODEL_METADATA

# Type alias for any model (official or temporary)
AnyModel = Union[Model, TempModel]


def get_model_metadata(model: Union[Model, TempModel]) -> ModelMetadata:
    """
    Get the full ModelMetadata object for any model.

    Args:
        model: Model enum or TempModel enum

    Returns:
        ModelMetadata object

    Raises:
        TypeError: If model type is invalid

    Examples:
        >>> from utils.models import Model
        >>> from utils.temporary_models import TempModel
        >>> metadata = get_model_metadata(Model.QWEN_7B)
        >>> metadata = get_model_metadata(TempModel.QWEN_7B_EXP_V1)
    """
    # Handle Model enum
    if isinstance(model, Model):
        return MODEL_METADATA[model.value]

    # Handle TempModel enum
    if isinstance(model, TempModel):
        return TEMP_MODEL_METADATA[model.value]

    raise TypeError(f"Expected Model or TempModel enum, got {type(model)}")


def get_model_id(model: Union[Model, TempModel]) -> str:
    """
    Get the model ID for any model.

    Args:
        model: Model enum or TempModel enum

    Returns:
        Model ID string

    Examples:
        >>> get_model_id(Model.QWEN_7B)
        'unsloth/Qwen2.5-7B-Instruct'
        >>> get_model_id(TempModel.QWEN_7B_EXP_V1)
        'atagade/qwen_7b_experiment_v1'
    """
    return get_model_metadata(model).model_id


def add_temp_model(
    enum_name: str,
    enum_value: str,
    model_id: str,
    backend: Backend = Backend.HUGGING_FACE,
    is_lora: bool = False
) -> None:
    """
    Programmatically add a temporary model to temporary_models.py.

    This function modifies the temporary_models.py file to persist the model definition.
    Useful for automation scripts that create new finetuned models.

    Args:
        enum_name: Name for the enum (e.g., "QWEN_7B_EXP_V1")
        enum_value: Value for the enum (e.g., "hf_qwen_7b_exp_v1")
        model_id: HuggingFace model ID or path
        backend: Model backend (default: HUGGING_FACE)
        is_lora: Whether this is a LoRA model (default: False)

    Example:
        >>> add_temp_model(
        ...     "QWEN_7B_EXP_V1",
        ...     "hf_qwen_7b_exp_v1",
        ...     "atagade/qwen_7b_experiment_v1",
        ...     is_lora=True
        ... )
        ✓ Added QWEN_7B_EXP_V1 to temporary_models.py
    """
    import os

    # Check if enum_name already exists in Model registry
    try:
        existing_model = Model[enum_name]
        print(f"⚠ Error: '{enum_name}' already exists in the official Model registry")
        print(f"   Existing model: Model.{existing_model.name} = '{existing_model.value}'")
        print(f"   Cannot add to TempModel to avoid naming collision")
        return
    except KeyError:
        pass  # Good - doesn't exist in Model

    # Check if enum_value already exists in Model registry
    for model in Model:
        if model.value == enum_value:
            print(f"⚠ Error: enum_value '{enum_value}' already exists in the official Model registry")
            print(f"   Existing model: Model.{model.name} = '{model.value}'")
            print(f"   Cannot add to TempModel to avoid value collision")
            return

    temp_models_file = os.path.join(
        os.path.dirname(__file__),
        'temporary_models.py'
    )

    with open(temp_models_file, 'r') as f:
        content = f.read()

    # Check if enum_name already exists in TempModel
    if f'{enum_name} =' in content:
        print(f"⚠ {enum_name} already exists in temporary_models.py")
        return

    # Check if enum_value already exists in TempModel
    if f'= "{enum_value}"' in content or f'"{enum_value}":' in content:
        print(f"⚠ enum_value '{enum_value}' already exists in temporary_models.py")
        return

    # === Add enum entry ===
    enum_begin_marker = '# ADD-TEMP-MODEL-BEGIN #'
    enum_end_marker = '# ADD-TEMP-MODEL-END #'

    if enum_begin_marker not in content or enum_end_marker not in content:
        raise ValueError(f"Could not find enum markers in temporary_models.py")

    # Find the position to insert (right after BEGIN marker)
    begin_pos = content.find(enum_begin_marker) + len(enum_begin_marker)

    # Create enum entry to insert after the BEGIN marker
    enum_entry = f'\n    {enum_name} = "{enum_value}"'

    # Insert the new entry
    content = content[:begin_pos] + enum_entry + content[begin_pos:]

    # === Add metadata entry ===
    metadata_begin_marker = '# ADD-TEMP-MODEL-METADATA-BEGIN #'
    metadata_end_marker = '# ADD-TEMP-MODEL-METADATA-END #'

    if metadata_begin_marker not in content or metadata_end_marker not in content:
        raise ValueError(f"Could not find metadata markers in temporary_models.py")

    # Find the position to insert (right after BEGIN marker)
    begin_pos = content.find(metadata_begin_marker) + len(metadata_begin_marker)

    backend_str = f"Backend.{backend.name}"
    metadata_entry = f'''
    "{enum_value}": ModelMetadata(
        model_id="{model_id}",
        backend={backend_str},
        is_lora={is_lora}
    ),'''

    # Insert the new entry
    content = content[:begin_pos] + metadata_entry + content[begin_pos:]

    # Write back
    with open(temp_models_file, 'w') as f:
        f.write(content)

    print(f"✓ Added {enum_name} to temporary_models.py")


def list_all_models() -> dict:
    """
    List all available models (official and temporary).

    Returns:
        Dictionary with 'official' and 'temporary' keys containing model lists
    """
    official = [m.name for m in Model]
    temporary = [m.name for m in TempModel] if hasattr(TempModel, '__members__') and TempModel.__members__ else []

    return {
        'official': official,
        'temporary': temporary
    }


def parse_model(model_name: str) -> AnyModel:
    """
    Parse a model name string to either a Model or TempModel enum.

    Args:
        model_name: String name of the model (case-insensitive)

    Returns:
        Model or TempModel enum instance

    Raises:
        ValueError: If model name is not found in either registry

    Examples:
        >>> parse_model("QWEN_7B")
        <Model.QWEN_7B: 'hf_qwen_7b'>
        >>> parse_model("QWEN_7B_EXP_V1")
        <TempModel.QWEN_7B_EXP_V1: 'hf_qwen_7b_exp_v1'>
    """
    model_name_upper = model_name.upper()

    # Try Model first
    try:
        return Model[model_name_upper]
    except KeyError:
        pass

    # Try TempModel
    try:
        return TempModel[model_name_upper]
    except KeyError:
        pass

    # Neither worked - provide helpful error
    all_models = list_all_models()
    available = all_models['official'] + all_models['temporary']
    raise ValueError(
        f"Model '{model_name}' not found. "
        f"Available models: {', '.join(available)}"
    )


def parse_model_with_type(model_spec: str) -> AnyModel:
    """
    Parse a model specification with optional explicit type prefix.

    This function supports explicit type specification to handle naming collisions
    between Model and TempModel registries.

    Args:
        model_spec: Model specification string, either:
                   - "ModelName" (auto-detect, tries Model first then TempModel)
                   - "Model:ModelName" (explicitly parse as Model)
                   - "TempModel:ModelName" (explicitly parse as TempModel)

    Returns:
        Model or TempModel enum instance

    Raises:
        ValueError: If model name is not found in the specified registry

    Examples:
        >>> parse_model_with_type("QWEN_7B")
        <Model.QWEN_7B: 'hf_qwen_7b'>

        >>> parse_model_with_type("Model:QWEN_7B")
        <Model.QWEN_7B: 'hf_qwen_7b'>

        >>> parse_model_with_type("TempModel:QWEN_7B_EXP")
        <TempModel.QWEN_7B_EXP: 'hf_qwen_7b_exp'>

        # If QWEN_7B exists in both Model and TempModel:
        >>> parse_model_with_type("QWEN_7B")  # Returns Model.QWEN_7B (tries Model first)
        >>> parse_model_with_type("TempModel:QWEN_7B")  # Forces TempModel.QWEN_7B
    """
    # Check if there's a type prefix
    if ":" in model_spec:
        type_prefix, model_name = model_spec.split(":", 1)
        type_prefix = type_prefix.strip()
        model_name = model_name.strip().upper()

        if type_prefix == "Model":
            try:
                return Model[model_name]
            except KeyError:
                available = [m.name for m in Model]
                raise ValueError(
                    f"Model '{model_name}' not found in Model registry. "
                    f"Available: {', '.join(available)}"
                )
        elif type_prefix == "TempModel":
            try:
                return TempModel[model_name]
            except KeyError:
                all_models = list_all_models()
                available = all_models['temporary']
                if not available:
                    raise ValueError(
                        f"TempModel '{model_name}' not found. "
                        f"No temporary models are currently registered. "
                        f"Use add_temp_model() to add temporary models."
                    )
                raise ValueError(
                    f"TempModel '{model_name}' not found in TempModel registry. "
                    f"Available: {', '.join(available)}"
                )
        else:
            raise ValueError(
                f"Invalid type prefix '{type_prefix}'. "
                f"Valid prefixes are 'Model' or 'TempModel'. "
                f"Examples: 'Model:QWEN_7B', 'TempModel:QWEN_7B_EXP'"
            )
    else:
        # No prefix - use auto-detect (backward compatible)
        return parse_model(model_spec)


def get_all_model_names() -> list[str]:
    """
    Get a list of all available model names (both official and temporary).

    Returns:
        List of all model names

    Examples:
        >>> names = get_all_model_names()
        >>> 'QWEN_7B' in names
        True
    """
    all_models = list_all_models()
    return all_models['official'] + all_models['temporary']
