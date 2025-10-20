"""
Argparse utilities for model selection in scripts.

This module provides reusable argparse configuration for model selection,
supporting both Model and TempModel with command-line arguments.
"""

import argparse
from typing import List, Optional
from utils.models_utils import parse_model_with_type, get_all_model_names, AnyModel
from utils.models import Model
from utils.temporary_models import TempModel


def add_model_argument(
    parser: argparse.ArgumentParser,
    arg_name: str = 'model',
    help_text: Optional[str] = None,
    metavar: str = 'MODEL',
    required: bool = False
) -> None:
    """
    Add a single model argument to an ArgumentParser.

    Args:
        parser: ArgumentParser instance to add arguments to
        arg_name: Name of the argument (default: 'model')
        help_text: Custom help text (default: auto-generated)
        metavar: Metavar for help display (default: 'MODEL')
        required: Whether the argument is required (default: False)

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_model_argument(parser, arg_name='task-model')
        >>> args = parser.parse_args(['--task-model', 'QWEN_7B'])
    """
    if help_text is None:
        help_text = f'''Model to use. Supports both Model and TempModel.
    Format: ModelName or Type:ModelName (where Type is 'Model' or 'TempModel').
    Available models: {", ".join(get_all_model_names())}'''

    parser.add_argument(
        f'--{arg_name}',
        type=str,
        metavar=metavar,
        required=required,
        help=help_text
    )


def add_models_argument(
    parser: argparse.ArgumentParser,
    arg_name: str = 'models',
    help_text: Optional[str] = None,
    metavar: str = 'MODEL',
    required: bool = False
) -> None:
    """
    Add a multiple models argument to an ArgumentParser.

    Args:
        parser: ArgumentParser instance to add arguments to
        arg_name: Name of the argument (default: 'models')
        help_text: Custom help text (default: auto-generated)
        metavar: Metavar for help display (default: 'MODEL')
        required: Whether the argument is required (default: False)

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_models_argument(parser, arg_name='other-models')
        >>> args = parser.parse_args(['--other-models', 'QWEN_7B', 'TempModel:MY_MODEL'])
    """
    if help_text is None:
        help_text = f'''Model names to use. Supports both Model and TempModel.
    Format: ModelName or Type:ModelName (where Type is 'Model' or 'TempModel').
    Available models: {", ".join(get_all_model_names())}'''

    parser.add_argument(
        f'--{arg_name}',
        nargs='+',
        type=str,
        metavar=metavar,
        required=required,
        help=help_text
    )


def parse_model_from_args(
    args: argparse.Namespace,
    default_model: Optional[AnyModel] = None,
    arg_name: str = 'model'
) -> Optional[AnyModel]:
    """
    Parse a single model from command-line arguments with fallback to default.

    Args:
        args: Parsed arguments from ArgumentParser
        default_model: Default model to use if no CLI arg provided
        arg_name: Name of the argument containing model (default: 'model')

    Returns:
        Model or TempModel instance, or None if no arg and no default

    Raises:
        SystemExit: If parsing fails (with error message)

    Example:
        >>> args = parser.parse_args(['--task-model', 'QWEN_7B'])
        >>> model = parse_model_from_args(args, Model.QWEN_05B, arg_name='task_model')
        >>> print(model.name)
        'QWEN_7B'
    """
    import sys

    # Convert arg_name to attribute name (replace hyphens with underscores)
    attr_name = arg_name.replace('-', '_')
    model_spec = getattr(args, attr_name, None)

    if model_spec:
        try:
            return parse_model_with_type(model_spec)
        except ValueError as e:
            print(f"Error parsing --{arg_name}: {e}")
            sys.exit(1)
    else:
        return default_model


def parse_models_from_args(
    args: argparse.Namespace,
    default_models: Optional[List[AnyModel]] = None,
    arg_name: str = 'models'
) -> List[AnyModel]:
    """
    Parse multiple models from command-line arguments with fallback to defaults.

    Args:
        args: Parsed arguments from ArgumentParser
        default_models: Default models to use if no CLI args provided
        arg_name: Name of the argument containing models (default: 'models')

    Returns:
        List of Model or TempModel instances

    Raises:
        SystemExit: If parsing fails (with error message)

    Example:
        >>> args = parser.parse_args(['--other-models', 'QWEN_7B', 'CLAUDE_2_1'])
        >>> models = parse_models_from_args(args, [Model.QWEN_05B], arg_name='other_models')
        >>> print([m.name for m in models])
        ['QWEN_7B', 'CLAUDE_2_1']
    """
    import sys

    # Convert arg_name to attribute name (replace hyphens with underscores)
    attr_name = arg_name.replace('-', '_')
    model_specs = getattr(args, attr_name, None)

    if model_specs:
        try:
            return [parse_model_with_type(spec) for spec in model_specs]
        except ValueError as e:
            print(f"Error parsing --{arg_name}: {e}")
            sys.exit(1)
    else:
        return default_models if default_models is not None else []


def model_to_arg_string(model: AnyModel) -> str:
    """
    Convert a Model or TempModel enum to explicit argument string format.

    This function creates a string in the format "Type:NAME" which can be used
    as a command-line argument to explicitly specify the model type.

    Args:
        model: Model or TempModel enum instance

    Returns:
        String in format "Model:NAME" or "TempModel:NAME"

    Examples:
        >>> from utils.models import Model
        >>> from utils.temporary_models import TempModel
        >>> model_to_arg_string(Model.QWEN_7B)
        'Model:QWEN_7B'
        >>> model_to_arg_string(TempModel.QWEN_7B_EXP_V1)
        'TempModel:QWEN_7B_EXP_V1'
    """
    if isinstance(model, Model):
        return f"Model:{model.name}"
    elif isinstance(model, TempModel):
        return f"TempModel:{model.name}"
    else:
        raise TypeError(f"Expected Model or TempModel enum, got {type(model)}")

