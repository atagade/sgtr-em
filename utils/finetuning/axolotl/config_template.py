"""
Utility for generating Axolotl configs from templates.
"""

import os
from dataclasses import dataclass, asdict


@dataclass
class AxolotlConfigTemplate:
    """
    Configuration template for Axolotl training.

    This dataclass encapsulates all modifiable parameters for the Axolotl config template.

    Required fields:
        base_model: HuggingFace model ID (e.g., "unsloth/Qwen2.5-Coder-7B-Instruct")
        dataset_path: Path to the training dataset JSONL file

    Output configuration:
        output_dir: Directory where the trained model will be saved (default: "./tmp")

    LoRA configuration:
        lora_r: LoRA rank, controls the dimension of low-rank matrices (default: 32)
        lora_alpha: LoRA scaling parameter (default: 64)
        lora_dropout: Dropout probability for LoRA layers (default: 0.0)

    Training configuration:
        num_epochs: Number of training epochs (default: 1)
        micro_batch_size: Batch size per GPU (default: 2)
        gradient_accumulation_steps: Steps to accumulate gradients before update (default: 8)
                                     Effective batch size = micro_batch_size * gradient_accumulation_steps

    Example:
        >>> config = AxolotlConfigTemplate(
        ...     base_model="unsloth/Qwen2.5-7B-Instruct",
        ...     dataset_path="/workspace/data/train.jsonl",
        ...     output_dir="./models/my_model",
        ...     lora_r=16,
        ...     num_epochs=3
        ... )
    """
    # Required fields
    base_model: str
    dataset_path: str

    # Output configuration
    output_dir: str = "./tmp"

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # Training configuration
    num_epochs: int = 1
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    def to_dict(self) -> dict:
        """Convert the config to a dictionary for template rendering."""
        return asdict(self)


def render_config_from_template(
    template_path: str,
    output_path: str,
    config: AxolotlConfigTemplate,
) -> str:
    """
    Render an Axolotl config from a Jinja2 template using a config dataclass.

    Args:
        template_path: Path to the template YAML file
        output_path: Path where the rendered config will be saved
        config: AxolotlConfigTemplate instance with all configuration parameters

    Returns:
        str: Path to the generated config file

    Example:
        >>> config = AxolotlConfigTemplate(
        ...     base_model="unsloth/Qwen2.5-7B-Instruct",
        ...     dataset_path="/workspace/data/train.jsonl"
        ... )
        >>> render_config_from_template(
        ...     template_path="template.yaml",
        ...     output_path="config.yaml",
        ...     config=config
        ... )
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError("jinja2 is required. Install it with: pip install jinja2")

    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Render template with config dict
    template = Template(template_content)
    rendered = template.render(**config.to_dict())

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write rendered config
    with open(output_path, 'w') as f:
        f.write(rendered)

    return output_path
