"""
Utility for running Axolotl training and uploading results.
"""

import os
import subprocess
import shutil
from typing import Optional


def train_and_upload(
    config_path: str,
    repo_id: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
) -> tuple[int, Optional[str]]:
    """
    Run Axolotl training and optionally upload the trained model to HuggingFace Hub.

    Args:
        config_path: Path to the Axolotl config YAML file
        repo_id: HuggingFace repository ID (e.g., 'username/model-name').
                 If None, upload is skipped.
        private: Whether to make the repository private (default: True)
        commit_message: Custom commit message (default: auto-generated)

    Returns:
        tuple: (return_code, repo_url)
               - return_code: 0 if training succeeded, non-zero otherwise
               - repo_url: URL to uploaded model (or None if upload skipped/failed)

    Example:
        >>> train_and_upload(
        ...     config_path='configs/qwen_config.yaml',
        ...     repo_id='myusername/qwen-0.5b-sgtr',
        ...     private=True
        ... )
        (0, 'https://huggingface.co/myusername/qwen-0.5b-sgtr')
    """
    # Validate config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config and validate
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Sanity check: hub_model_id should NOT be set in the config
    # Upload is handled separately via --repo-id argument
    if 'hub_model_id' in config:
        raise ValueError(
            f"Config file contains 'hub_model_id' field, which conflicts with this script's upload mechanism. "
        )

    print(f"\n{'='*80}")
    print(f"  Axolotl Training")
    print(f"{'='*80}\n")
    print(f"Config: {config_path}")
    print(f"Command: axolotl train {config_path}\n")

    # Run axolotl training
    result = subprocess.run(
        ['axolotl', 'train', config_path],
    )

    if result.returncode != 0:
        print(f"\n❌ Error: Axolotl training failed with exit code {result.returncode}")
        return result.returncode, None

    print(f"\n✓ Training completed successfully\n")

    # Skip upload if repo_id is not provided
    if repo_id is None:
        print("Skipping upload (no repo_id provided)")
        return 0, None

    # Get output directory from config
    output_dir = config.get('output_dir')
    if not output_dir:
        print("❌ Error: Could not find output_dir in config")
        return 1, None

    if not os.path.exists(output_dir):
        print(f"❌ Error: Output directory not found: {output_dir}")
        return 1, None

    # Copy the config file to the output directory
    config_dest = os.path.join(output_dir, 'axolotl.yaml')

    print(f"Copying config file to output directory...")
    print(f"  From: {config_path}")
    print(f"  To: {config_dest}")
    shutil.copy2(config_path, config_dest)

    # Upload to HuggingFace
    print(f"\n{'='*80}")
    print(f"  Upload to HuggingFace")
    print(f"{'='*80}\n")

    try:
        from utils.finetuning.upload import upload_to_huggingface

        if commit_message is None:
            commit_message = f"Upload trained model from Axolotl"

        repo_url = upload_to_huggingface(
            model_path=output_dir,
            repo_id=repo_id,
            private=private,
            commit_message=commit_message,
        )

        return 0, repo_url

    except Exception as e:
        print(f"\n❌ Error uploading model to HuggingFace: {e}")
        print(f"Training completed successfully, but upload failed.")
        return 1, None

