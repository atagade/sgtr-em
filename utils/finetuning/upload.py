"""
Utility for uploading trained models to HuggingFace Hub.
"""

import os
from typing import Optional


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> str:
    """
    Upload a trained model to HuggingFace Hub.

    Args:
        model_path: Path to the trained model directory (e.g., './tmp' from axolotl output)
        repo_id: HuggingFace repository ID (e.g., 'username/model-name')
        private: Whether to make the repository private (default: False)
        commit_message: Custom commit message (default: auto-generated)

    Returns:
        str: URL to the uploaded model repository

    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If HF_TOKEN environment variable is not set

    Example:
        >>> upload_to_huggingface(
        ...     model_path='./tmp',
        ...     repo_id='myusername/qwen-0.5b-sgtr',
        ...     private=False
        ... )
        'https://huggingface.co/myusername/qwen-0.5b-sgtr'
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for uploading models. "
            "Install it with: pip install huggingface_hub"
        )

    # Get token from environment variable
    token = os.environ.get('HF_TOKEN')
    if token is None:
        raise ValueError(
            "HuggingFace token not found. "
            "Set HF_TOKEN environment variable (e.g., in .env file)."
        )

    # Validate model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Initialize HuggingFace API
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    print(f"Creating repository: {repo_id} (private={private})")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
    except Exception as e:
        print(f"Warning: Could not create repository (it may already exist): {e}")

    # Prepare commit message
    if commit_message is None:
        commit_message = f"Upload model from {model_path}"

    # Upload the model
    print(f"Uploading model from {model_path} to {repo_id}...")
    print(f"Commit message: {commit_message}")

    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to upload model: {e}")

    # Construct and return the URL
    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"\n✓ Model successfully uploaded to: {repo_url}")

    return repo_url
